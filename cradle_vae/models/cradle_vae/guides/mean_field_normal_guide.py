from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from cradle_vae.data.utils.perturbation_datamodule import (
    ObservationNormalizationStatistics,
)
from cradle_vae.models.utils.gumbel_softmax_bernoulli import (
    GumbelSoftmaxBernoulliStraightThrough,
)
from cradle_vae.models.utils.mlp import get_likelihood_mlp
from cradle_vae.models.utils.normalization import get_normalization_module


class cradle_vae_MeanFieldNormalGuide(nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_treatments: int,
        n_qc:int,
        n_phenos: int,
        basal_encoder_n_layers: int,
        basal_encoder_n_hidden: int,
        basal_encoder_input_normalization: Optional[
            Literal["standardize", "log_standardize"]
        ],
        x_normalization_stats: Optional[ObservationNormalizationStatistics],
        embedding_loc_init_scale: float = 0,
        embedding_scale_init: float = 1,
        mask_init_logits: float = 0,
        gs_temperature: float = 1,
        mean_field_encoder: bool = False,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_treatments = n_treatments
        self.n_qc = n_qc
        self.n_phenos = n_phenos

        self.basal_encoder_input_normalization = basal_encoder_input_normalization
        self.x_normalization_stats = x_normalization_stats

        self.mean_field_encoder = mean_field_encoder

        self.param_dict = torch.nn.ParameterDict()
        self.param_dict["q_mask_logits"] = torch.nn.Parameter(
            mask_init_logits * torch.ones((n_treatments, n_latent))
        )

        self.param_dict["q_E_loc"] = torch.nn.Parameter(
            embedding_loc_init_scale * torch.randn((n_treatments, n_latent))
        )
        self.param_dict["q_E_log_scale"] = torch.nn.Parameter(
            np.log(embedding_scale_init) * torch.ones((n_treatments, n_latent))
        )

        self.param_dict["q_qc_mask_logits"] = torch.nn.Parameter(
            mask_init_logits * torch.ones((n_qc, n_latent))
        )

        self.param_dict["q_qc_E_loc"] = torch.nn.Parameter(
            embedding_loc_init_scale * torch.randn((n_qc, n_latent))
        )
        self.param_dict["q_qc_E_log_scale"] = torch.nn.Parameter(
            np.log(embedding_scale_init) * torch.ones((n_qc, n_latent))
        )

        if self.basal_encoder_input_normalization is None:
            self.normalization_module = None
        else:
            assert x_normalization_stats is not None, "Missing x_normalization_stats"
            self.normalization_module = get_normalization_module(
                key=self.basal_encoder_input_normalization,
                normalization_stats=x_normalization_stats,
            )

        self.z_basal_encoder = get_likelihood_mlp(
            likelihood_key="normal",
            n_input=n_phenos if mean_field_encoder else n_phenos + n_latent + n_latent,
            n_output=n_latent,
            n_layers=basal_encoder_n_layers,
            n_hidden=basal_encoder_n_hidden,
            use_batch_norm=False,
        )

        self.register_buffer("gs_temperature", gs_temperature * torch.ones((1,)))

        self.var_eps = 1e-4

    def get_var_keys(self):
        var_keys = ["z_basal", "E", "mask", "qc_E"]
        return var_keys

    def forward(
        self,
        X: Optional[torch.Tensor] = None,
        D: Optional[torch.Tensor] = None,
        qc: Optional[torch.Tensor] = None,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
    ) -> Tuple[Dict[str, torch.distributions.Distribution], Dict[str, torch.Tensor]]:
        """
        Compute q(z_basal, M, E | X, D) = q(M) q(E) q(z_basal | M, E, X, D) and sample

        Parameters
        ----------
        X: observations
        D: perturbation dosages
        condition_values: values for random variables to condition on
        n_particles: number of samples to take from q

        Returns
        -------
        Tuple of guide distribution dict and guide samples dict
        Each has string keys for variable names
        """
        if condition_values is None:
            condition_values = dict()

        guide_distributions: Dict[str, torch.distributions.Distribution] = {}
        guide_samples: Dict[str, torch.Tensor] = {}

        guide_distributions["q_mask"] = GumbelSoftmaxBernoulliStraightThrough(
            temperature=self.gs_temperature,
            logits=self.param_dict["q_mask_logits"],
        )

        guide_distributions["q_E"] = Normal(
            self.param_dict["q_E_loc"],
            torch.exp(self.param_dict["q_E_log_scale"]) + self.var_eps,
        )

        guide_distributions["q_qc_mask"] = GumbelSoftmaxBernoulliStraightThrough(
            temperature=self.gs_temperature,
            logits=self.param_dict["q_qc_mask_logits"],
        )

        guide_distributions["q_qc_E"] = Normal(
            self.param_dict["q_qc_E_loc"],
            torch.exp(self.param_dict["q_qc_E_log_scale"]) + self.var_eps,
        )

        for k in ["mask", "E", "qc_mask", "qc_E"]:
            if k not in condition_values:
                guide_samples[k] = guide_distributions[f"q_{k}"].rsample((n_particles,))
            else:
                guide_samples[k] = condition_values[k]

        if X is not None and D is not None:
            # compute q(z_basal|x) if mean field encoder, q(z_basal | x, M, E) if not
            encoder_input = X

            # normalize input for z_basal encoder
            if self.normalization_module is not None:
                encoder_input = self.normalization_module(encoder_input)

            # expand encoder_input on dim 0 to match n_particles
            encoder_input = torch.unsqueeze(encoder_input, dim=0).expand(
                n_particles, -1, -1
            )

            if not self.mean_field_encoder:
                # q(z_basal|x, M, E) by concatenating estimated latent offsets to x
                latent_offset = torch.matmul(
                    D, guide_samples["mask"] * guide_samples["E"]
                )
                latent_qc_offset = torch.matmul(
                    qc, guide_samples["qc_mask"] * guide_samples["qc_E"]
                )
                encoder_input = torch.cat([encoder_input, latent_offset, latent_qc_offset], dim=-1)

            guide_distributions["q_z_basal"] = self.z_basal_encoder(encoder_input)
            guide_samples["z_basal"] = guide_distributions[
                "q_z_basal"
            ].rsample()  # already n_particles

        if "z_basal" in condition_values:
            guide_samples["z_basal"] = condition_values["z_basal"]

        return guide_distributions, guide_samples
