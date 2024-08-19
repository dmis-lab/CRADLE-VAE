from typing import Dict, List, Optional, Tuple

import torch
from torch.distributions import Bernoulli, Distribution, Normal

from cradle_vae.models.utils.mlp import LIKELIHOOD_KEY_DTYPE, get_likelihood_mlp


class cradle_vae_Model(torch.nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_treatments: int,
        n_phenos: int,
        n_qc: int,
        mask_prior_prob: float,
        embedding_prior_scale: float,
        likelihood_key: LIKELIHOOD_KEY_DTYPE,
        decoder_n_layers: int,
        decoder_n_hidden: int,
        sum_mode: bool = True,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_treatments = n_treatments
        self.n_phenos = n_phenos
        self.n_qc = n_qc
        self.likelihood_key = likelihood_key
        self.decoder_n_layers = decoder_n_layers
        self.decoder_n_hidden = decoder_n_hidden
        self.sum_mode = sum_mode

        self.register_buffer("p_E_loc", torch.zeros((n_treatments, n_latent)))
        self.register_buffer(
            "p_E_scale", embedding_prior_scale * torch.ones((n_treatments, n_latent))
        )
        self.register_buffer(
            "p_mask_probs", mask_prior_prob * torch.ones((n_treatments, n_latent))
        )
        
        self.register_buffer("p_qc_E_loc", torch.zeros((n_qc, n_latent)))
        self.register_buffer(
            "p_qc_E_scale", embedding_prior_scale * torch.ones((n_qc, n_latent))
        )

        self.decoder = get_likelihood_mlp(
            likelihood_key=likelihood_key,
            n_input=n_latent if sum_mode else n_latent + n_latent + n_latent,
            n_output=n_phenos,
            n_layers=decoder_n_layers+1,
            n_hidden=decoder_n_hidden,
            use_batch_norm=False,
            activation_fn=torch.nn.LeakyReLU,
        )

    def get_var_keys(self) -> List[str]:
        return ["z_basal", "E", "mask", "qc_E"] # , "z_noise"

    def forward(
        self,
        D: torch.Tensor,
        qc : Optional[torch.Tensor],
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
    ) -> Tuple[Dict[str, Distribution], Dict[str, torch.Tensor]]:
        """
        Sample from generative process, conditioned on D and other condition_values

        Parameters
        ----------
        D: dosages (n, n_perturbations)
        condition_values: optional dictionary of conditioning values (missing variables
                are sampled from prior)
        n_particles: number of samples to draw from generative distribution
                for each observed dosage

        Returns
        -------
        Tuple with dictionaries of generative distributions and samples, each
        with batch size n_particles. Keys are strings specifying variables
        """
        n = D.shape[0]
        device = D.device

        if condition_values is None:
            condition_values = dict()

        # define generative distribution
        generative_dists = {}
        generative_dists["p_z_basal"] = Normal(
            torch.zeros((n, self.n_latent)).to(device),
            torch.ones((n, self.n_latent)).to(device),
        )

        generative_dists["p_E"] = Normal(self.p_E_loc, self.p_E_scale)
        generative_dists["p_mask"] = Bernoulli(logits=torch.logit(self.p_mask_probs))
        generative_dists["p_qc_E"] = Normal(self.p_qc_E_loc, self.p_qc_E_scale)

        # get samples (sampled from priors if not in conditioning set)
        samples = {}
        for k in self.get_var_keys():
            if condition_values.get(k) is not None:
                value = condition_values[k]
                # expand to align with n_particles
                if type(value) is dict:
                    samples[k] = {'win':value['win'], 'lose':value['lose']}
                    continue
                if len(value.shape) == 2:
                    value = value.unsqueeze(0).expand((n_particles, -1, -1))
                samples[k] = value
            else:
                samples[k] = generative_dists[f"p_{k}"].sample((n_particles,))

        # forward model
        # shape: (n_particles, n, n_latent)
        if qc is not None:
            if self.sum_mode:
                z_good = samples["z_basal"] + torch.matmul(D, samples["E"] * samples["mask"])
                z_bad  = samples["z_basal"] + torch.matmul(D, samples["E"] * samples["mask"]) + torch.matmul(torch.ones(qc.shape).to(device), samples["qc_E"])
            else:
                z_good = torch.cat([samples["z_basal"], torch.matmul(D, samples["E"] * samples["mask"]), torch.matmul(torch.zeros(qc.shape).to(device), samples["qc_E"])], dim=-1)
                z_bad  = torch.cat([samples["z_basal"], torch.matmul(D, samples["E"] * samples["mask"]), torch.matmul(torch.ones(qc.shape).to(device), samples["qc_E"])], dim=-1)

            if self.likelihood_key != "library_nb":
                generative_dists["p_x_good"] = self.decoder(z_good)
                generative_dists["p_x_bad"]  = self.decoder(z_bad)
            else:
                generative_dists["p_x_good"] = self.decoder(z_good, condition_values["library_size"])
                generative_dists["p_x_bad"]  = self.decoder(z_bad, condition_values["library_size"])

            samples["x_good"] = generative_dists["p_x_good"].sample()
            samples["x_bad"]  = generative_dists["p_x_bad"].sample()
            
        else:
            z = samples["z_basal"] + torch.matmul(D, samples["E"] * samples["mask"])

            if self.likelihood_key != "library_nb":
                generative_dists["p_x"] = self.decoder(z)
            else:
                generative_dists["p_x"] = self.decoder(z, condition_values["library_size"])

            samples["x"] = generative_dists["p_x"].sample()

        return generative_dists, samples
