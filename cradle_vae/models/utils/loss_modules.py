from typing import Dict, Iterable, Optional

import numpy as np
import torch



class PerturbationPlatedELBOLossModule(torch.nn.Module):
    """
    Computes ELBO with option to specify variables that are on perturbation-based
    plates (eg perturbation embeddings that are shared between samples that receive
    a given perturbation)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        guide: torch.nn.Module,
        local_variables: Optional[Iterable[str]] = None,
        perturbation_plated_variables: Optional[Iterable[str]] = None,
        pos_weight: int = None,
        alpha: int = None,
        beta: int = 1,
    ):
        super().__init__()
        local_variables = list(local_variables) if local_variables is not None else []
        perturbation_plated_variables = (
            list(perturbation_plated_variables)
            if perturbation_plated_variables is not None
            else []
        )

        assert sorted(model.get_var_keys()) == sorted(
            guide.get_var_keys()
        ), "Mismatch in model and guide variables"

        # make sure that all variables are specified as a local variable or a
        # perturbation plated variable
        variables = local_variables + perturbation_plated_variables
        assert sorted(list(model.get_var_keys())) == sorted(
            variables
        ), "Mismatch between model variables and variables specified to loss module"

        self.model = model
        self.guide = guide

        self.local_variables = local_variables
        self.perturbation_plated_variables = perturbation_plated_variables

        self.pos_weight = pos_weight
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        X: torch.Tensor,
        D: torch.Tensor,
        qc: torch.Tensor,
        cfX: torch.Tensor,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
    ):
        if condition_values is None:
            condition_values = dict()

        pred_qc=None


        guide_dists, guide_samples = self.guide(
            X=X,
            D=D,
            qc=qc,
            n_particles=n_particles,
        )

        for k, v in guide_samples.items():
            condition_values[k] = v

        if "qm" in self.model._get_name():
            model_dists, model_samples = self.model(
                D=D,
                qc=qc,
                condition_values=condition_values,
                n_particles=n_particles,
            )

        else:
            model_dists, model_samples = self.model(
                D=D,
                qc=qc,
                condition_values=condition_values,
                n_particles=n_particles,
            )

        # cf_out = (qc)*model_dists['p_x_good'].sample().median(-3)[0] + (1-qc)*model_dists['p_x_bad'].sample().median(-3)[0]
        # cf_out = (qc)*model_dists['p_x_good'].mean.mean(-3) + (1-qc)*model_dists['p_x_bad'].mean.mean(-3)
        cf_guide_dists, cf_guide_samples = self.guide(
            X=cfX,
            D=D,
            qc=(1-qc),
            n_particles=n_particles,
        )
        # cf_guide_dists = None
        # cf_guide_samples = None

        return guide_dists, model_dists, model_samples, cf_guide_dists, cf_guide_samples, pred_qc
        
        

    def loss(
        self,
        X: torch.Tensor,
        D: torch.Tensor,
        D_obs_counts: torch.Tensor,
        qc: torch.Tensor,
        qc_obs_counts: torch.Tensor,
        cfX: list = None,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
        test: bool = False,
        x_var_info = None,
    ):
        guide_dists, model_dists, samples, cf_guide_dists, cf_guide_samples, pred_qc = self.forward(
            X=X,
            D=D,
            cfX=cfX,
            qc=qc,
            condition_values=condition_values,
            n_particles=n_particles,
        )

        loss_terms = {}
        
        p_x_good_log_p = model_dists['p_x_good'].log_prob(X)
        p_x_bad_log_p = model_dists['p_x_bad'].log_prob(X)
        
        alpha = self.alpha
        if alpha == None:
            alpha = qc.sum()/qc.shape[0]
        if qc.shape[1] > 1:
            qc = qc.max(axis=1)[0].unsqueeze(-1)
        p_x_real_log_p = alpha*((1-qc)*p_x_good_log_p) + (1-alpha)*(qc*p_x_bad_log_p)
        loss_terms["reconstruction"] = p_x_real_log_p.sum(-1)

        idx = torch.nonzero(cfX.sum(1) != 0).squeeze(1)
        kl_loss = self.beta * self.kldiv_normal(
            cf_guide_dists['q_z_basal'].mean[:,idx,:], 
            cf_guide_dists['q_z_basal'].stddev[:,idx,:],
            guide_dists['q_z_cf_basal'].mean[:,idx,:], 
            guide_dists['q_z_cf_basal'].stddev[:,idx,:])
        kl_loss = kl_loss.sum()

        for k in guide_dists.keys():

            var_key = k[2:]  # drop 'q_'
            if k == 'q_z_cf_basal':
                continue

            # score sample under prior
            loss_term = model_dists[f"p_{var_key}"].log_prob(samples[var_key])
            loss_term = loss_term - guide_dists[f"q_{var_key}"].log_prob(samples[var_key])
            if k == 'z_basal':
                loss_term = loss_term

            if var_key in self.perturbation_plated_variables:
                # reweight perturbation plated variables
                if 'qc' in var_key:
                    loss_term = self._compute_reweighted_perturbation_plated_loss_term(
                        qc, qc_obs_counts, loss_term
                    )
                else:
                    loss_term = self._compute_reweighted_perturbation_plated_loss_term(
                        D, D_obs_counts, loss_term
                    )

            loss_term = loss_term.sum(-1)
            loss_terms[var_key] = loss_term

        batch_elbo: torch.Tensor = sum([v for k, v in loss_terms.items()])
        loss = -batch_elbo.mean()
        loss += kl_loss
        metrics = {
            f"loss_term_{k}": -v.detach().cpu().mean() for k, v in loss_terms.items()
        }
        metrics["cf_KLdivergence"] = kl_loss

        return loss, metrics


    def kldiv_normal(self, mu1: torch.Tensor, sigma1: torch.Tensor,
            mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
        logvar1 = 2 * sigma1.log()
        logvar2 = 2 * sigma2.log()

        return torch.mean(-0.5 * torch.sum(1. + logvar1-logvar2 
            - (mu1-mu2)** 2 - (logvar1-logvar2).exp(), dim = 1), dim = 0)



    def _compute_reweighted_perturbation_plated_loss_term(
        self, conditioning_variable, total_obs_per_condition, loss_term
    ):
        """
        Reweight loss term corresponding to variable sampled on perturbation indexed plate

        In the ELBO, these variables are sampled a single time and shared across all samples
        that share a perturbation (eg the perturbation embedding). When computing the loss
        per mini-batch, we scale these loss terms so that the average mini-batch ELBO
        estimates the full ELBO. To do so, the weight for a perturbation plated variable
        is 1 / n_perturbation_samples for each sample that received that perturbation,
        where n_perturbation_samples is the total number of samples that received the
        perturbation.

        conditioning_variable: n x n_conditions, non-zero indicates perturbation
        applied to sample
        total_obs_per_condition: n_conditions, total number of samples where each
        perturbation is applied
        loss_term: n_particles x n_conditions x n_variables, unweighted loss term on
        perturbation plated variables

        Returns: rw_loss_term, n_particles x n x n_variables, where loss has been
        reweighted for perturbation plated variables
        """
        condition_nonzero = (conditioning_variable != 0).type(torch.float32)

        obs_scaling = 1 / total_obs_per_condition
        obs_scaling[torch.isinf(obs_scaling)] = 0
        obs_scaling = obs_scaling.reshape(1, -1)

        rw_condition_nonzero = condition_nonzero * obs_scaling
        rw_loss_term = torch.matmul(rw_condition_nonzero, loss_term)
        return rw_loss_term


class PerturbationPlatedELBOCustomReweightedLossModule(torch.nn.Module):
    """
    Computes ELBO with option to specify variables that are on perturbation-based
    plates (eg perturbation embeddings that are shared between samples that receive
    a given perturbation)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        guide: torch.nn.Module,
        local_variables: Optional[Iterable[str]] = None,
        perturbation_plated_variables: Optional[Iterable[str]] = None,
        custom_prior_weights: Optional[Dict[str, float]] = None,
        custom_plated_prior_additional_weight_proportional_n: Optional[
            Dict[str, float]
        ] = None,
        custom_loss_term_weights: Optional[Iterable[str]] = None,
        custom_plated_loss_term_additional_weight_proportional_n: Optional[
            Dict[str, float]
        ] = None,
    ):
        """
        ELBO loss with option to reweight priors / loss terms

        custom_prior_weights: dict mapping from var_name to multiplier of prior in ELBO
        custom_plated_prior_additional_weight_proportional_n: dict mapping from perturbation
            plated variable name to proportional additional weight to apply per samples. Rather
            than each sample receiving 1 / n_perturbation_samples weight for the prior for the
            perturbations that it receives (would add up to 1 over the dataset, for the global
            variable), the sample receives (w + 1 / n_perturbations_samples) weight of the prior.
            This results in receiving prior weight 1 + w * n_perturbations sample over the
            whole dataset.

        custom_loss_term_weights and custom_plated_loss_term_additional_weights_proportional_n
        are equivalent variables for reweighting the full KL term, rather than just the prior

        Can only specify one type of reweighting at a time (this can be changed if desired)
        """
        super().__init__()
        local_variables = list(local_variables) if local_variables is not None else []
        perturbation_plated_variables = (
            list(perturbation_plated_variables)
            if perturbation_plated_variables is not None
            else []
        )

        assert sorted(model.get_var_keys()) == sorted(
            guide.get_var_keys()
        ), "Mismatch in model and guide variables"

        # make sure that all variables are specified as a local variable or a
        # perturbation plated variable
        variables = local_variables + perturbation_plated_variables
        assert sorted(list(model.get_var_keys())) == sorted(
            variables
        ), "Mismatch between model variables and variables specified to loss module"

        self.model = model
        self.guide = guide

        self.local_variables = local_variables
        self.perturbation_plated_variables = perturbation_plated_variables

        # TODO: clean this up (adding quickly to have example)

        # make sure that variables specified for custom prior / loss term reweighting are
        # valid
        if custom_prior_weights is not None:
            assert set(custom_prior_weights.keys()).issubset(set(variables))
        if custom_plated_prior_additional_weight_proportional_n is not None:
            assert set(
                custom_plated_prior_additional_weight_proportional_n.keys()
            ).issubset(set(perturbation_plated_variables))
        if custom_loss_term_weights is not None:
            assert set(custom_loss_term_weights).issubset(set(variables))
        if custom_plated_loss_term_additional_weight_proportional_n is not None:
            assert set(
                custom_plated_loss_term_additional_weight_proportional_n.keys()
            ).issubset(
                set(perturbation_plated_variables),
            )

        custom_weights_dicts = [
            custom_loss_term_weights,
            custom_plated_loss_term_additional_weight_proportional_n,
            custom_prior_weights,
            custom_plated_prior_additional_weight_proportional_n,
        ]
        if sum(d is not None for d in custom_weights_dicts) > 1:
            # TODO: have not though about how to handle this (likely not necessary)
            raise NotImplementedError(
                "Handling multiple reweighting types not implemented"
            )

        self.custom_prior_weights = custom_prior_weights
        self.custom_plated_prior_additional_weight_proportional_n = (
            custom_plated_prior_additional_weight_proportional_n
        )
        self.custom_loss_term_weights = custom_loss_term_weights
        self.custom_plated_loss_term_additional_weight_proportional_n = (
            custom_plated_loss_term_additional_weight_proportional_n
        )

    def forward(
        self,
        X: torch.Tensor,
        D: torch.Tensor,
        qc: torch.Tensor,
        qc_obs_counts: torch.Tensor,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
    ):
        if condition_values is None:
            condition_values = dict()

        guide_dists, guide_samples = self.guide(
            X=X,
            D=D,
            qc=qc,
            n_particles=n_particles,
        )

        for k, v in guide_samples.items():
            condition_values[k] = v

        model_dists, model_samples = self.model(
            D=D,
            qc=qc,
            condition_values=condition_values,
            n_particles=n_particles,
        )
        return guide_dists, model_dists, model_samples

    def loss(
        self,
        X: torch.Tensor,
        D: torch.Tensor,
        D_obs_counts: torch.Tensor,
        qc: torch.Tensor,
        qc_obs_counts: torch.Tensor,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
    ):
        guide_dists, model_dists, samples = self.forward(
            X=X,
            D=D,
            qc=qc,
            condition_values=condition_values,
            n_particles=n_particles,
        )

        loss_terms = {}
        loss_terms["reconstruction"] = model_dists["p_x"].log_prob(X).sum(-1)

        for k in guide_dists.keys():
            var_key = k[2:]  # drop 'q_'

            # score sample under prior
            p = model_dists[f"p_{var_key}"].log_prob(samples[var_key])
            q = guide_dists[f"q_{var_key}"].log_prob(samples[var_key])

            if var_key in self.perturbation_plated_variables:
                p = self._compute_reweighted_perturbation_plated_loss_term(
                    var_key,
                    D,
                    D_obs_counts,
                    qc,
                    qc_obs_counts,
                    p,
                    is_prior=True,
                )
                q = self._compute_reweighted_perturbation_plated_loss_term(
                    var_key,
                    D,
                    D_obs_counts,
                    qc,
                    qc_obs_counts,
                    q,
                    is_prior=False,
                )

            # scale prior / loss term by custom weight
            if self.custom_prior_weights is not None:
                if var_key in self.custom_prior_weights:
                    p = self.custom_prior_weights[var_key] * p

            if self.custom_loss_term_weights is not None:
                if var_key in self.custom_loss_term_weights:
                    p = self.custom_loss_term_weights[var_key] * p  # type: ignore
                    q = self.custom_loss_term_weights[var_key] * q  # type: ignore

            loss_term = p - q
            loss_term = loss_term.sum(-1)

            loss_terms[var_key] = loss_term

        batch_elbo: torch.Tensor = sum([v for k, v in loss_terms.items()])
        loss = -batch_elbo.mean()

        metrics = {
            f"loss_term_{k}": -v.detach().cpu().mean() for k, v in loss_terms.items()
        }
        return loss, metrics

    def _compute_reweighted_perturbation_plated_loss_term(
        self,
        var_key,
        conditioning_variable,
        total_obs_per_condition,
        loss_term,
        is_prior: bool = False,
    ):
        """
        Reweight loss term corresponding to variable sampled on perturbation indexed plate

        In the ELBO, these variables are sampled a single time and shared across all samples
        that share a perturbation (eg the perturbation embedding). When computing the loss
        per mini-batch, we scale these loss terms so that the average mini-batch ELBO
        estimates the full ELBO. To do so, the weight for a perturbation plated variable
        is 1 / n_perturbation_samples for each sample that received that perturbation,
        where n_perturbation_samples is the total number of samples that received the
        perturbation.

        conditioning_variable: n x n_conditions, non-zero indicates perturbation
        applied to sample
        total_obs_per_condition: n_conditions, total number of samples where each
        perturbation is applied
        loss_term: n_particles x n_conditions x n_variables, unweighted loss term on
        perturbation plated variables

        Returns: rw_loss_term, n_particles x n x n_variables, where loss has been
        reweighted for perturbation plated variables
        """
        condition_nonzero = (conditioning_variable != 0).type(torch.float32)

        obs_scaling = 1 / total_obs_per_condition
        obs_scaling[torch.isinf(obs_scaling)] = 0

        # add additional custom weight proportional to n_samples for given perturbation
        if is_prior:
            if self.custom_plated_prior_additional_weight_proportional_n is not None:
                if var_key in self.custom_plated_prior_additional_weight_proportional_n:
                    obs_scaling = (
                        obs_scaling
                        + self.custom_plated_prior_additional_weight_proportional_n[
                            var_key
                        ]
                    )

        if self.custom_plated_loss_term_additional_weight_proportional_n is not None:
            if var_key in self.custom_plated_loss_term_additional_weight_proportional_n:
                obs_scaling = (
                    obs_scaling
                    + self.custom_plated_loss_term_additional_weight_proportional_n[
                        var_key
                    ]
                )

        obs_scaling = obs_scaling.reshape(1, -1)

        rw_condition_nonzero = condition_nonzero * obs_scaling
        rw_loss_term = torch.matmul(rw_condition_nonzero, loss_term)
        return rw_loss_term


class PerturbationPlatedIWELBOLossModule(PerturbationPlatedELBOLossModule):
    """
    Loss module for optimizing the importance weighted ELBO, as in
    Importance Weighted Autoencoders (https://arxiv.org/abs/1509.00519)
    """

    def loss(
        self,
        X: torch.Tensor,
        D: torch.Tensor,
        D_obs_counts: torch.Tensor,
        qc: torch.Tensor,
        qc_obs_counts: torch.Tensor,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
    ):
        guide_dists, model_dists, samples = self.forward(
            X=X,
            D=D,
            qc=qc,
            condition_values=condition_values,
            n_particles=n_particles,
        )

        loss_terms = {}
        loss_terms["reconstruction"] = model_dists["p_x"].log_prob(X).sum(-1)

        for k in guide_dists.keys():
            var_key = k[2:]  # drop 'q_'

            # score sample under prior
            loss_term = model_dists[f"p_{var_key}"].log_prob(samples[var_key])
            loss_term = loss_term - guide_dists[f"q_{var_key}"].log_prob(
                samples[var_key]
            )

            if var_key in self.perturbation_plated_variables:
                # reweight perturbation plated variables
                if 'qc' in var_key:
                    loss_term = self._compute_reweighted_perturbation_plated_loss_term(
                        qc, qc_obs_counts, loss_term
                    )
                else:
                    loss_term = self._compute_reweighted_perturbation_plated_loss_term(
                        D, D_obs_counts, loss_term
                    )
            loss_term = loss_term.sum(-1)
            loss_terms[var_key] = loss_term

        # Key difference from ELBO is here
        # shape: (n_particles, n)
        loss_terms_tensor = torch.stack([v for k, v in loss_terms.items()]).sum(0)
        # shape: (n,)
        batch_iwelbo = torch.logsumexp(loss_terms_tensor, dim=0)
        batch_iwelbo = batch_iwelbo - np.log(n_particles)
        loss = -batch_iwelbo.mean()

        metrics = {
            f"loss_term_{k}": -v.detach().cpu().mean() for k, v in loss_terms.items()
        }
        with torch.no_grad():
            batch_elbo: torch.Tensor = sum([v for k, v in loss_terms.items()])
            metrics["batch_elbo"] = batch_elbo.mean()
        return loss, metrics



class dpo_loss_module(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        guide: torch.nn.Module,
        local_variables: Optional[Iterable[str]] = None,
        perturbation_plated_variables: Optional[Iterable[str]] = None,
        dosage_independent_variables: Optional[Iterable[str]] = None,
    ):
        super().__init__()

        # convert variables to lists
        local_variables = list(local_variables) if local_variables is not None else []
        perturbation_plated_variables = (
            list(perturbation_plated_variables)
            if perturbation_plated_variables is not None
            else []
        )

        # check valid variable lists
        assert sorted(model.get_var_keys()) == sorted(
            guide.get_var_keys()
        ), "Mismatch in model and guide variables"

        # make sure that all variables are specified as a local variable or a
        # perturbation plated variable
        variables = local_variables + perturbation_plated_variables
        assert sorted(list(model.get_var_keys())) == sorted(
            variables
        ), "Mismatch between model variables and variables specified to loss module"

        # make sure that dosage_independent_variables are valid
        if dosage_independent_variables is not None:
            assert set(dosage_independent_variables).issubset(set(variables))

        # store passed in values
        self.model = model
        self.guide = guide.eval()
        self.local_variables = local_variables
        self.perturbation_plated_variables = perturbation_plated_variables
        self.dosage_independent_variables = dosage_independent_variables

    def forward(
        self,
        X: torch.Tensor,
        D: torch.Tensor,
        qc: torch.Tensor,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
    ):
        if condition_values is None:
            condition_values = dict()

        pred_qc=None

        guide_dists, guide_samples = self.guide(
            X=X,
            D=D,
            qc=qc,
            n_particles=n_particles,
        )

        for k, v in guide_samples.items():
            condition_values[k] = v

        if "qm" in self.model._get_name():
            model_dists, model_samples = self.model(
                D=D,
                qc=qc,
                condition_values=condition_values,
                n_particles=n_particles,
            )

        else:
            model_dists, model_samples = self.model(
                D=D,
                qc=qc,
                condition_values=condition_values,
                n_particles=n_particles,
            )

        return guide_dists, model_dists, model_samples, pred_qc
        
        

    def loss(
        self,
        X: torch.Tensor,
        D: torch.Tensor,
        D_obs_counts: torch.Tensor,
        qc: torch.Tensor,
        qc_obs_counts: torch.Tensor,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
        test: bool = False,
    ):
        guide_dists, model_dists, samples, pred_qc = self.forward(
            X=X,
            D=D,
            qc=qc,
            condition_values=condition_values,
            n_particles=n_particles,
        )

        loss_terms = {}
        # loss_terms["reconstruction"] = model_dists["p_x"].log_prob(X).sum(-1) # 512, 5, 20000
        p_x_good_log_p = model_dists['p_x_good'].log_prob(X)
        p_x_bad_log_p = model_dists['p_x_bad'].log_prob(X)
        p_x_real_log_p = ((1-qc)*p_x_good_log_p) + (qc*p_x_bad_log_p)
        loss_terms["reconstruction"] = p_x_real_log_p.sum(-1)

        for k in guide_dists.keys():

            var_key = k[2:]  # drop 'q_'

            # score sample under prior
            prior_log_prob = model_dists[f"p_{var_key}"].log_prob(samples[var_key])
            guide_log_prob = guide_dists[f"q_{var_key}"].log_prob(samples[var_key])
            loss_term = prior_log_prob - guide_log_prob

            if var_key in self.perturbation_plated_variables:
                # reweight perturbation plated variables
                if 'qc' in var_key:
                    loss_term = self._compute_reweighted_perturbation_plated_loss_term(
                        qc, qc_obs_counts, loss_term
                    )
                else:
                    loss_term = self._compute_reweighted_perturbation_plated_loss_term(
                        D, D_obs_counts, loss_term
                    )

            loss_term = loss_term.sum(-1)
            loss_terms[var_key] = loss_term

        batch_elbo: torch.Tensor = sum([v for k, v in loss_terms.items()])
        loss = -batch_elbo.mean()
        metrics = {
            f"loss_term_{k}": -v.detach().cpu().mean() for k, v in loss_terms.items()
        }
        
        return loss, metrics