from typing import Dict, Iterable, Literal, Optional, Sequence, Union

import anndata
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cradle_vae.analysis.average_treatment_effects import (
    estimate_model_average_treatment_effect,
)
from cradle_vae.data.utils.perturbation_datamodule import PerturbationDataModule
from cradle_vae.data.utils.perturbation_dataset import PerturbationDataset

import scanpy as sc
import rapids_singlecell as rsc
import cupy as cp
import gc
from datetime import datetime
from pytz import timezone

class PerturbationPlatedPredictor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        guide: nn.Module,
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
        self.model = model.eval()
        self.guide = guide.eval()
        self.local_variables = local_variables
        self.perturbation_plated_variables = perturbation_plated_variables
        self.dosage_independent_variables = dosage_independent_variables
        
        '''
        '''
        n_phenos = guide.n_phenos
        

    def _get_device(self):
        # TODO: clean up device management approach
        # assumes all parameters/buffers for model and guide are on same device
        device = next(self.model.parameters()).device
        return device

    @torch.no_grad()
    def compute_predictive_iwelbo(
        self,
        loaders: Union[DataLoader, Sequence[DataLoader]],
        n_particles: int,
    ) -> pd.DataFrame:
        """
        Compute IWELBO(X|variables, theta, phi) for trained model
        Importantly, does not score plated variables against priors

        Parameters
        ----------
        loaders: dataloaders with perturbation datasets
        n_particles: number of particles to compute predictive IWELBO

        Returns
        -------
        Dataframe with estimated predictive IWELBO for each datapoint
        in column "IWELBO", sample IDs in index

        """
        if isinstance(loaders, DataLoader):
            loaders = [loaders]

        device = self._get_device()

        # sample perturbation plated variables to share across batches
        guide_dists, guide_samples = self.guide(n_particles=n_particles)
        condition_values = {}
        for var_name in self.perturbation_plated_variables:
            condition_values[var_name] = guide_samples[var_name]

        # compute importance weighted ELBO
        id_list = []
        iwelbo_list = []
        for loader in loaders:
            idx_list_curr = []
            for batch in tqdm(loader):
                for k in batch:
                    batch[k] = batch[k].to(device)
                idx_list_curr.append(batch["idx"].detach().cpu().numpy())

                # catch adding library size if it becomes relevant
                # note: this part is not necessary for the guide
                # typically the llk is not evaluated in the guide, so we can skip this
                if self.model.likelihood_key == "library_nb":
                    condition_values["library_size"] = batch["library_size"]

                guide_dists, guide_samples = self.guide(
                    X=batch["X"],
                    D=batch["D"],
                    qc=batch["qc"],
                    condition_values=condition_values,
                    n_particles=n_particles,
                )

                # catch adding library size if it becomes relevant to the likelihood
                # necessary to evaluate predictive
                # this is strictly not the elegant way to do this, that would
                # be via args/kwargs, but a quick fix
                if self.model.likelihood_key == "library_nb":
                    guide_samples["library_size"] = batch["library_size"]

                if "qm" in self.model._get_name():
                    model_dists, model_samples = self.model(
                        D=batch["D"],
                        qc=batch["qc"],
                        condition_values=guide_samples,
                        n_particles=n_particles,
                    )
                else:
                    model_dists, model_samples = self.model(
                        D=batch["D"],
                        qc=batch["qc"],
                        condition_values=guide_samples,
                        n_particles=n_particles,
                    )

                iwelbo_terms_dict = {}
                # shape: (n_particles, n_samples)
                # iwelbo_terms_dict["x"] = model_dists["p_x"].log_prob(batch["X"]).sum(-1)
                p_x_good_log_p = model_dists['p_x_good'].log_prob(batch["X"])
                p_x_bad_log_p = model_dists['p_x_bad'].log_prob(batch["X"])
                if batch["qc"].shape[1] > 1:                    
                    b_qc = batch["qc"].max(axis=1)[0].unsqueeze(-1)
                    p_x_real_log_p = ((1-b_qc)*p_x_good_log_p) + (b_qc*p_x_bad_log_p)
                else:
                    p_x_real_log_p = ((1-batch["qc"])*p_x_good_log_p) + (batch["qc"]*p_x_bad_log_p)
                iwelbo_terms_dict["x"] = p_x_real_log_p.sum(-1)
                for var_name in self.local_variables:
                    p = (
                        model_dists[f"p_{var_name}"]
                        .log_prob(guide_samples[var_name])
                        .sum(-1)
                    )
                    q = (
                        guide_dists[f"q_{var_name}"]
                        .log_prob(guide_samples[var_name])
                        .sum(-1)
                    )
                    iwelbo_terms_dict[var_name] = p - q

                # shape: (n_particles, n_samples)
                iwelbo_terms = sum([v for k, v in iwelbo_terms_dict.items()])
                # compute batch IWELBO
                # shape: (n_samples,)
                batch_iwelbo = torch.logsumexp(iwelbo_terms, dim=0) - np.log(
                    n_particles
                )

                iwelbo_list.append(batch_iwelbo.detach().cpu().numpy())

            idx_curr = np.concatenate(idx_list_curr)
            dataset: PerturbationDataset = loader.dataset
            ids_curr = dataset.convert_idx_to_ids(idx_curr)
            id_list.append(ids_curr)

        iwelbo = np.concatenate(iwelbo_list)
        ids = np.concatenate(id_list)

        iwelbo_df = pd.DataFrame(
            index=ids, columns=["IWELBO"], data=iwelbo.reshape(-1, 1)
        )
        return iwelbo_df

    @torch.no_grad()
    def sample_observations(
        self,
        dosages: torch.Tensor,
        perturbation_names: Optional[Sequence[str]],
        n_particles: int = 1,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        x_var_info: Optional[pd.DataFrame] = None,
        qc=None
    ) -> anndata.AnnData:
        """
        Sample observations conditioned on perturbations

        Parameters
        ----------
        dosages: encoded dosages for perturbations of interest
        perturbation_names: optional string names for each row in dosages
        n_particles: number of samples to take for each dosage

        Returns
        -------
        anndata of samples dosage index, perturbation name, and particle_idx in obs,
        sampled observations in X, and x_var_info in var
        """
        device = self._get_device()
        dosages = dosages.to(device)
        qc = qc.to(device)
        # sample perturbation plated variables to share across batches
        guide_dists, guide_samples = self.guide(n_particles=n_particles)
        if condition_values is None:
            condition_values = dict()
        else:
            condition_values = {k: v.to(device) for k, v in condition_values.items()}
        for var_name in self.perturbation_plated_variables:
            condition_values[var_name] = guide_samples[var_name]

        x_good_samples_list, x_bad_samples_list = [], []
        for i in tqdm(range(dosages.shape[0])):
            D = dosages[i : i + 1]
            _, model_samples = self.model(
                D=D, condition_values=condition_values, n_particles=n_particles, qc=qc,
            )
            x_good_samples_list.append(model_samples["x_good"].detach().cpu().numpy().squeeze())
            x_bad_samples_list.append(model_samples["x_bad"].detach().cpu().numpy().squeeze())
            

        x_good_samples = np.concatenate(x_good_samples_list)
        obs = pd.DataFrame(index=np.arange(x_good_samples.shape[0]))
        obs["perturbation_idx"] = np.repeat(np.arange(dosages.shape[0]), n_particles)
        obs["particle_idx"] = np.tile(np.arange(dosages.shape[0]), n_particles)
        if perturbation_names is not None:
            obs["perturbation_name"] = np.array(perturbation_names)[
                obs["perturbation_idx"].to_numpy()
            ]

        adata_good = anndata.AnnData(obs=obs, X=x_good_samples)
        if x_var_info is not None:
            adata_good.var = x_var_info.copy()

        x_bad_samples = np.concatenate(x_bad_samples_list)
        obs = pd.DataFrame(index=np.arange(x_bad_samples.shape[0]))
        obs["perturbation_idx"] = np.repeat(np.arange(dosages.shape[0]), n_particles)
        obs["particle_idx"] = np.tile(np.arange(dosages.shape[0]), n_particles)
        if perturbation_names is not None:
            obs["perturbation_name"] = np.array(perturbation_names)[
                obs["perturbation_idx"].to_numpy()
            ]

        adata_bad = anndata.AnnData(obs=obs, X=x_bad_samples)
        if x_var_info is not None:
            adata_bad.var = x_var_info.copy()
        return adata_good, adata_bad

    def sample_observations_data_module(
        self,
        data_module: PerturbationDataModule,
        n_particles: int,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        qc = None,
    ):
        """
        Sample observations from each unique intervention observed in a PerturbationDataModule
        TODO: come up with better naming for this method

        Parameters
        ----------
        data_module
        n_particles

        Returns
        -------
        anndata with samples from unique interventions in data module
        obs will have perturabtion name and particle idx, X will have sampled observations,
        and var dataframe will have
        """
        perturbation_names = data_module.get_unique_observed_intervention_info().index
        D = data_module.get_unique_observed_intervention_dosages(perturbation_names)
        x_var_info = data_module.get_x_var_info()

        adata_good, adata_bad = self.sample_observations(
            dosages=D,
            perturbation_names=perturbation_names,
            x_var_info=x_var_info,
            n_particles=n_particles,
            condition_values=condition_values,
            qc=qc
        )

        return adata_good, adata_bad

    @torch.no_grad()
    def estimate_average_treatment_effects(
        self,
        dosages_alt: torch.Tensor,
        dosages_control: torch.Tensor,
        quality: torch.Tensor,
        method: Literal["mean", "perturbseq"],
        n_particles: int = 1000,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        perturbation_names_alt: Optional[Sequence[str]] = None,
        perturbation_name_control: Optional[str] = None,
        x_var_info: Optional[pd.DataFrame] = None,
        batch_size: int = 500,
    ) -> anndata.AnnData:
        """
        Estimate average treatment effects of alternate dosages relative control dosage using model

        Parameters
        ----------
        dosages_alt: alternate dosages
        dosages_control: control dosage
        method: mean or perturbseq (log fold change after normalization for library size)
        n_particles: number of samples per treatment for estimate
        condition_values: any additional conditioning variables for model / guide
        perturbation_names: names for dosages, will be used as obs index if provided
        x_var_info: names of observed variables, will be included as var if provided

        Returns
        -------
        anndata with average treatment effects in X, perturbation names as obs index if provided
        (aligned to dosages_alt otherwise), and x_var_info as var if provided
        """
        device = self._get_device()
        dosages_alt = dosages_alt.to(device)
        dosages_control = dosages_control.to(device)
        quality = quality.to(device)
        if condition_values is not None:
            for k in condition_values:
                condition_values[k] = condition_values[k].to(device)

        average_treatment_effects = estimate_model_average_treatment_effect(
            model=self.model,
            guide=self.guide,
            dosages_alt=dosages_alt,
            dosages_control=dosages_control,
            quality=quality,
            n_particles=n_particles,
            method=method,
            condition_values=condition_values,
            batch_size=batch_size,
            dosage_independent_variables=self.dosage_independent_variables,
        )
        adata = anndata.AnnData(average_treatment_effects)
        if perturbation_names_alt is not None:
            adata.obs = pd.DataFrame(index=np.array(perturbation_names_alt))
        if perturbation_name_control is not None:
            adata.uns["control"] = perturbation_name_control
        if x_var_info is not None:
            adata.var = x_var_info.copy()
        return adata

    def estimate_average_effects_data_module(
        self,
        data_module: PerturbationDataModule,
        control_label: str,
        method: Literal["mean", "perturbseq"],
        n_particles: int = 100,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: int = 500,
    ):
        perturbation_names = data_module.get_unique_observed_intervention_info().index
        perturbation_names_alt = [
            name for name in perturbation_names if name != control_label
        ]

        dosages_alt = data_module.get_unique_observed_intervention_dosages(
            perturbation_names_alt
        )
        dosages_ref = data_module.get_unique_observed_intervention_dosages(
            [control_label]
        )
        # good_quality = torch.zeros((1, data_module.get_qc_var_info().shape[0]))
        good_quality = torch.ones((1, data_module.get_qc_var_info().shape[0]))

        x_var_info = data_module.get_x_var_info()

        adata = self.estimate_average_treatment_effects(
            dosages_alt=dosages_alt,
            dosages_control=dosages_ref,
            quality=good_quality,
            method=method,
            n_particles=n_particles,
            condition_values=condition_values,
            perturbation_names_alt=perturbation_names_alt,
            perturbation_name_control=control_label,
            x_var_info=x_var_info,
            batch_size=batch_size,
        )
        return adata

    def estimate_qc_ratio(
        self,
        data_module: PerturbationDataModule,
        n_particles: int,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        thr: int = 3,
    ):
        good_quality = torch.zeros((1, data_module.get_qc_var_info().shape[0]))
        # good_quality = torch.ones((1, data_module.get_qc_var_info().shape[0]))
        adata, adata_bad = self.sample_observations_data_module(data_module,100,condition_values, good_quality)
        sub_adata_list = []
        
        n_each_pert = adata.shape[0]/len(adata.obs.perturbation_idx.unique())
        step = int(n_each_pert/10)
        for i in tqdm(range(10)):
            idx = adata.obs.groupby('perturbation_idx').apply(lambda x: x[i*step:(i+1)*step]).index.get_level_values(1)
            sub_adata = adata[idx]
            sub_adata = get_qc_annotation(sub_adata)
            sub_adata_list.append(sub_adata)
        adata = anndata.concat(sub_adata_list, axis=0)

        result = {}
        for thr in [3,4,5]:
            adata, qc_col = get_qc_one_hot_cols(adata, thr, data_module.thr_values[thr])
            qc_pass,qc_fail = adata.obs['total_qc'].value_counts()
            result[thr] = round(qc_pass / (qc_pass+qc_fail),4)
        return result



def get_qc_annotation(adata):
    print(f"START : get QC annoation\t{datetime.now(timezone('Asia/Seoul')).strftime('%H:%M:%S')}")
    adata.var.rename(columns={'index':'ensemble_id'}, inplace=True)
    adata.var['ncounts']    = adata.X.sum(axis=0).tolist()[0]
    adata.var['ncells']     = (adata.X > 0).sum(axis=0).tolist()[0]
    adata.obs['UMI_count']  = adata.X.sum(axis=1)
    adata.obs['ngenes']     = (adata.X > 0).sum(axis=1)
    adata.var["mt"]         = adata.var.index.str.startswith("MT-")
    adata.var["ribo"]       = adata.var.index.str.startswith(("RPS", "RPL"))
    adata.var["hb"]         = adata.var.index.str.contains("^HB[^(P)]")
    # with cp.cuda.Device(5):
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], log1p=True)
    rsc.pp.scrublet(adata)
    rsc.get.anndata_to_CPU(adata)
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    print(f"DONE  : get QC annoation\t{datetime.now(timezone('Asia/Seoul')).strftime('%H:%M:%S')}")
    return adata


def get_qc_one_hot_cols(adata, thr, thr_values):
    adata.obs['qc_UMI_count']           = _get_result_of_qc(adata.obs['UMI_count'], metric='mad', thr=-thr, threshold = thr_values['thr_qc_UMI_count'])
    adata.obs['qc_ngenes']              = _get_result_of_qc(adata.obs['ngenes'], metric='mad', thr=-thr, threshold = thr_values['thr_qc_ngenes'])
    adata.obs['qc_pct_counts_mt']       = _get_result_of_qc(adata.obs['pct_counts_mt'], metric='mad', thr=thr, threshold = thr_values['thr_qc_pct_counts_mt'])
    adata.obs['qc_pct_counts_ribo']     = _get_result_of_qc(adata.obs['pct_counts_ribo'], metric='mad', thr=thr, threshold = thr_values['thr_qc_pct_counts_ribo'])
    adata.obs['qc_pct_counts_hb']       = _get_result_of_qc(adata.obs['pct_counts_hb'], metric='mad', thr=thr, threshold = thr_values['thr_qc_pct_counts_hb'])
    adata.obs['qc_predicted_doublet']   = (adata.obs['predicted_doublet'] == True).astype(int)
    qc_one_hot_cols = [col for col in adata.obs.columns if "qc_" in col]
    adata.obs["num_qc"] = adata.obs[qc_one_hot_cols].sum(1)
    adata.obs['total_qc'] = (adata.obs["num_qc"]>0).astype(int)

    return adata, qc_one_hot_cols


def _get_result_of_qc(x_series, metric = 'iqr', thr=1.5, threshold=None):

    if threshold == None:
        if metric == 'iqr':
            Q1 = np.percentile(x_series, 25)
            Q3 = np.percentile(x_series, 75)
            IQR = Q3 - Q1
            threshold = Q3 + thr * IQR # 1.5, 3, 4.5
        elif metric == 'mad': # median absolute deviation
            med = np.median(x_series)
            MAD = np.median(abs(x_series-med))
            threshold = med + MAD * thr
    
    if thr < 0:
        result_of_qc = (x_series < threshold).astype(int)
    elif thr > 0:
        result_of_qc = (x_series > threshold).astype(int)

    return result_of_qc