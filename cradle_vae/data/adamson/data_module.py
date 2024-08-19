from typing import Literal, Optional, Sequence

import anndata
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import scanpy as sc
import rapids_singlecell as rsc
import cupy as cp
import gc
from collections import defaultdict

from cradle_vae.analysis.average_treatment_effects import (
    estimate_data_average_treatment_effects,
)
from cradle_vae.data.utils.perturbation_datamodule import (
    ObservationNormalizationStatistics,
    PerturbationDataModule,
)
from cradle_vae.data.utils.perturbation_dataset import SCRNASeqTensorPerturbationDataset


class AdamsonDataModule(PerturbationDataModule):
    def __init__(
        self,
        # deprecated argument
        data_key: Optional[
            Literal["K562_genome_wide_filtered", "K562_essential"]
        ] = None,
        batch_size: int = 128,
        data_path: Optional[str] = None,
        qc_threshold: int = 3,
        n_qc_pass: int = None,
        n_qc_fail: int = None,
        use_each_qc: bool = False
    ):
        super().__init__()
        self.batch_size = batch_size
        self.qc_threshold = qc_threshold

        if data_path is None:
            raise "There is no data"
        
        self.adata = anndata.read_h5ad(data_path)
        if 'qc' not in data_path:
            self.adata = get_qc_annotation(self.adata)
            sc.pp.filter_genes(self.adata, min_cells=20)
            self.adata = self.adata[self.adata.obs['perturbation']!= '*']
            self.adata = self.adata[~self.adata.obs.isna().any(axis=1)]
            self.adata.obs['T'] = self.adata.obs['perturbation'].apply(lambda x: 'non-targeting' if '(mod)' in x else f"{x.split('_')[0]}")
            self.adata.write_h5ad(f"{data_path.split('.h5ad')[0]}_qc.h5ad")

        self.adata, qc_one_hot_cols, self.thr_values = get_qc_one_hot_cols(self.adata, thr=qc_threshold)
        adata = self.adata.copy()
        self.thr_values = defaultdict(dict)
        for qc_thr in [3,4,5]:
            _, _, self.thr_values[qc_thr] = get_qc_one_hot_cols(adata, thr=qc_thr)
        self.n_qc_pass, self.n_qc_fail = self.adata.obs['total_qc'].value_counts()

        qc = self.adata.obs['total_qc'].to_numpy().astype(np.float32)
        self.qc_var_info = pd.DataFrame(index=['total_qc'])
        qc = torch.from_numpy(qc).unsqueeze(1)

        # define splits
        idx = np.arange(self.adata.shape[0])
        train_idx, test_idx = train_test_split(idx, train_size=0.8, random_state=0)
        train_idx, val_idx = train_test_split(train_idx, train_size=0.8, random_state=0)

        self.adata.obs["split"] = None
        self.adata.obs.iloc[
            train_idx, self.adata.obs.columns.get_loc("split")
        ] = "train"
        self.adata.obs.iloc[val_idx, self.adata.obs.columns.get_loc("split")] = "val"
        self.adata.obs.iloc[test_idx, self.adata.obs.columns.get_loc("split")] = "test"

        # encode dosages
        # combine non-targeting guides to single label
        
        dosage_df = pd.get_dummies(self.adata.obs["T"])
        # encode non-targeting guides as 0
        dosage_df = dosage_df.drop(columns=["non-targeting"])

        self.d_var_info = dosage_df.T[[]]
        D = torch.from_numpy(dosage_df.to_numpy().astype(np.float32))

        X = torch.from_numpy(self.adata.X.toarray())

        ids_tr = self.adata.obs[self.adata.obs["split"] == "train"].index
        X_tr = X[(self.adata.obs["split"] == "train").to_numpy()]
        D_tr = D[(self.adata.obs["split"] == "train").to_numpy()]

        ids_val = self.adata.obs[self.adata.obs["split"] == "val"].index
        X_val = X[(self.adata.obs["split"] == "val").to_numpy()]
        D_val = D[(self.adata.obs["split"] == "val").to_numpy()]

        ids_test = self.adata.obs[self.adata.obs["split"] == "test"].index
        X_test = X[(self.adata.obs["split"] == "test").to_numpy()]
        D_test = D[(self.adata.obs["split"] == "test").to_numpy()]

        qc_tr = qc[(self.adata.obs["split"] == "train").to_numpy()]
        qc_val = qc[(self.adata.obs["split"] == "val").to_numpy()]
        qc_test = qc[(self.adata.obs["split"] == "test").to_numpy()]

        cfX_tr = self._get_cfX(adata=self.adata, X=X, split="train")
        torch.cuda.empty_cache()
        cfX_val = self._get_cfX(adata=self.adata, X=X, split="val")
        torch.cuda.empty_cache()
        cfX_test = self._get_cfX(adata=self.adata, X=X, split="test")
        torch.cuda.empty_cache()
        
        self.train_dataset = SCRNASeqTensorPerturbationDataset(
            X=X_tr, D=D_tr, ids=ids_tr, qc=qc_tr, cfX=cfX_tr,
        )
        self.val_dataset = SCRNASeqTensorPerturbationDataset(
            X=X_val, D=D_val, ids=ids_val, qc=qc_val, cfX=cfX_val,
        )
        self.test_dataset = SCRNASeqTensorPerturbationDataset(
            X=X_test, D=D_test, ids=ids_test, qc=qc_test, cfX=cfX_test,
        )

        x_tr_mean = X_tr.mean(0)
        x_tr_std = X_tr.std(0)
        log_x_tr = torch.log(X_tr + 1)
        log_x_tr_mean = log_x_tr.mean(0)
        log_x_tr_std = log_x_tr.std(0)

        self.x_train_statistics = ObservationNormalizationStatistics(
            x_mean=x_tr_mean,
            x_std=x_tr_std,
            log_x_mean=log_x_tr_mean,
            log_x_std=log_x_tr_std,
        )

        # because there are no perturbation combinations in this simulation,
        # unique_perturbations are the same as the observed perturbations
        # generate unique intervention info dataframe
        df = self.adata.obs.groupby("T")["split"].agg(set).reset_index()
        for split in ["train", "val", "test"]:
            df[split] = df["split"].apply(lambda x: split in x)
        df = df.set_index("T").drop(columns=["split"])
        self.unique_observed_intervention_df = df

        # generate mapping from intervention names to dosages
        self.adata.obs["i"] = np.arange(self.adata.shape[0])
        idx_map = self.adata.obs.drop_duplicates("T").set_index("T")["i"].to_dict()
        self.unique_intervention_dosage_map = {k: D[v] for k, v in idx_map.items()}

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_train_perturbation_obs_counts(self) -> torch.Tensor:
        return self.train_dataset.get_dosage_obs_per_dim()

    def get_val_perturbation_obs_counts(self) -> torch.Tensor:
        #
        return self.val_dataset.get_dosage_obs_per_dim()

    def get_test_perturbation_obs_counts(self) -> torch.Tensor:
        return self.test_dataset.get_dosage_obs_per_dim()

    def get_train_qc_obs_counts(self) -> torch.Tensor:
        return self.train_dataset.get_qc_obs_per_dim()

    def get_val_qc_obs_counts(self) -> torch.Tensor:
        return self.val_dataset.get_qc_obs_per_dim()

    def get_test_qc_obs_counts(self) -> torch.Tensor:
        return self.test_dataset.get_qc_obs_per_dim()

    def get_x_var_info(self) -> pd.DataFrame:
        return self.adata.var.copy()

    def get_d_var_info(self) -> pd.DataFrame:
        return self.d_var_info.copy()

    def get_qc_var_info(self) -> pd.DataFrame:
        return self.qc_var_info.copy()
    
    def get_qc_n(self) -> dict:
        return {'n_qc_pass':self.n_qc_pass,
                'n_qc_fail':self.n_qc_fail}

    def get_obs_info(self) -> pd.DataFrame:
        return self.adata.obs.copy()

    def get_x_train_statistics(self) -> ObservationNormalizationStatistics:
        return self.x_train_statistics

    def get_unique_observed_intervention_info(self) -> pd.DataFrame:
        return self.unique_observed_intervention_df.copy()

    def get_unique_observed_intervention_dosages(
        self, pert_names: Sequence
    ) -> torch.Tensor:
        D = torch.zeros((len(pert_names), self.d_var_info.shape[0]))
        for i, pert_name in enumerate(pert_names):
            D[i] = self.unique_intervention_dosage_map[pert_name]
        return D

    def get_estimated_average_treatment_effects(
        self,
        method: Literal["mean", "perturbseq"],
        split: Optional[str] = None,
        qc_pass: bool = False,
    ) -> Optional[anndata.AnnData]:
        adata = self.adata
        if qc_pass:
            adata = adata[adata.obs['total_qc']==0]
        if split is not None:
            adata = adata[adata.obs["split"] == split]
        return estimate_data_average_treatment_effects(
            adata,
            label_col="T",
            control_label="non-targeting",
            method=method,
        )

    def get_simulated_latent_effects(self) -> Optional[anndata.AnnData]:
        return None
    
    def _get_cfX(self, adata, X, split):        
        print("get_cfX\tSTART")
        idx_per_t_qc_pass = adata[(adata.obs["split"] == split)&(adata.obs["total_qc"] == 0)].obs.groupby("T", observed=False).indices
        idx_per_t_qc_fail = adata[(adata.obs["split"] == split)&(adata.obs["total_qc"] == 1)].obs.groupby("T", observed=False).indices
        pass_fail = set(idx_per_t_qc_pass.keys()) - set(idx_per_t_qc_fail.keys())
        fail_pass = set(idx_per_t_qc_fail.keys()) - set(idx_per_t_qc_pass.keys())
        for i in pass_fail:
            idx_per_t_qc_fail[i] = np.array([])
        for i in fail_pass:
            idx_per_t_qc_pass[i] = np.array([])
        # cf_idx = adata[adata.obs["split"] == split].obs.apply(lambda x: idx_per_t_qc_pass[x.treatment] if x.total_qc == 1 else idx_per_t_qc_fail[x.treatment], axis=1)
        cf_idx = adata[adata.obs["split"] == split].obs.apply(lambda x: np.random.choice(idx_per_t_qc_fail[x["T"]], min(len(idx_per_t_qc_fail[x["T"]]), 50)), axis=1)
        X = X.cuda()
        # cfX = torch.stack(cf_idx.apply(lambda i: X[i].mean(0).detach().cpu() if len(i)>0 else torch.zeros(X.shape[1])).to_list())
        cfX = torch.stack(cf_idx.apply(lambda i: X[i].median(0)[0].detach().cpu() if len(i)>0 else torch.zeros(X.shape[1])).to_list())
        print("get_cfX\tDone")
        return cfX


def get_qc_annotation(adata):
    print('START : get QC annoation...')
    if adata.var.index.name != 'gene_name': 
        adata.var = adata.var.rename_axis('gene_name', axis='index')
    adata.var = adata.var.reset_index().set_index('gene_name').rename(columns={'gene_id':'ensemble_id'})
    adata.var.index = adata.var.index.astype(str)
    adata.var_names_make_unique()
    adata.var['ncounts']    = adata.X.sum(axis=0).tolist()[0]
    adata.var['ncells']     = (adata.X > 0).sum(axis=0).tolist()[0]
    adata.obs['UMI_count']  = adata.X.sum(axis=1)
    adata.obs['ngenes']     = (adata.X > 0).sum(axis=1)
    adata.var["mt"]         = adata.var.index.str.startswith("MT-")
    adata.var["ribo"]       = adata.var.index.str.startswith(("RPS", "RPL"))
    adata.var["hb"]         = adata.var.index.str.contains("^HB[^(P)]")

    rsc.get.anndata_to_GPU(adata)
    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], log1p=True)
    rsc.pp.scrublet(adata)
    rsc.get.anndata_to_CPU(adata)

    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    print('DONE  : get QC annoation...')
    return adata


def get_qc_one_hot_cols(adata, thr):
    thr_values = {}
    adata.obs['qc_UMI_count'], thr_values['thr_qc_UMI_count']            = _get_result_of_qc(adata.obs['UMI_count'], metric='mad', thr=-thr)
    adata.obs['qc_ngenes'], thr_values['thr_qc_ngenes']                  = _get_result_of_qc(adata.obs['ngenes'], metric='mad', thr=-thr)
    adata.obs['qc_pct_counts_mt'], thr_values['thr_qc_pct_counts_mt']    = _get_result_of_qc(adata.obs['pct_counts_mt'], metric='mad', thr=thr)
    adata.obs['qc_pct_counts_ribo'], thr_values['thr_qc_pct_counts_ribo']= _get_result_of_qc(adata.obs['pct_counts_ribo'], metric='mad', thr=thr)
    adata.obs['qc_pct_counts_hb'], thr_values['thr_qc_pct_counts_hb']    = _get_result_of_qc(adata.obs['pct_counts_hb'], metric='mad', thr=thr)
    adata.obs['qc_predicted_doublet']   = (adata.obs['predicted_doublet'] == True).astype(int)
    qc_one_hot_cols = [col for col in adata.obs.columns if "qc_" in col]
    adata.obs["num_qc"] = adata.obs[qc_one_hot_cols].sum(1)
    adata.obs['total_qc'] = (adata.obs["num_qc"]>0).astype(int)

    return adata, qc_one_hot_cols, thr_values


def _get_result_of_qc(x_series, metric = 'iqr', thr=1.5):

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

    return result_of_qc, threshold