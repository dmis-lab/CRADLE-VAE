from typing import Dict, Literal, Optional, Sequence

import random
import anndata
import numpy as np
import pandas as pd
from sympy import subsets
import torch
import scanpy as sc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Sampler
from traitlets import Bool
from tqdm import tqdm
import parmap
from collections import defaultdict

from cradle_vae.analysis.average_treatment_effects import (
    estimate_data_average_treatment_effects,
)
from cradle_vae.data.norman.download import download_norman_dataset
from cradle_vae.data.utils.batch_statistics import batch_log_mean, batch_log_std
from cradle_vae.data.utils.perturbation_datamodule import (
    ObservationNormalizationStatistics,
    PerturbationDataModule,
)
from cradle_vae.data.utils.perturbation_dataset import SCRNASeqTensorPerturbationDataset


class BaseNormanDataModule(PerturbationDataModule):
    def __init__(
        self,
        split_kwargs: Dict,
        encode_combos_as_unique: bool = False,
        batch_size: int = 128,
        highly_variable_genes_only: bool = False,
        data_path: Optional[str] = None,
        do_curriculum: bool = False,
        n_qc_pass: int = None,
        n_qc_fail: int = None,
        qc_threshold: int = 3,
        use_each_qc: bool = False,
    ):
        super().__init__()

        self.encode_combos_as_unique = encode_combos_as_unique
        self.batch_size = batch_size
        self.do_curriculum = do_curriculum
        self.qc_threshold = qc_threshold
        self.use_each_qc = use_each_qc

        if data_path is None:
            data_path = download_norman_dataset()

        self.adata = anndata.read_h5ad(data_path)
        if 'qc' not in data_path:
            self.adata.X = self.adata.layers["counts"]
            del self.adata.layers
            self.adata = get_qc_annotation(self.adata)
            self.adata.write_h5ad(f"{data_path.split('.h5ad')[0]}_qc.h5ad")
        
        self.adata, qc_one_hot_cols, self.thr_values = get_qc_one_hot_cols(self.adata, thr=qc_threshold)
        self.thr_values = defaultdict(dict)
        adata = self.adata.copy()
        for qc_thr in [3,4,5]:
            _, _, self.thr_values[qc_thr] = get_qc_one_hot_cols(adata, thr=qc_thr)
        self.n_qc_pass, self.n_qc_fail = self.adata.obs['total_qc'].value_counts()


        if use_each_qc:
            qc = self.adata.obs[qc_one_hot_cols].to_numpy().astype(np.float32)
            qc = torch.from_numpy(qc)
            self.qc_var_info = pd.DataFrame(index=qc_one_hot_cols)
        else:
            qc = self.adata.obs['total_qc'].to_numpy().astype(np.float32)
            qc = torch.from_numpy(qc).unsqueeze(1)
            self.qc_var_info = pd.DataFrame(index=['total_qc'])
        
        
        if highly_variable_genes_only:
            self.adata = self.adata[:, self.adata.var["highly_variable"].to_numpy()]

        guide_one_hot_cols = get_guide_one_hot_cols(self.adata.obs)
        self.adata.obs["num_guides"] = self.adata.obs[guide_one_hot_cols].sum(1)

        # generate splits (implemented in subclass)
        split_labels = self._get_split_labels(self.adata.obs, **split_kwargs)
        self.adata.obs["split"] = split_labels

        # get scRNA-seq observation matrix
        X = torch.from_numpy(self.adata.X.toarray())

        # encode perturbation dosages
        treatment_labels = self.adata.obs["guide_identity"].astype(str)
        treatment_labels[
            self.adata.obs[guide_one_hot_cols].sum(1) == 0
        ] = "non-targeting"
        self.adata.obs["treatment"] = treatment_labels
        if not self.encode_combos_as_unique:
            # combinations encoded as application of two individual guides
            D = self.adata.obs[guide_one_hot_cols].to_numpy().astype(np.float32)
            self.d_var_info = pd.DataFrame(index=guide_one_hot_cols)
            D = torch.from_numpy(D)
        else:
            # combinations encoded as new treatments
            D_df = pd.get_dummies(self.adata.obs["treatment"])
            # encode non-targeting as no perturbation for consistency with other encoding
            D_df = D_df.drop(columns=["non-targeting"])
            self.d_var_info = D_df.T[[]]
            D = torch.from_numpy(D_df.to_numpy().astype(np.float32))

        # generate datasets
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


        # compute normalization statistics
        x_tr_mean = X_tr.mean(0)
        x_tr_std = X_tr.std(0)
        log_x_tr_mean = batch_log_mean(X_tr)
        log_x_tr_std = batch_log_std(X_tr, log_x_tr_mean)

        self.x_train_statistics = ObservationNormalizationStatistics(
            x_mean=x_tr_mean,
            x_std=x_tr_std,
            log_x_mean=log_x_tr_mean,
            log_x_std=log_x_tr_std,
        )

        # generate unique intervention info dataframe
        df = self.adata.obs.groupby("treatment")["split"].agg(set).reset_index()
        for split in ["train", "val", "test"]:
            df[split] = df["split"].apply(lambda x: split in x)
        df = df.set_index("treatment").drop(columns=["split"])
        self.unique_observed_intervention_df = df

        # generate mapping from intervention names to dosages
        self.adata.obs["i"] = np.arange(self.adata.shape[0])
        idx_map = (
            self.adata.obs.drop_duplicates("treatment")
            .set_index("treatment")["i"]
            .to_dict()
        )
        self.unique_intervention_dosage_map = {k: D[v] for k, v in idx_map.items()}

    def _get_cfX(self, adata, X, split):        
        print("get_cfX\tSTART")
        idx_per_t_qc_pass = adata[(adata.obs["split"] == split)&(adata.obs["total_qc"] == 0)].obs.groupby("treatment").indices
        idx_per_t_qc_fail = adata[(adata.obs["split"] == split)&(adata.obs["total_qc"] == 1)].obs.groupby("treatment").indices
        pass_fail = set(idx_per_t_qc_pass.keys()) - set(idx_per_t_qc_fail.keys())
        fail_pass = set(idx_per_t_qc_fail.keys()) - set(idx_per_t_qc_pass.keys())
        for i in pass_fail:
            idx_per_t_qc_fail[i] = np.array([])
        for i in fail_pass:
            idx_per_t_qc_pass[i] = np.array([])
        # cf_idx = adata[adata.obs["split"] == split].obs.apply(lambda x: idx_per_t_qc_pass[x.treatment] if x.total_qc == 1 else idx_per_t_qc_fail[x.treatment], axis=1)
        cf_idx = adata[adata.obs["split"] == split].obs.apply(lambda x: np.random.choice(idx_per_t_qc_fail[x.treatment], min(len(idx_per_t_qc_fail[x.treatment]), 50)), axis=1)
        X = X.cuda()
        # cfX = torch.stack(cf_idx.apply(lambda i: X[i].mean(0).detach().cpu() if len(i)>0 else torch.zeros(X.shape[1])).to_list())
        cfX = torch.stack(cf_idx.apply(lambda i: X[i].median(0)[0].detach().cpu() if len(i)>0 else torch.zeros(X.shape[1])).to_list())
        print("get_cfX\tDone")
        return cfX


    @staticmethod
    def _get_split_labels(
        obs: pd.DataFrame,
        split_seed: int,
        # used only by OOD data module
        frac_combinations_train: float = 0,
        frac_combinations_test: float = 0.2,
        # used only by data efficiency data module
        frac_combination_cells_train: float = 0,
    ) -> pd.Series:
        """
        Returns split labels for each cell (series with ID as index, aligned with obs)
        Implemented by subclasses
        """
        raise NotImplementedError

    def train_dataloader(self):
        if self.do_curriculum:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=CurriculumSampler(self.train_dataset))
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_train_perturbation_obs_counts(self) -> torch.Tensor:
        return self.train_dataset.get_dosage_obs_per_dim()

    def get_val_perturbation_obs_counts(self) -> torch.Tensor:
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
        self, method: Literal["mean", "perturbseq"], split: Optional[str] = None, qc_pass: bool = False,
    ) -> Optional[anndata.AnnData]:
        adata = self.adata
        if qc_pass:
            adata = adata[adata.obs['total_qc']==0]
        if split is not None:
            adata = adata[adata.obs["split"] == split]
        return estimate_data_average_treatment_effects(
            adata,
            label_col="treatment",
            control_label="non-targeting",
            method=method,
        )

    def get_simulated_latent_effects(self) -> Optional[anndata.AnnData]:
        return None


class NormanOODCombinationDataModule(BaseNormanDataModule):
    def __init__(
        self,
        frac_combinations_train: float,
        frac_combinations_test: float = 0.2,
        split_seed: int = 0,
        encode_combos_as_unique: bool = False,
        batch_size: int = 128,
        highly_variable_genes_only: bool = False,
        data_path: Optional[str] = None,
        do_curriculum: bool = False,
        n_qc_pass: int = None,
        n_qc_fail: int = None,
        qc_threshold: int = 3,
        use_each_qc: bool = False,
    ):
        
        split_kwargs = dict(
            frac_combinations_train=frac_combinations_train,
            frac_combinations_test=frac_combinations_test,
            split_seed=split_seed,
        )
        self.split_seed = split_seed
        super().__init__(
            split_kwargs=split_kwargs,
            encode_combos_as_unique=encode_combos_as_unique,
            batch_size=batch_size,
            highly_variable_genes_only=highly_variable_genes_only,
            data_path=data_path,
            do_curriculum=do_curriculum,
            qc_threshold=qc_threshold,
            use_each_qc=use_each_qc,
        )

    @staticmethod
    def _get_split_labels(
        obs: pd.DataFrame,
        split_seed: int,
        # used only by OOD data module
        frac_combinations_train: float = 0,
        frac_combinations_test: float = 0.2,
        # used only by data efficiency data module
        frac_combination_cells_train: float = 0,
    ):
        # TODO: how to implement cleanly? Mypy error for changing signature
        # from superclass
        combo_guide_identities = np.sort(
            obs[obs["num_guides"] == 2]["guide_identity"].astype(str).unique()
        )

        # randomly shuffle combo guide identities using split seed
        rng = np.random.default_rng(split_seed)
        rng.shuffle(combo_guide_identities)

        # select first frac_combinations_train combos for train/val sets
        num_train_combos = int(frac_combinations_train * len(combo_guide_identities))
        train_combos = combo_guide_identities[:num_train_combos]

        # select last frac_combinations_test combos for test set
        num_test_combos = int(
            np.ceil(frac_combinations_test * len(combo_guide_identities))
        )
        test_combos = combo_guide_identities[-num_test_combos:]

        # split combo cells
        # splitting is done before filtering by train / test combos to ensure that samples
        # accumulate across different frac_combinations_train values
        obs_combo = obs[obs["num_guides"] == 2]
        obs_combo_tr_val, obs_combo_test = train_test_split(
            obs_combo,
            test_size=0.2,
            random_state=split_seed,
        )
        obs_combo_tr, obs_combo_val = train_test_split(
            obs_combo_tr_val,
            test_size=0.2,
            random_state=split_seed,
        )
        obs_combo_tr = obs_combo_tr[obs_combo_tr["guide_identity"].isin(train_combos)]
        obs_combo_val = obs_combo_val[
            obs_combo_val["guide_identity"].isin(train_combos)
        ]
        obs_combo_test = obs_combo_test[
            obs_combo_test["guide_identity"].isin(test_combos)
        ]

        # split non-combo cells
        obs_no_combo = obs[obs["num_guides"] < 2]
        obs_no_combo_tr, obs_no_combo_val = train_test_split(
            obs_no_combo, test_size=0.2, random_state=split_seed
        )

        # generate full split obs dataframes
        obs_tr = pd.concat([obs_no_combo_tr, obs_combo_tr])
        obs_val = pd.concat([obs_no_combo_val, obs_combo_val])
        obs_test = obs_combo_test

        split_labels = pd.Series(index=obs.index.copy(), data=None)
        split_labels.loc[obs.index.isin(obs_tr.index)] = "train"
        split_labels.loc[obs.index.isin(obs_val.index)] = "val"
        split_labels.loc[obs.index.isin(obs_test.index)] = "test"

        return split_labels
    


def get_guide_one_hot_cols(obs: pd.DataFrame):
    guide_one_hot_cols = [
        col
        for col in obs.columns
        if "guide_" in col and col not in ("guide_identity", "guide_ids")
    ]
    return guide_one_hot_cols


def get_qc_annotation(adata):
    # print('START : get QC annoation...')
    adata.var.rename(columns={'index':'ensemble_id'}, inplace=True)
    adata.var['ncounts']    = adata.X.sum(axis=0).tolist()[0]
    adata.var['ncells']     = (adata.X > 0).sum(axis=0).tolist()[0]
    adata.obs['UMI_count']  = adata.X.sum(axis=1)
    adata.obs['ngenes']     = (adata.X > 0).sum(axis=1)
    adata.var["mt"]         = adata.var.index.str.startswith("MT-")
    adata.var["ribo"]       = adata.var.index.str.startswith(("RPS", "RPL"))
    adata.var["hb"]         = adata.var.index.str.contains("^HB[^(P)]")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)
    # sc.pp.scrublet(adata)
    # print('DONE  : get QC annoation...')
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
