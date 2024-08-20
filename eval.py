"""
Script to generate test set evaluation metrics from a training run

Usage:
python [experiment_path] [--wandb] [--perturbseq] [--batch_size {int}]

Examples:
    python eval.py michael/debug/mw545rhs --wandb --perturbseq
    - Saves evaluation metrics to wandb run summary

    python eval.py results/example --perturbseq
    - Saves evaluation metrics to results/example/test_metrics.csv (local experiment)

    python eval.py {checkpoint_path}.ckpt --perturbseq
    - Runs evaluation for specified checkpoint, and saves metrics to
      {checkpoint_path}_test_metrics.csv
"""

import argparse
import os, sys
from os.path import basename, join, splitext
from typing import Any, Dict, Literal

import numpy as np
import pandas as pd
import torch
import wandb
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

os.chdir('/'.join(__file__.split('/')[:-1]))
sys.path.append('/'.join(__file__.split('/')[:-1]))

from cradle_vae.data.utils.anndata import align_adatas
from cradle_vae.models.utils.perturbation_lightning_module import (
    TrainConfigPerturbationLightningModule,
)
import cupy as cp

def evaluate_checkpoint(
    checkpoint_path: str,
    average_treatment_effect_method: Literal["mean", "perturbseq"],
    batch_size: int = 500,
    ate_n_particles: int = 2500,
    qc_pass: bool = False,
    thr: int = 5,
    devices = None,
) -> Dict[str, Any]:
    """
    Compute test set metrics for a given checkpoint


    Parameters
    ----------
    checkpoint_path: path to checkpoint
    average_treatment_effect_method: method to compute average treatment effect. "perturbseq"
        normalizes for library size and applies log transform before assessing effect
    batch_size: batch size to use for IWELBO computation

    Returns
    -------
    dictionary with test set metrics
    """
    lightning_module = load_checkpoint(checkpoint_path, devices)
    data_module = lightning_module.get_data_module()
    thr = data_module.thr_values
    predictor = lightning_module.predictor

    metrics = {}

    # compute test set IWELBO
    test_loader = DataLoader(
        data_module.test_dataloader().dataset,
        batch_size=batch_size,
    )
    test_iwelbo_df = predictor.compute_predictive_iwelbo(
        loaders=test_loader, n_particles=100
    )
    test_iwelbo = test_iwelbo_df["IWELBO"].mean()
    metrics["test/IWELBO"] = test_iwelbo

    # assess correlation between estimated average treatment effects from model and data
    data_ate = data_module.get_estimated_average_treatment_effects(
        method=average_treatment_effect_method,
        qc_pass=qc_pass
    )

    if data_ate is not None:
        model_ate = predictor.estimate_average_effects_data_module(
            data_module=data_module,
            control_label=data_ate.uns["control"],
            method=average_treatment_effect_method,
            n_particles=ate_n_particles,
            condition_values=dict(library_size=10000 * torch.ones((1,))),
            batch_size=batch_size,
        )

        data_ate, model_ate = align_adatas(data_ate, model_ate)

        intervention_info = data_module.get_unique_observed_intervention_info()

        metrics["ATE_n_particles"] = ate_n_particles

        # compute average treatment effect metrics for all perturbations
        ate_metrics_all_splits = get_ate_metrics(data_ate, model_ate)
        for k, v in ate_metrics_all_splits.items():
            metrics[f"{k}-all"] = v

        # compute average treatment effect metrics for perturbations available
        # in each split
        for split in ["train", "val", "test"]:
            split_perturbations = intervention_info[intervention_info[split]].index
            idx = data_ate.obs.index.isin(split_perturbations)
            ate_metrics_split = get_ate_metrics(data_ate[idx], model_ate[idx])
            for k, v in ate_metrics_split.items():
                metrics[f"{k}-{split}"] = v

    print('Check data quality')
    qc_ratio = predictor.estimate_qc_ratio(
        data_module=data_module,
        n_particles=ate_n_particles,
        condition_values=dict(library_size=10000 * torch.ones((1,))),
        thr=thr,
    )
    for thr in [3,4,5]:
        metrics[f'qc_ratio_thr{thr}'] = qc_ratio[thr]

    return metrics


def get_ate_metrics(data_ate, model_ate):
    metrics = {}
    top_20_idx_X = np.argpartition(np.abs(data_ate.X.copy()), data_ate.shape[1] - 20)[
        :, -20:
    ]
    top_20_idx_Y = np.argpartition(np.abs(model_ate.X.copy()), data_ate.shape[1] - 20)[
        :, -20:
    ]

    top_50_idx_X = np.argpartition(np.abs(data_ate.X.copy()), data_ate.shape[1] - 50)[
        :, -50:
    ]
    top_50_idx_Y = np.argpartition(np.abs(model_ate.X.copy()), data_ate.shape[1] - 50)[
        :, -50:
    ]


    ### pearson / R2 ###
    x = data_ate.X.flatten()
    y = model_ate.X.flatten()

    metrics["ATE_pearsonr"] = pearsonr(x, y)[0]
    metrics["ATE_r2"] = r2_score(x, y)

    # evaluate correlation / R2 across top 20 DE genes per perturbation
    x = np.take_along_axis(data_ate.X.copy(), top_20_idx_X, axis=-1).flatten()
    y = np.take_along_axis(model_ate.X.copy(), top_20_idx_X, axis=-1).flatten()

    metrics["ATE_pearsonr_top20"] = pearsonr(x, y)[0]
    metrics["ATE_r2_top20"] = r2_score(x, y)

    ### jaccard ###
    metrics["jaccard_sim_top20"] = jaccard_sim(top_20_idx_X, top_20_idx_Y)
    metrics["jaccard_sim_top50"] = jaccard_sim(top_50_idx_X, top_50_idx_Y)
    # vectorized_jaccard = np.vectorize(jaccard_similarity_row, signature='(n),(n)->()')
    # metrics["jaccard_sim_top20"] = np.mean(vectorized_jaccard(top_20_idx_X, top_20_idx_Y))
    # metrics["jaccard_sim_top50"] = np.mean(vectorized_jaccard(top_50_idx_X, top_50_idx_Y))

    return metrics


def jaccard_sim(X, Y):
    jaccard_sim_avg = np.mean([len(set(x)&set(y)) / len(set(x)|set(y)) for x,y in zip(X, Y)])
    return jaccard_sim_avg


# def jaccard_similarity_row(X, Y):
#     X_set, Y_set = set(X), set(Y)
#     return len(X_set & Y_set) / len(X_set | Y_set)


def evaluate_local_experiment(
    experiment_path: str,
    average_treatment_effect_method: Literal["mean", "perturbseq"],
    batch_size: int = 128,
    ate_n_particles: int = 2500,
    qc_pass: bool = False,
    thr: int = 5,
    devices = None,
):
    """
    Compute and save evaluation metrics for checkpoint with best eval loss in
     local experiment to `{experiment_path}/test_metrics.csv`

    Parameters
    ----------
    experiment_path: path to experiment (typically in results/ directory)
    average_treatment_effect_method
    batch_size: batch size used during IWELBO computation
    """
    checkpoint_names = os.listdir(join(experiment_path, "checkpoints"))
    # TODO: add better logic if needed
    best_checkpoints = [x for x in checkpoint_names if x[:4] == "best"]
    assert len(best_checkpoints) == 1
    checkpoint_path = join(experiment_path, "checkpoints", best_checkpoints[0])
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    metrics = evaluate_checkpoint(
        checkpoint_path,
        average_treatment_effect_method=average_treatment_effect_method,
        batch_size=batch_size,
        ate_n_particles=ate_n_particles,
        thr=thr,
    )
    metrics["checkpoint"] = checkpoint_name

    metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()}).T
    metrics_path = join(experiment_path, "test_metrics.csv")
    metrics_df.to_csv(metrics_path)


def evaluate_local_checkpoint(
    checkpoint_path: str,
    average_treatment_effect_method: Literal["mean", "perturbseq"],
    batch_size: int = 128,
    ate_n_particles: int = 2500,
    qc_pass: bool = False,
    thr: int = 5,
    devices = None,
):
    """
    Compute and save evaluation metrics specified checkpoint_path,
    saves results to {checkpoint_path}_test_metrics.csv

    Parameters
    ----------
    experiment_path: path to experiment (typically in results/ directory)
    average_treatment_effect_method
    batch_size: batch size used during IWELBO computation
    """
    checkpoint_base = splitext(checkpoint_path)[0]
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    metrics = evaluate_checkpoint(
        checkpoint_path,
        average_treatment_effect_method=average_treatment_effect_method,
        batch_size=batch_size,
        ate_n_particles=ate_n_particles,
        qc_pass=qc_pass,
        thr=thr,
    )
    metrics["checkpoint"] = checkpoint_name

    metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()}).T
    metrics_path = checkpoint_base + "_test_metrics.csv"
    metrics_df.to_csv(metrics_path)


def evaluate_wandb_experiment(
    experiment_path: str,
    average_treatment_effect_method: Literal["mean", "perturbseq"],
    batch_size: int = 128,
    ate_n_particles: int = 2500,
    qc_pass: bool = False,
    thr: int = 5,
    devices: list = None,
):
    """
    Compute and save evaluation metrics for checkpoint with best eval loss
    Metrics are saved to wandb run summary
    """
    cp.cuda.Device(devices[0]).use()
    api = wandb.Api()
    run = api.run(experiment_path)

    # TODO: improve logic if needed
    run_file_paths = [x.name for x in run.files()]
    best_checkpoint_paths = [
        x
        for x in run_file_paths
        if os.path.split(x)[0] == "checkpoints" and "best" in x
    ]
    assert len(best_checkpoint_paths) == 1
    wandb_file = run.file(best_checkpoint_paths[0])

    # download checkpoint
    basedir = run.name + "/" + "exp_bk2/"
    os.makedirs(basedir, exist_ok=True)
    checkpoint_path = wandb_file.download(root=basedir, replace=True).name

    metrics = evaluate_checkpoint(
        checkpoint_path,
        average_treatment_effect_method=average_treatment_effect_method,
        batch_size=batch_size,
        ate_n_particles=ate_n_particles,
        qc_pass=qc_pass,
        devices=devices,
        thr=thr,
    )

    # save metrics to run summary
    if qc_pass:
        for k in metrics:
            run.summary[f"{k}_qcpass"] = metrics[k]
    else:        
        for k in metrics:
            run.summary[k] = metrics[k]

    run.summary.update()


def load_checkpoint(checkpoint_path: str, devices):
    if devices is None:
        lightning_module = TrainConfigPerturbationLightningModule.load_from_checkpoint(
            checkpoint_path
        )
    else:
        lightning_module = TrainConfigPerturbationLightningModule.load_from_checkpoint(
            checkpoint_path, map_location=lambda storage, loc: storage.cuda(devices[0]) if torch.cuda.is_available() else storage
        )
    return lightning_module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", default='seungheun/cradle_vae_debug/2pj88nkq')
    parser.add_argument("--wandb", default=True) # action="store_true"
    parser.add_argument("--perturbseq", default=True) # action="store_true"
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ate_n_particles", type=int, default=2500)
    parser.add_argument("--qc_pass", type=bool, default=True)
    parser.add_argument("--devices", type=list, default=[2])
    parser.add_argument("--thr", type=int, default=3)
    
    args = parser.parse_args()

    cp.cuda.Device(args.devices[0]).use()

    method: Literal["mean", "perturbseq"] = "perturbseq" if args.perturbseq else "mean"
    if args.wandb:
        evaluate_wandb_experiment(
            args.experiment_path,
            method,
            batch_size=args.batch_size,
            ate_n_particles=args.ate_n_particles,
            qc_pass=args.qc_pass,
            thr=args.thr,
            devices=args.devices,
        )
    elif os.path.isdir(args.experiment_path):
        evaluate_local_experiment(
            args.experiment_path,
            method,
            batch_size=args.batch_size,
            ate_n_particles=args.ate_n_particles,
            thr=args.thr,
            devices=args.devices,
        )
    else:
        evaluate_local_checkpoint(
            args.experiment_path,
            method,
            batch_size=args.batch_size,
            ate_n_particles=args.ate_n_particles,
            qc_pass=args.qc_pass,
            thr=args.thr,
            devices=args.devices,
        )
