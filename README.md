# CRADLE-VAE: Enhancing Single-Cell Gene Perturbation Modeling with Counterfactual Reasoning-based Artifact Disentanglement
---
### Install environment (Linux)
```
# 1. Input your env path into prefix in environment.yaml
# vim environment.yaml
# prefix: /{your path}/anaconda3/envs/cradle_vae

# 2. Create env
conda env create --file environment.yaml
```

### Download datasets
If you want to annotated dataset when training our model,
```
wget "" # norman, dixit, replogle, adamson
tar -zxvf datasets.tar.gz
```
### Training models
The easiest way to train a model is specify a config file (eg `demo/cradle_vae_norman.yaml`) with data, model, and training hyperparameters
```
python train_norman.py --config ./demo/cradle_vae_norman.yaml
```
For larger experiments, we provide support for wandb sweeps using redun.
```
bash sweep_norman.sh
```

## Replicating results

We provide sweep configurations, python scripts, and jupyter notebooks to replicate each analysis from the paper in the `paper/experiments/` directory.
Additionally, we provide our precomputed metrics and checkpoints for download to allow exploration of the results without rerunning all experiments.
Detailed instructions for replicating each analysis are available in the README files of the `paper/experiments/` directory.
