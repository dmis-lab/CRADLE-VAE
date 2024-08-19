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
gdown https://drive.google.com/uc?id=1OIi1Z3fiw8yKbzarLXMlxy5tJRm1w8Rx # datasets(norman, dixit, replogle, adamson)
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
