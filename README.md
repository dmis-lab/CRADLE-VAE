# CRADLE-VAE: Enhancing Single-Cell Gene Perturbation Modeling with Counterfactual Reasoning-based Artifact Disentanglement
---
### Prerequisites
This project is tested with following environments:
- Python: 3.9.19
- CUDA: 11.8
- Pytorch-lightning: 2.4.0
- Rapids-singlecell: 0.10.8
- Scanpy: 1.10.2
---
### Install environment (Linux)
```
conda env create --file environment.yml 
```
- If you encounter a conflict, run this command: `conda config --set channel_priority disabled`

```
pip install 'rapids-singlecell[rapids11]' --extra-index-url=https://pypi.nvidia.com #CUDA11.X

pip install 'rapids-singlecell[rapids12]' --extra-index-url=https://pypi.nvidia.com #CUDA12
```
- Install `rapids-singlecell` according to your CUDA version

---
### Download datasets
If you want to annotated dataset when training our model,
```
pip install gdown
gdown https://drive.google.com/uc?id=1OIi1Z3fiw8yKbzarLXMlxy5tJRm1w8Rx # datasets(norman, dixit, replogle, adamson)
tar -zxvf datasets.tar.gz
```
---
### Training models
The easiest way to train a model is specify a config file (eg `demo/cradle_vae_norman.yaml`) with data, model, and training hyperparameters
```
python train_norman.py --config ./demo/cradle_vae_norman.yaml
```
For larger experiments, we provide support for wandb sweeps using redun.
```
bash sweep_norman.sh
```
