from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from cradle_vae import data, models
from cradle_vae.data.utils.perturbation_dataset import PerturbationDataSample
from cradle_vae.models.utils.loss_modules import PerturbationPlatedELBOLossModule
from cradle_vae.models.utils.predictor import PerturbationPlatedPredictor


class PerturbationLightningModule(pl.LightningModule):
    """
    Pytorch LightningModule to manage training / evaluation of perturbation models
    """

    def __init__(
        self,
        loss_module: PerturbationPlatedELBOLossModule,
        lr: float,
        n_treatments: int,
        n_qc:int,
        D_obs_counts_train: Optional[torch.Tensor] = None,
        D_obs_counts_val: Optional[torch.Tensor] = None,
        D_obs_counts_test: Optional[torch.Tensor] = None,
        qc_obs_counts_train: Optional[torch.Tensor] = None,
        qc_obs_counts_val: Optional[torch.Tensor] = None,
        qc_obs_counts_test: Optional[torch.Tensor] = None,
        weight_decay: float = 1e-6,
        n_particles: int = 5,
        predictor: Optional[PerturbationPlatedPredictor] = None,
        x_var_info = None,
        use_scheduler = False,
    ):
        super().__init__()
        self.loss_module = loss_module
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_treatments = n_treatments
        self.n_qc = n_qc
        self.predictor = predictor
        self.x_var_info = x_var_info
        self.use_scheduler = use_scheduler

        # store as buffer so always on same device as module
        if D_obs_counts_train is not None:
            self.register_buffer("D_obs_counts_train", D_obs_counts_train)
        else:
            self.register_buffer("D_obs_counts_train", -1 * torch.ones((n_treatments,)))

        if D_obs_counts_train is not None:
            self.register_buffer("D_obs_counts_val", D_obs_counts_val)
        else:
            self.register_buffer("D_obs_counts_val", -1 * torch.ones((n_treatments,)))

        if D_obs_counts_train is not None:
            self.register_buffer("D_obs_counts_test", D_obs_counts_test)
        else:
            self.register_buffer("D_obs_counts_test", -1 * torch.ones((n_treatments,)))

        if qc_obs_counts_train is not None:
            self.register_buffer("qc_obs_counts_train", qc_obs_counts_train)
        else:
            self.register_buffer("qc_obs_counts_train", -1 * torch.ones((n_qc,)))

        if qc_obs_counts_train is not None:
            self.register_buffer("qc_obs_counts_val", qc_obs_counts_val)
        else:
            self.register_buffer("qc_obs_counts_val", -1 * torch.ones((n_qc,)))

        if qc_obs_counts_train is not None:
            self.register_buffer("qc_obs_counts_test", qc_obs_counts_test)
        else:
            self.register_buffer("qc_obs_counts_test", -1 * torch.ones((n_qc,)))

        self.n_particles = n_particles

        # self.apply(self.init_weights)

    def _get_extra_condition_values(self, batch: PerturbationDataSample):
        # extract extra condition values (eg library size for scRNA-seq data)
        # TODO: unify conditioning so this isn't necessary?
        condition_values = {}
        for k, v in batch.items():
            if k not in ("X", "D"):
                condition_values[k] = v
        return condition_values

    def training_step(self, batch: PerturbationDataSample, batch_idx: int):
        condition_values = self._get_extra_condition_values(batch)
        loss, metrics = self.loss_module.loss(
            X=batch["X"],
            D=batch["D"],
            qc=batch["qc"],
            cfX=batch["cfX"],
            condition_values=condition_values,
            D_obs_counts=self.D_obs_counts_train,
            qc_obs_counts=self.qc_obs_counts_train,
            n_particles=self.n_particles,
            test=False,
            x_var_info=self.x_var_info,
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in metrics.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: PerturbationDataSample, batch_idx: int):
        condition_values = self._get_extra_condition_values(batch)
        loss, metrics = self.loss_module.loss(
            X=batch["X"],
            D=batch["D"],
            qc=batch["qc"],
            cfX=batch["cfX"],
            condition_values=condition_values,
            D_obs_counts=self.D_obs_counts_val,
            qc_obs_counts=self.qc_obs_counts_val,
            n_particles=self.n_particles,
            test=True,
            x_var_info=self.x_var_info,
        )
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in metrics.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: PerturbationDataSample, batch_idx: int):
        condition_values = self._get_extra_condition_values(batch)
        loss, metrics = self.loss_module.loss(
            X=batch["X"],
            D=batch["D"],
            qc=batch["qc"],
            cfX=batch["cfX"],
            condition_values=condition_values,
            D_obs_counts=self.D_obs_counts_test,
            qc_obs_counts=self.qc_obs_counts_test,
            n_particles=self.n_particles,
            test=True,
            x_var_info=self.x_var_info,
        )
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in metrics.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.use_scheduler:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0005, max_lr=0.005, step_size_up=447, step_size_down=None, mode='triangular')
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'name': 'cyclicLR_lr',
                    'interval': 'step',
                }
            }
        else:
            return optimizer

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)



class TrainConfigPerturbationLightningModule(PerturbationLightningModule):
    """
    Perturbation lightning module subclass designed to be initialized
    from a config. Initializing from config allows easy saving of
    hyperparameters in checkpoints, so models can be reloaded more easily
    """

    def __init__(
        self,
        config: Dict,
        D_obs_counts_train: Optional[torch.Tensor] = None,
        D_obs_counts_val: Optional[torch.Tensor] = None,
        D_obs_counts_test: Optional[torch.Tensor] = None,
        qc_obs_counts_train: Optional[torch.Tensor] = None,
        qc_obs_counts_val: Optional[torch.Tensor] = None,
        qc_obs_counts_test: Optional[torch.Tensor] = None,
        x_var_info = None,
        use_scheduler = False,
    ):
        self.save_hyperparameters(
            ignore=["D_obs_counts_train", "D_obs_counts_val", "D_obs_counts_test",
                    "qc_obs_counts_train", "qc_obs_counts_val", "qc_obs_counts_test"]
        )
        self.config = config
        model = self._init_model()
        guide = self._init_guide()
        loss_module = self._init_loss_module(model, guide)
        predictor = self._init_predictor(model, guide)
        super().__init__(
            loss_module,
            n_treatments=config["model_kwargs"]["n_treatments"],
            n_qc=config["model_kwargs"]["n_qc"],
            D_obs_counts_train=D_obs_counts_train,
            D_obs_counts_val=D_obs_counts_val,
            D_obs_counts_test=D_obs_counts_test,
            qc_obs_counts_train=qc_obs_counts_train,
            qc_obs_counts_val=qc_obs_counts_val,
            qc_obs_counts_test=qc_obs_counts_test,
            predictor=predictor,
            x_var_info = x_var_info,
            **config.get("lightning_module_kwargs", dict()),
            use_scheduler=use_scheduler,
        )

    def _init_model(self):
        kwargs = self.config.get("model_kwargs", dict())
        model = getattr(models, self.config["model"])(**kwargs)
        return model

    def _init_guide(self):
        kwargs = self.config.get("guide_kwargs", dict())
        guide = getattr(models, self.config["guide"])(**kwargs)
        return guide

    def _init_loss_module(self, model: torch.nn.Module, guide: torch.nn.Module):
        kwargs = self.config.get("loss_module_kwargs", dict())
        kwargs["model"] = model
        kwargs["guide"] = guide
        kwargs["pos_weight"] = round(self.config['data_module_kwargs']['n_qc_pass']/(self.config['data_module_kwargs']['n_qc_pass']+self.config['data_module_kwargs']['n_qc_fail']),3)
        loss_module = getattr(models, self.config["loss_module"])(**kwargs)
        return loss_module

    def _init_predictor(self, model: torch.nn.Module, guide: torch.nn.Module):
        predictor = getattr(models, self.config["predictor"])(model, guide)
        return predictor

    def get_data_module(self):
        kwargs = self.config.get("data_module_kwargs", dict())
        data_module = getattr(data, self.config["data_module"])(**kwargs)
        return data_module
    