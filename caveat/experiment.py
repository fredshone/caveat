from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from torch import optim
from torch import tensor as Tensor

from caveat.models.base import BaseVAE


class Experiment(pl.LightningModule):
    def __init__(
        self,
        model: BaseVAE,
        LR: float = 0.005,
        weight_decay: float = 0.0,
        scheduler_gamma: float = 0.95,
        kld_weight: float = 0.00025,
    ) -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.LR = LR
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.kld_weight = kld_weight
        self.curr_device = None
        self.save_hyperparameters(ignore=["model"])

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        self.curr_device = batch.device

        results = self.forward(batch)
        train_loss = self.model.loss_function(
            *results,
            kld_weight=self.kld_weight,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {key: val.item() for key, val in train_loss.items()}, sync_dist=True
        )

        return train_loss["loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        self.curr_device = batch.device

        results = self.forward(batch)
        val_loss = self.model.loss_function(
            *results,
            kld_weight=self.kld_weight,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()},
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_end(self) -> None:
        self.regenerate_val_batch()
        self.sample_sequences(100)

    def regenerate_test_batch(self):
        x = next(iter(self.trainer.datamodule.test_dataloader()))
        x = x.to(self.curr_device)
        self.regenerate_batch(x, name="test_reconstructions")

    def regenerate_val_batch(self):
        x = next(iter(self.trainer.datamodule.val_dataloader()))
        x = x.to(self.curr_device)
        self.regenerate_batch(x, name="reconstructions")

    def regenerate_batch(self, x: Tensor, name: str):
        reconstructed = self.model.generate(x, self.curr_device)
        vutils.save_image(
            reconstructed.data,
            Path(
                self.logger.log_dir,
                name,
                f"recons_{self.logger.name}_epoch_{self.current_epoch}.png",
            ),
            normalize=True,
            nrow=10,
            pad_value=0.5,
        )

    def sample_sequences(self, num_samples: int, name: str = "samples") -> None:
        z = torch.randn(num_samples, self.model.latent_dim)
        samples = self.model.predict_step(z, self.curr_device)
        vutils.save_image(
            samples.cpu().data,
            Path(
                self.logger.log_dir,
                name,
                f"{self.logger.name}_epoch_{self.current_epoch}.png",
            ),
            normalize=True,
            nrow=10,
        )

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(), lr=self.LR, weight_decay=self.weight_decay
        )
        optims.append(optimizer)

        if self.scheduler_gamma is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(
                optims[0], gamma=self.scheduler_gamma
            )
            scheds.append(scheduler)
        return optims, scheds

    def predict_step(self, batch):
        return self.model.predict_step(batch, self.curr_device)
