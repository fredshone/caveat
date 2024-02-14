from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from torch import Tensor, optim

from caveat.models.base import BaseVAE


class Experiment(pl.LightningModule):
    def __init__(
        self,
        model: BaseVAE,
        LR: float = 0.005,
        weight_decay: float = 0.0,
        scheduler_gamma: float = 0.95,
        kld_weight: float = 0.00025,
        duration_weight: float = 1.0,
    ) -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.LR = LR
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.kld_weight = kld_weight
        self.duration_weight = duration_weight
        self.curr_device = None
        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch: Tensor, teacher=None, **kwargs) -> List[Tensor]:
        return self.model(batch, teacher, **kwargs)

    def training_step(self, batch, batch_idx):
        (x, x_mask), (y, y_mask) = batch
        self.curr_device = x.device

        results = self.forward(x, teacher=x)  # use x as teacher (shifted left)
        train_loss = self.model.loss_function(
            *results,
            target=y,
            mask=y_mask,
            kld_weight=self.kld_weight,
            duration_weight=self.duration_weight,
            batch_idx=batch_idx,
        )
        self.log_dict(
            {key: val.item() for key, val in train_loss.items()}, sync_dist=True
        )

        return train_loss["loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        (x, x_mask), (y, y_mask) = batch
        self.curr_device = x.device

        results = self.forward(x)
        val_loss = self.model.loss_function(
            *results,
            target=y,
            mask=y_mask,
            kld_weight=self.kld_weight,
            duration_weight=self.duration_weight,
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
        (x, _), (y, _) = next(iter(self.trainer.datamodule.test_dataloader()))
        x = x.to(self.curr_device)
        y = y.to(self.curr_device)
        self.regenerate_batch(x, target=y, name="test_reconstructions")

    def regenerate_val_batch(self):
        (x, _), (y, _) = next(iter(self.trainer.datamodule.val_dataloader()))
        x = x.to(self.curr_device)
        y = y.to(self.curr_device)
        self.regenerate_batch(x, target=y, name="reconstructions")

    def regenerate_batch(self, x: Tensor, target: Tensor, name: str):
        y_probs = self.model.generate(x, self.curr_device).squeeze()
        image = unpack(target, y_probs, self.curr_device)
        div = torch.ones_like(y_probs)
        images = torch.cat((image.squeeze(), div, y_probs), dim=-1)
        vutils.save_image(
            pre_process(images.data),
            Path(
                self.logger.log_dir,
                name,
                f"recons_{self.logger.name}_epoch_{self.current_epoch}.png",
            ),
            normalize=False,
            nrow=1,
            pad_value=1,
        )

    def sample_sequences(self, num_samples: int, name: str = "samples") -> None:
        z = torch.randn(num_samples, self.model.latent_dim)
        y_probs = self.model.predict_step(z, self.curr_device)
        vutils.save_image(
            pre_process(y_probs.cpu().data),
            Path(
                self.logger.log_dir,
                name,
                f"{self.logger.name}_epoch_{self.current_epoch}.png",
            ),
            normalize=False,
            nrow=1,
            pad_value=1,
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


def unpack(x, y, current_device):
    if x.dim() == 2:
        # assume cat encoding and unpack into image
        channels = y.shape[-1]
        eye = torch.eye(channels)
        eye = eye.to(current_device)
        ximage = eye[x.long()].squeeze()
        return ximage

    elif x.shape[-1] == 2:
        # assume cat encoding and unpack into image
        channels = y.shape[-1] - 1
        acts, durations = x.split([1, 1], dim=-1)
        eye = torch.eye(channels)
        eye = eye.to(current_device)
        ximage = eye[acts.long()].squeeze()
        ximage = torch.cat((ximage, durations), dim=-1)
        return ximage
    return x


def pre_process(images):
    # hack for dealing with outputs encoded as [N, h, w]
    # need to add channel dim and rearrange to [N, C, h, w]
    # todo remove C/3d encoder
    s = images.shape
    if len(s) == 3:
        # need to add dim and move channel to front
        return images.reshape(s[0], s[1], s[2], 1).permute(0, 3, 1, 2)
    return images
