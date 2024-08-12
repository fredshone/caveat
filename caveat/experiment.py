from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from pandas import DataFrame
from torch import Tensor, optim

from caveat.models.utils import ScheduledOptim


class Experiment(pl.LightningModule):

    def __init__(
        self,
        gen: bool = False,
        test: bool = False,
        LR: float = 0.005,
        weight_decay: float = 0.0,
        kld_weight: float = 0.00025,
        duration_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super(Experiment, self).__init__()
        self.gen = gen
        self.test = test
        self.LR = LR
        self.weight_decay = weight_decay
        self.kwargs = kwargs
        self.kld_weight = kld_weight
        self.duration_weight = duration_weight
        self.curr_device = None
        self.save_hyperparameters()
        self.test_outputs = []

    def training_step(self, batch, batch_idx):
        (x, x_mask), (y, y_mask), conditionals = batch

        self.curr_device = x.device

        results = self.forward(x, conditionals=conditionals, target=y)
        train_loss = self.loss_function(
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
        (x, _), (y, y_mask), conditionals = batch
        self.curr_device = x.device

        results = self.forward(x, conditionals=conditionals)
        # z = results[-1]
        # DataFrame(z.cpu().numpy()).to_csv(
        #     Path(
        #         self.logger.log_dir,
        #         "val_z",
        #         f"z_epoch_{self.current_epoch}.csv",
        #     ),
        #     index=False,
        #     header=False,
        #     mode="a",
        # )
        val_loss = self.loss_function(
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
        if self.gen:
            self.regenerate_val_batch()
            self.sample_sequences()

    def test_step(self, batch, batch_idx):
        if self.test:
            (x, _), (y, y_mask), conditionals = batch
            self.curr_device = x.device

            results = self.forward(x, conditionals=conditionals)
            test_loss = self.loss_function(
                *results,
                target=y,
                mask=y_mask,
                kld_weight=self.kld_weight,
                duration_weight=self.duration_weight,
                batch_idx=batch_idx,
            )

            self.log_dict(
                {f"test_{key}": val.item() for key, val in test_loss.items()},
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def regenerate_val_batch(self):
        (x, _), (y, _), conditionals = next(
            iter(self.trainer.datamodule.val_dataloader())
        )
        x = x.to(self.curr_device)
        y = y.to(self.curr_device)
        conditionals = conditionals.to(self.curr_device)
        self.regenerate_batch(
            x, target=y, name="reconstructions", conditionals=conditionals
        )

    def regenerate_batch(
        self,
        x: Tensor,
        target: Tensor,
        name: str,
        conditionals: Optional[Tensor] = None,
    ):
        probs, _ = self.infer(
            x, conditionals=conditionals, device=self.curr_device
        )
        probs = probs.squeeze()
        image = unpack(target, probs, self.curr_device)
        div = torch.ones_like(probs)
        images = torch.cat((image.squeeze(), div, probs), dim=-1)
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

    def sample_sequences(self, name: str = "samples") -> None:
        _, _, conditionals = next(
            iter(self.trainer.datamodule.test_dataloader())
        )
        conditionals = conditionals.to(self.curr_device)
        z = torch.randn(len(conditionals), self.latent_dim)
        y_probs = self.predict(
            z, conditionals=conditionals, device=self.curr_device
        )
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

    def get_scheduler(self, optimizer):
        if self.kwargs.get("scheduler_gamma") is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.kwargs["scheduler_gamma"]
            )

        elif self.kwargs.get("warmup") is not None:
            lr_mul = self.kwargs.get("lr_mul", 2)
            d_model = self.kwargs.get("d_model", 512)
            n_warmup_steps = self.kwargs.get("warmup")
            scheduler = {
                "scheduler": ScheduledOptim(
                    optimizer,
                    lr_mul=lr_mul,
                    d_model=d_model,
                    n_warmup_steps=n_warmup_steps,
                ),
                "monitor": "val_loss",
                "interval": "step",
            }

        else:
            raise UserWarning("No scheduler specified")

        return scheduler

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.LR, weight_decay=self.weight_decay
        )

        scheduler = self.get_scheduler(optimizer)

        return [optimizer], [scheduler]

    def predict_step(self, batch):
        if len(batch) == 2:  # generative process
            zs, conditionals = batch
            return (
                conditionals,
                self.predict(
                    zs, conditionals=conditionals, device=self.curr_device
                ),
                zs,
            )
        # inference process only
        (x, _), (_, _), conditionals = batch
        preds, zs = self.infer(
            x, conditionals=conditionals, device=self.curr_device
        )
        return x, preds, zs, conditionals


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
    elif 1 in x.shape:
        print(f"Unknown encoding; {x.shape}, squeezing")
        return unpack(x.squeeze(), y.squeeze(), current_device)
    else:
        print(f"Unknown encoding; {x.shape}, returning input x")
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
