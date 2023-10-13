from pathlib import Path

import pytorch_lightning as pl
import torchvision.utils as vutils
from torch import optim
from torch import tensor as Tensor

from caveat.models.base import BaseVAE

default_params = {"kld_weight": 0.00025, "LR": 0.005, "weight_decay": 0.0}


class Experiment(pl.LightningModule):
    def __init__(self, model: BaseVAE) -> None:
        super(Experiment, self).__init__()

        self.model = model
        self.params = default_params
        self.curr_device = None

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        self.curr_device = batch.device

        results = self.forward(batch)
        train_loss = self.model.loss_function(
            *results,
            M_N=self.params[
                "kld_weight"
            ],  # al_img.shape[0]/ self.num_train_imgs,
            optimizer_idx=optimizer_idx,
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
            M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()},
            sync_dist=True,
        )

    def on_validation_end(self) -> None:
        self.sample_sequences()

    def sample_sequences(self):
        # Get sample reconstruction image
        x = next(iter(self.trainer.datamodule.test_dataloader()))
        x = x.to(self.curr_device)

        # test_input, test_label = batch
        reconstructed = self.model.generate(x)
        vutils.save_image(
            reconstructed.data,
            Path(
                self.logger.log_dir,
                "reconstructions",
                f"recons_{self.logger.name}_epoch_{self.current_epoch}.png",
            ),
            normalize=True,
            nrow=2,
        )

        # sample from latent space
        samples = self.model.sample(144, self.curr_device)
        vutils.save_image(
            samples.cpu().data,
            Path(
                self.logger.log_dir,
                "samples",
                f"{self.logger.name}_Epoch_{self.current_epoch}.png",
            ),
            normalize=True,
            nrow=2,
        )

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params["LR"],
            weight_decay=self.params["weight_decay"],
        )
        optims.append(optimizer)

        if self.params.get("scheduler_gamma") is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(
                optims[0], gamma=self.params["scheduler_gamma"]
            )
            scheds.append(scheduler)
        return optims, scheds
