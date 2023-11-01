from typing import Union

import torch
from torch import nn, tensor
from torchmetrics.classification import MulticlassHammingDistance
from torchmetrics.regression import MeanSquaredError

from caveat.models.base import BaseVAE
from caveat.models.utils import calc_output_padding, conv_size, hot_argmax


class VAE(BaseVAE):
    def __init__(
        self,
        in_shape: tuple[int, int, int],
        latent_dim: int,
        hidden_dims: list = None,
        kernel_size: Union[tuple[int, int], int] = 3,
        stride: Union[tuple[int, int], int] = 2,
        padding: Union[tuple[int, int], int] = 1,
        **kwargs,
    ) -> None:
        """Simple VAE model.

        Args:
            in_shape (tuple[int, int, int]): [C, time_step, activity_encoding].
            latent_dim (int): Latent space size.
            hidden_dims (list, optional): _description_. Defaults to None.
            kernel_size (Union[tuple[int, int], int], optional): _description_. Defaults to 3.
            stride (Union[tuple[int, int], int], optional): _description_. Defaults to 2.
            padding (Union[tuple[int, int], int], optional): _description_. Defaults to 1.
        """
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.MSE = MeanSquaredError()
        self.hamming = MulticlassHammingDistance(
            num_classes=in_shape[-1], average="micro"
        )
        modules = []
        target_shapes = []
        channels, h, w = in_shape

        # Build Encoder
        for hidden_channels in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=hidden_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.LeakyReLU(),
                )
            )
            target_shapes.append((h, w))
            h, w = conv_size(
                (h, w), kernel_size=kernel_size, padding=padding, stride=stride
            )
            channels = hidden_channels

        self.shape_before_flattening = (-1, channels, h, w)
        self.encoder = nn.Sequential(*modules)
        flat_size = int(channels * h * w)
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_var = nn.Linear(flat_size, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, flat_size)

        hidden_dims.reverse()

        _, channels, h, w = self.shape_before_flattening

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=channels,
                        out_channels=hidden_dims[i],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=calc_output_padding(target_shapes[i]),
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_dims[-1],
                out_channels=in_shape[0],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=calc_output_padding(target_shapes[-1]),
            ),
            nn.BatchNorm2d(in_shape[0]),
            nn.Tanh(),
        )

    def encode(self, input: tensor) -> list[tensor]:
        """Encodes the input by passing through the encoder network.

        Args:
            input (tensor): _description_

        Returns:
            list[tensor]: _description_
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: tensor) -> tensor:
        """Maps the given latent codes.

        Args:
            z (tensor): _description_

        Returns:
            tensor: _description_
        """
        result = self.decoder_input(z)
        result = result.view(self.shape_before_flattening)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: tensor, logvar: tensor) -> tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu (tensor): Mean of the latent Gaussian [B x D]
            logvar (tensor): Standard deviation of the latent Gaussian [B x D]

        Returns:
            tensor: [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: tensor, **kwargs) -> list[tensor]:
        """Forward pass.

        Args:
            input (tensor): _description_

        Returns:
            list[tensor]: _description_
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, recons, input, mu, log_var, **kwargs) -> dict:
        r"""Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1))
        = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

        Returns:
            dict: _description_
        """

        kld_weight = kwargs[
            "M_N"
        ]  # Account for the minibatch samples from the dataset
        # recons_loss = F.mse_loss(recons, input)
        recons_mse_loss = self.MSE(recons, input)
        recon_argmax = torch.argmax(recons, dim=-1)
        input_argmax = torch.argmax(input, dim=-1)

        recons_ham_loss = self.hamming(recon_argmax, input_argmax)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0,
        )

        loss = recons_mse_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "reconstruction_loss": recons_mse_loss.detach(),
            "recons_ham_loss": recons_ham_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> tensor:
        """Sample from the latent space and return the corresponding decoder space map.

        Args:
            num_samples (int): Number of samples.
            current_device (int): Device to run the model

        Returns:
            tensor: _description_
        """

        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        samples = hot_argmax(samples, -1)
        return samples

    def generate(self, x: tensor, **kwargs) -> tensor:
        """Given an encoder input, return reconstructed output.

        Args:
            x (tensor): [B x C x H x W]

        Returns:
            tensor: [B x C x H x W]
        """

        samples = self.forward(x)[0]
        samples = hot_argmax(samples, -1)
        return samples
