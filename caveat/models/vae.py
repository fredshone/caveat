from typing import Union

import torch
from torch import nn, tensor
from torch.nn import functional as F

from caveat.models.base import BaseVAE
from caveat.models.utils import argmax_on_axis, conv_size


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
            in_shape (tuple[int, int, int]): _description_
            latent_dim (int): _description_
            hidden_dims (list, optional): _description_. Defaults to None.
            kernel_size (Union[tuple[int, int], int], optional): _description_. Defaults to 3.
            stride (Union[tuple[int, int], int], optional): _description_. Defaults to 2.
            padding (Union[tuple[int, int], int], optional): _description_. Defaults to 1.
        """
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        channels, h, w = in_shape

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=h_dim,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            h, w = conv_size(
                (h, w), kernel_size=kernel_size, padding=padding, stride=stride
            )
            channels = h_dim

        self.shape_before_flattening = (-1, channels, h, w)
        self.encoder = nn.Sequential(*modules)
        flat_size = int(channels * h * w)
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_var = nn.Linear(flat_size, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, flat_size)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=(0, 1),
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                out_channels=in_shape[0],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=(0, 1),
            ),
            nn.BatchNorm2d(in_shape[0]),
            # nn.LeakyReLU(),
            # nn.Conv2d(
            #     hidden_dims[-1],
            #     out_channels=in_shape[0],
            #     kernel_size=3,
            #     padding=1,
            # ),
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

    def loss_function(self, *args, **kwargs) -> dict:
        r"""Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1))
        = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

        Returns:
            dict: _description_
        """

        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs[
            "M_N"
        ]  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0,
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
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
        samples = argmax_on_axis(samples, 2)
        return samples

    def generate(self, x: tensor, **kwargs) -> tensor:
        """Given an encoder input, return reconstructed output.

        Args:
            x (tensor): [B x C x H x W]

        Returns:
            tensor: [B x C x H x W]
        """

        samples = self.forward(x)[0]
        samples = argmax_on_axis(samples, 2)
        return samples
