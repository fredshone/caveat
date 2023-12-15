import torch
from torch import nn, tensor
from torchmetrics.classification import MulticlassHammingDistance
from torchmetrics.regression import MeanSquaredError

from caveat.models.base import BaseVAE
from caveat.models.utils import hot_argmax


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        """LSTM Encoder.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            num_layers (int): number of lstm layers.
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        """LSTM Decoder.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            num_layers (int): number of lstm layers.
        """
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)


class IMAGE_LSTM_VAE(BaseVAE):
    def __init__(
        self,
        in_shape: tuple[int, int],
        latent_dim: int,
        hidden_layers: int,
        hidden_size: int,
        **kwargs,
    ) -> None:
        """Image LSTM VAE model.

        Args:
            in_shape (tuple[int, int]): [time_step, activity one-hot encoding].
            latent_dim (int): Latent space size.
            hidden_layers (int): Lstm  layers in encoder and decoder.
            hidden_size (int): Size of lstm layers.
        """
        super(IMAGE_LSTM_VAE, self).__init__()

        self.MSE = MeanSquaredError()
        self.hamming = MulticlassHammingDistance(
            num_classes=in_shape[-1], average="micro"
        )
        if len(in_shape) > 2:
            raise UserWarning(f"{self} only supports 2d encodings.")

        self.steps, self.acts = in_shape
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        self.lstm_enc = Encoder(
            input_size=self.acts,
            hidden_size=self.hidden_size,
            num_layers=hidden_layers,
        )
        self.lstm_dec = Decoder(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.acts,
            num_layers=hidden_layers,
        )
        flat_size = self.hidden_layers * self.hidden_size * 2
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_var = nn.Linear(flat_size, latent_dim)
        self.fc_hidden = nn.Linear(latent_dim, flat_size)

    def encode(self, input: tensor) -> list[tensor]:
        """Encodes the input by passing through the encoder network.

        Args:
            input (tensor): Input sequence batch [N, steps, acts].

        Returns:
            list[tensor]: Latent layer input (means and variances) [N, latent_dims].
        """
        # batch_size, seq_len, feature_dim = x.shape
        hidden = self.lstm_enc(input)
        # flatten last hidden cell layer
        flat = torch.cat(hidden).permute(1, 0, 2).flatten(start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(flat)
        log_var = self.fc_var(flat)

        return [mu, log_var]

    def decode(self, z: tensor) -> tensor:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        # initialize hidden state as inputs
        h = self.fc_hidden(z)

        # initialize hidden state
        hidden = (
            h.unflatten(1, (2 * self.hidden_layers, self.hidden_size))
            .permute(1, 0, 2)
            .split(self.hidden_layers)
        )
        # decode latent space to input space
        reps = int(self.steps * self.hidden_size / self.latent_dim)
        z = z.repeat(1, reps).reshape((-1, self.steps, self.hidden_size))

        reconstruct_output, hidden = self.lstm_dec(z, hidden)

        return reconstruct_output

    def reparameterize(self, mu: tensor, logvar: tensor) -> tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu (tensor): Mean of the latent Gaussian [N x latent_dims].
            logvar (tensor): Standard deviation of the latent Gaussian [N x latent_dims].

        Returns:
            tensor: [N x latent_dims].
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: tensor, **kwargs) -> list[tensor]:
        """Forward pass, also return latent parameterization.

        Args:
            input (tensor): Input sequences [N, steps, acts].

        Returns:
            list[tensor]: [Output [N, steps, acts], Input [N, steps, acts], mu [N, latent_dims], var [N, latent_dims]].
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, recons, input, mu, log_var, **kwargs) -> dict:
        r"""Computes the VAE loss function.

        Returns:
            dict: Losses.
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

    def predict_step(self, z: tensor, current_device: int, **kwargs) -> tensor:
        """Sample from the latent space and return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """

        z = z.to(current_device)
        samples = self.decode(z)
        samples = hot_argmax(samples, -1)
        return samples

    def generate(self, x: tensor, current_device: int, **kwargs) -> tensor:
        """Given an encoder input, return reconstructed output.

        Args:
            x (tensor): [N, steps, acts].

        Returns:
            tensor: [N, steps, acts].
        """
        samples = self.forward(x)[0]
        samples = samples.to(current_device)
        samples = hot_argmax(samples, -1)
        return samples
