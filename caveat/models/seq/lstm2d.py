import torch
from torch import nn, tensor
from torchmetrics.classification import MulticlassHammingDistance
from torchmetrics.regression import MeanSquaredError

from caveat import current_device
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
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, max_length
    ):
        """LSTM Decoder with teacher forcings.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            num_layers (int): number of lstm layers.
            max_length (int): max length of sequences.
        """
        super(Decoder, self).__init__()
        self.current_device = current_device()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.activity_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Sigmoid()

    def forward(self, batch_size, hidden, target=None, **kwargs):
        decoder_input = torch.zeros(batch_size, 1, self.input_size).to(
            device=self.current_device
        )
        decoder_input[:, :, -3] = 1  # set as SOS
        hidden, cell = hidden
        hidden = hidden.contiguous()
        cell = cell.contiguous()
        decoder_hidden = (hidden, cell)
        outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            outputs.append(decoder_output.squeeze())

            if target is not None:
                # teacher forcing for next step
                decoder_input = target[:, i : i + 1, :]  # (slice maintains dim)
            else:
                # no teacher forcing
                decoder_input = decoder_output

        outputs = torch.stack(outputs).permute(1, 0, 2)  # [N, steps, acts]
        return outputs, decoder_hidden, None

    def forward_step(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        prediction = self.fc(output)
        acts, durations = torch.split(
            prediction, [self.output_size - 1, 1], dim=-1
        )
        acts = self.activity_activation(acts)
        durations = self.duration_activation(durations)
        return torch.cat((acts, durations), dim=-1), hidden


class LSTM2d(nn.Module):
    def __init__(
        self,
        in_shape: tuple[int, int],
        latent_dim: int,
        hidden_layers: int,
        hidden_size: int,
        teacher_forcing_ratio: float,
        **kwargs,
    ) -> None:
        """Seq to seq via VAE model.

        Args:
            in_shape (tuple[int, int]): [time_step, activity one-hot encoding].
            latent_dim (int): Latent space size.
            hidden_layers (int): Lstm  layers in encoder and decoder.
            hidden_size (int): Size of lstm layers.
        """
        super(LSTM2d, self).__init__()

        self.MSE = MeanSquaredError()
        self.masked_MSE = nn.MSELoss(reduction="none")
        self.hamming = MulticlassHammingDistance(
            num_classes=in_shape[-1], average="micro"
        )

        self.steps, self.width = in_shape
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.lstm_enc = Encoder(
            input_size=self.width,
            hidden_size=self.hidden_size,
            num_layers=hidden_layers,
        )
        self.lstm_dec = Decoder(
            input_size=self.width,
            hidden_size=self.hidden_size,
            output_size=self.width,
            num_layers=hidden_layers,
            max_length=self.steps,
        )
        flat_size_encode = self.hidden_layers * self.hidden_size * 2
        self.fc_mu = nn.Linear(flat_size_encode, latent_dim)
        self.fc_var = nn.Linear(flat_size_encode, latent_dim)
        self.fc_hidden = nn.Linear(latent_dim, flat_size_encode)

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

    def decode(self, z: tensor, target=None, **kwargs) -> tensor:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        # initialize hidden state as inputs
        h = self.fc_hidden(z)

        # initialize hidden state
        hidden = h.unflatten(
            1, (2 * self.hidden_layers, self.hidden_size)
        ).permute(
            1, 0, 2
        )  # ([2xhidden, N, layers])
        hidden = hidden.split(
            self.hidden_layers
        )  # ([hidden, N, layers, [hidden, N, layers]])
        batch_size = z.shape[0]

        if target is not None and torch.rand(1) < self.teacher_forcing_ratio:
            # use teacher forcing
            reconstruct_output, hidden, _ = self.lstm_dec(
                batch_size=batch_size, hidden=hidden, target=target
            )
        else:
            reconstruct_output, hidden, _ = self.lstm_dec(
                batch_size=batch_size, hidden=hidden, target=None
            )

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

    def forward(self, x: tensor, target=None, **kwargs) -> list[tensor]:
        """Forward pass, also return latent parameterization.

        Args:
            x (tensor): Input sequences [N, steps, acts].

        Returns:
            list[tensor]: [Output [N, steps, acts], Input [N, steps, acts], mu [N, latent_dims], var [N, latent_dims]].
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y = self.decode(z, target=target)
        return [y, x, mu, log_var]

    def unpack_encoding(self, input: tensor) -> tuple[tensor, tensor]:
        """Split the input into activity and duration.

        Args:
            input (tensor): Input sequences [N, steps, acts].

        Returns:
            tuple[tensor, tensor]: [activity [N, steps, acts], duration [N, steps, 1]].
        """
        acts = input[:, :, :-1].contiguous()
        durations = input[:, :, -1:].contiguous()
        return acts, durations

    def pack_encoding(self, acts: tensor, durations: tensor) -> tensor:
        """Pack the activity and duration into input.

        Args:
            acts (tensor): Activity [N, steps, acts].
            durations (tensor): Duration [N, steps, 1].

        Returns:
            tensor: Input sequences [N, steps, acts].
        """
        return torch.cat((acts, durations), dim=-1)

    def loss_function(self, recons, input, mu, log_var, mask, **kwargs) -> dict:
        r"""Computes the VAE loss function.

        Splits the input into activity and duration, and the recons into activity and duration.

        Returns:
            dict: Losses.
        """

        kld_weight = kwargs["kld_weight"]
        duration_weight = kwargs["duration_weight"]
        acts_x, durations_x = self.unpack_encoding(input)
        acts_y, durations_y = self.unpack_encoding(recons)

        # activity encodng
        activity_recons_mse_loss = self.masked_MSE(acts_y, acts_x)
        activity_recons_mse_loss = (
            activity_recons_mse_loss.squeeze() * mask.unsqueeze(1).float()
        ).sum() / mask.sum()

        # duration encodng
        duration_recons_mse_loss = self.masked_MSE(durations_y, durations_x)
        duration_recons_mse_loss = (
            duration_recons_mse_loss.squeeze() * mask.float()
        ).sum() / mask.sum()

        # combined
        recons_mse_loss = (
            activity_recons_mse_loss
            + duration_weight * duration_recons_mse_loss
        )

        recon_argmax = torch.argmax(acts_y, dim=-1)
        input_argmax = torch.argmax(acts_x, dim=-1)

        recons_ham_loss = self.hamming(recon_argmax, input_argmax)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0,
        )

        loss = recons_mse_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "reconstruction_loss": recons_mse_loss.detach(),
            "recons_act_loss": activity_recons_mse_loss.detach(),
            "recons_dur_loss": duration_recons_mse_loss.detach(),
            "recons_ham_loss": recons_ham_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

    def predict_step(
        self, z: tensor, current_device: int, decode: bool = False, **kwargs
    ) -> tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        z = z.to(current_device)
        samples = self.decode(z)
        if decode:
            acts, durations = self.unpack_encoding(samples)
            acts = hot_argmax(acts, -1)
            samples = self.pack_encoding(acts, durations)
        return samples

    def generate(
        self, x: tensor, current_device: int, decode: bool = False, **kwargs
    ) -> tensor:
        """Given an encoder input, return reconstructed output.

        Args:
            x (tensor): [N, steps, acts].

        Returns:
            tensor: [N, steps, acts].
        """
        samples = self.forward(x)[0]
        samples = samples.to(current_device)
        if decode:
            acts, durations = self.unpack_encoding(samples)
            acts = hot_argmax(acts, -1)
            samples = self.pack_encoding(acts, durations)
        return samples
