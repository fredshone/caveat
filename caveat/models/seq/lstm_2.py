import torch
from torch import nn, tensor
from torchmetrics.classification import MulticlassHammingDistance
from torchmetrics.regression import MeanSquaredError

from caveat.models.seq.lstm import SEQVAE, Encoder
from caveat.models.utils import hot_argmax


class Decoder(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, max_length
    ):
        """LSTM Decoder.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            num_layers (int): number of lstm layers.
            max_length (int): max length of sequences.
        """
        super(Decoder, self).__init__()
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
        self.activity_activation = nn.Softmax(dim=-1)
        self.duration_activation = nn.Sigmoid()

    def forward(self, x, hidden, target=None):
        batch_size = x.size(0)
        decoder_input = torch.empty(batch_size, self.input_size)
        decoder_input[:, -3] = 1  # SOS
        hidden, cell = hidden
        hidden = hidden[
            :, -1, :
        ]  # todo reduce hidden size input (remove from latent space)
        cell = cell[:, -1, :]
        decoder_hidden = (hidden, cell)
        decoder_outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            decoder_outputs.append(decoder_output)

            if target is not None:
                print("Forcing")
                # teacher forcing
                decoder_input = target[:, i, :].unsqueeze(1)
            else:
                # no teacher forcing
                decoder_input = decoder_output

        decoder_outputs = torch.stack(decoder_outputs).permute(1, 0, 2)
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        prediction = self.fc(output)
        acts, durations = torch.split(
            prediction, [self.output_size - 1, 1], dim=-1
        )
        acts = self.activity_activation(acts)
        durations = self.duration_activation(durations)
        return torch.cat((acts, durations), dim=-1), hidden


class SEQVAESEQ(SEQVAE):
    def __init__(
        self,
        in_shape: tuple[int, int],
        latent_dim: int,
        hidden_layers: int,
        hidden_size: int,
        teacher_forcing_ratio: float,
        **kwargs,
    ) -> None:
        """Image LSTM VAE model.

        Args:
            in_shape (tuple[int, int]): [time_step, activity one-hot encoding].
            latent_dim (int): Latent space size.
            hidden_layers (int): Lstm  layers in encoder and decoder.
            hidden_size (int): Size of lstm layers.
        """
        super(SEQVAE, self).__init__()

        self.MSE = MeanSquaredError()
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

        reconstruct_output, hidden, _ = self.lstm_dec(z, hidden)

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
        y = self.decode(z)
        return [y, input, mu, log_var]

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

    def loss_function(self, recons, input, mu, log_var, **kwargs) -> dict:
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
        activity_recons_mse_loss = self.MSE(acts_y, acts_x)
        # duration encodng
        duration_recons_mse_loss = self.MSE(durations_y, durations_x)
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

    def predict_step(self, z: tensor, current_device: int, **kwargs) -> tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        z = z.to(current_device)
        samples = self.decode(z)
        acts, durations = self.unpack_encoding(samples)
        acts = hot_argmax(acts, -1)
        return self.pack_encoding(acts, durations)

    def generate(self, x: tensor, current_device: int, **kwargs) -> tensor:
        """Given an encoder input, return reconstructed output.

        Args:
            x (tensor): [N, steps, acts].

        Returns:
            tensor: [N, steps, acts].
        """
        samples = self.forward(x)[0]
        samples = samples.to(current_device)
        acts, durations = self.unpack_encoding(samples)
        acts = hot_argmax(acts, -1)
        return self.pack_encoding(acts, durations)
