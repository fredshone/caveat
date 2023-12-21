import torch
from torch import Tensor, nn
from torchmetrics.classification import MulticlassHammingDistance
from torchmetrics.regression import MeanSquaredError

from caveat import current_device
from caveat.models.utils import hot_argmax


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        """LSTM Encoder.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            num_layers (int): number of lstm layers.
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size - 1)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded, durations = torch.split(x, [1, 1], dim=-1)
        embedded = self.dropout(self.embedding(embedded.int())).squeeze()
        embedded = torch.cat((embedded, durations), dim=-1)
        _, hidden = self.lstm(embedded)
        return hidden


class Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        max_length,
        sos: int = 0,
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
        self.sos = sos

        self.embedding = nn.Embedding(input_size, hidden_size - 1)
        self.activate = nn.ReLU()
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.activity_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Sigmoid()

    def forward(self, batch_size, hidden, target=None, **kwargs):
        decoder_input = torch.zeros(batch_size, 1, 2).to(
            device=self.current_device
        )
        decoder_input[:, :, 0] = self.sos  # set as SOS
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
                # no teacher forcing use decoder output
                decoder_input = self.pack(decoder_output)

        outputs = torch.stack(outputs).permute(1, 0, 2)  # [N, steps, acts]

        acts, durations = torch.split(
            outputs, [self.output_size - 1, 1], dim=-1
        )
        acts = self.activity_activation(acts)
        durations = self.duration_activation(durations)
        outputs = torch.cat((acts, durations), dim=-1)

        return outputs, decoder_hidden, None

    def forward_step(self, x, hidden):
        # [N, 1, 2]
        embedded, durations = torch.split(x, [1, 1], dim=-1)
        embedded = self.activate(self.embedding(embedded.int())).squeeze(-2)
        embedded = torch.cat((embedded, durations), dim=-1)
        output, hidden = self.lstm(embedded, hidden)
        prediction = self.fc(output)
        # [N, 1, encodings+1]
        return prediction, hidden

    def pack(self, x):
        # [N, 1, encodings+1]
        acts, duration = torch.split(x, [self.output_size - 1, 1], dim=-1)
        _, topi = acts.topk(1)
        act = (
            topi.squeeze(-1).detach().unsqueeze(-1)
        )  # detach from history as input
        duration = self.duration_activation(duration)
        outputs = torch.cat((act, duration), dim=-1)
        # [N, 1, 2]
        return outputs


class LSTM(nn.Module):
    def __init__(
        self,
        in_shape: tuple[int, int],
        latent_dim: int,
        hidden_layers: int,
        hidden_size: int,
        teacher_forcing_ratio: float,
        encodings: int,
        **kwargs,
    ) -> None:
        """Seq to seq via VAE model.

        Args:
            in_shape (tuple[int, int]): [time_step, activity one-hot encoding].
            latent_dim (int): Latent space size.
            hidden_layers (int): Lstm  layers in encoder and decoder.
            hidden_size (int): Size of lstm layers.
        """
        super(LSTM, self).__init__()

        self.MSE = MeanSquaredError()
        self.NLLL = nn.NLLLoss()
        self.masked_MSE = nn.MSELoss(reduction="none")
        self.hamming = MulticlassHammingDistance(
            num_classes=encodings, average="micro"
        )

        self.steps, self.width = in_shape
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encodings = encodings

        self.lstm_enc = Encoder(
            input_size=encodings,
            hidden_size=self.hidden_size,
            num_layers=hidden_layers,
        )
        self.lstm_dec = Decoder(
            input_size=encodings,
            hidden_size=self.hidden_size,
            output_size=encodings + 1,
            num_layers=hidden_layers,
            max_length=self.steps,
            sos=0,
        )
        flat_size_encode = self.hidden_layers * self.hidden_size * 2
        self.fc_mu = nn.Linear(flat_size_encode, latent_dim)
        self.fc_var = nn.Linear(flat_size_encode, latent_dim)
        self.fc_hidden = nn.Linear(latent_dim, flat_size_encode)

    def encode(self, input: Tensor) -> list[Tensor]:
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

    def decode(self, z: Tensor, target=None, **kwargs) -> Tensor:
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

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
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

    def forward(self, x: Tensor, target=None, **kwargs) -> list[Tensor]:
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

    def unpack_encoding(self, input: Tensor) -> tuple[Tensor, Tensor]:
        """Split the input into activity and duration.

        Args:
            input (tensor): Input sequences [N, steps, acts].

        Returns:
            tuple[tensor, tensor]: [activity [N, steps, acts], duration [N, steps, 1]].
        """
        acts = input[:, :, :-1].contiguous()
        durations = input[:, :, -1:].contiguous()
        return acts, durations

    def pack_encoding(self, acts: Tensor, durations: Tensor) -> Tensor:
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
        flat_mask = mask.view(-1).bool()

        target_acts, target_durations = self.unpack_encoding(input)
        pred_acts, pred_durations = self.unpack_encoding(recons)

        # activity encodng
        recon_acts_nlll = self.NLLL(
            pred_acts.view(-1, self.encodings)[flat_mask],
            target_acts.view(-1).long()[flat_mask],
        )
        # todo mask

        # duration encodng
        recon_dur_mse = self.masked_MSE(pred_durations, target_durations)
        recon_dur_mse = (
            duration_weight
            * (recon_dur_mse.squeeze() * mask.float()).sum()
            / mask.sum()
        )

        # combined
        recons_loss = recon_acts_nlll + recon_dur_mse

        recon_argmax = torch.argmax(pred_acts, dim=-1)
        recons_ham_loss = self.hamming(
            recon_argmax, target_acts.squeeze().long()
        )

        kld_loss = kld_weight * torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0,
        )

        loss = recons_loss + kld_loss
        return {
            "loss": loss,
            "recon_loss": recons_loss.detach(),
            "recon_act_loss": recon_acts_nlll.detach(),
            "recon_dur_loss": recon_dur_mse.detach(),
            "recons_ham_loss": recons_ham_loss.detach(),
            "KLD": kld_loss.detach(),
        }

    def predict_step(
        self, z: Tensor, current_device: int, decode: bool = False, **kwargs
    ) -> Tensor:
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
        self, x: Tensor, current_device: int, decode: bool = False, **kwargs
    ) -> Tensor:
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
