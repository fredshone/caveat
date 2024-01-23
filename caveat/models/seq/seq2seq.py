from typing import Tuple

import torch
from torch import Tensor, nn

from caveat import current_device
from caveat.models.base import BaseVAE, CustomEmbedding


class SEQ2SEQ(BaseVAE):
    def __init__(self, *args, **kwargs):
        """RNN based encoder and decoder with encoder embedding layer."""
        super().__init__(*args, **kwargs)

    def build(self, **config):
        self.latent_dim = config["latent_dim"]
        self.hidden_size = config["hidden_size"]
        self.hidden_layers = config["hidden_layers"]
        self.dropout = config["dropout"]
        length, _ = self.in_shape

        self.unflattened_shape = (2 * self.hidden_layers, self.hidden_size)
        flat_size_encode = self.hidden_layers * self.hidden_size * 2

        self.encoder = Encoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_layers,
            dropout=self.dropout,
        )
        self.decoder = Decoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            output_size=self.encodings + 1,
            num_layers=self.hidden_layers,
            max_length=length,
            latent_dim=self.latent_dim,
            flat_size_encode=flat_size_encode,
            unflattened_shape=self.unflattened_shape,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            sos=self.sos,
        )

        self.fc_mu = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_var = nn.Linear(flat_size_encode, self.latent_dim)

        if config.get("share_embed", False):
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def decode(self, z: Tensor, target=None, **kwargs) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        # initialize hidden state as inputs

        log_probs, probs = self.decoder(z)

        return log_probs, probs


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
            dropout (float): dropout. Defaults to 0.1.
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = CustomEmbedding(
            input_size, hidden_size, dropout=dropout
        )
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h1, h2) = self.lstm(embedded)
        # ([layers, N, C (output_size)], [layers, N, C (output_size)])
        h1 = self.norm(h1)
        h2 = self.norm(h2)
        hidden = torch.cat((h1, h2)).permute(1, 0, 2).flatten(start_dim=1)
        # [N, flatsize]
        return hidden


class Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        max_length,
        latent_dim,
        flat_size_encode,
        unflattened_shape,
        hidden_layers,
        dropout: float = 0.0,
        sos: int = 0,
    ):
        """LSTM Decoder with teacher forcing.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            num_layers (int): number of lstm layers.
            max_length (int): max length of sequences.
            dropout (float): dropout probability. Defaults to 0.
        """
        super(Decoder, self).__init__()
        self.current_device = current_device()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.unflattened_shape = unflattened_shape
        self.hidden_layers = hidden_layers
        self.max_length = max_length
        self.sos = sos

        self.fc_hidden = nn.Linear(latent_dim, flat_size_encode)
        self.fc_encode = nn.Linear(latent_dim, hidden_size)

        self.embedding = CustomEmbedding(
            input_size, hidden_size, dropout=dropout
        )
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.activity_prob_activation = nn.Softmax(dim=-1)
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Sigmoid()

    def forward(self, x, target=None, **kwargs):
        batch_size = x.shape[0]
        hidden = self.fc_hidden(x)

        # initialize hidden state
        hidden = hidden.unflatten(1, self.unflattened_shape).permute(
            1, 0, 2
        )  # ([2xhiddenlayers, N, hiddensize])
        hidden, cell = hidden.split(
            self.hidden_layers
        )  # [hidden, N, layers],[hidden, N, layers]

        hidden = hidden.contiguous()
        cell = cell.contiguous()
        x = self.fc_encode(x)  # [N, C]
        x = x.reshape(batch_size, 1, self.hidden_size).repeat(
            1, self.max_length, 1
        )  # [N, L, C]
        output, _ = self.lstm(x, (hidden, cell))  # [N, L, C]
        output = self.fc(output)  # [N, L, C]

        acts_logits, durations = torch.split(
            output, [self.output_size - 1, 1], dim=-1
        )
        acts_probs = self.activity_prob_activation(acts_logits)
        acts_log_probs = self.activity_logprob_activation(acts_logits)
        durations = self.duration_activation(durations)
        log_prob_outputs = torch.cat((acts_log_probs, durations), dim=-1)
        prob_outputs = torch.cat((acts_probs, durations), dim=-1)

        return log_prob_outputs, prob_outputs

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
