from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics.classification import MulticlassHammingDistance

from caveat.models.sequence.lstm import Encoder


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        max_length: int,
        sos: int,
        dropout: float = 0.1,
    ):
        super(AttnDecoderRNN, self).__init__()
        self.max_length = max_length
        self.sos = sos
        self.embedding = nn.Embedding(input_size, hidden_size - 1)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_size, encoder_outputs, encoder_hidden, target=None):
        decoder_input = torch.zeros(
            batch_size, 1, 2, dtype=torch.long, device=self.current_device
        )
        decoder_input[:, :, 0] = self.sos
        hidden, cell = encoder_hidden
        hidden = hidden.contiguous()
        cell = cell.contiguous()
        decoder_hidden = (hidden, cell)
        decoder_outputs = []
        attentions = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output.squeeze())
            attentions.append(attn_weights.squeeze())

            if target is not None:
                # teacher forcing for next step
                decoder_input = target[:, i : i + 1, :]  # (slice maintains dim)
            else:
                # no teacher forcing use decoder output
                decoder_input = self.pack(decoder_output)

        decoder_outputs = torch.stack(decoder_outputs).permute(
            1, 0, 2
        )  # [N, steps, acts]
        attentions = torch.stack(attentions).permute(
            1, 0, 2
        )  # [N, steps, encodings]

        acts_logits, durations = torch.split(
            decoder_outputs, [self.output_size - 1, 1], dim=-1
        )
        acts_probs = self.activity_prob_activation(acts_logits)
        acts_log_probs = self.activity_logprob_activation(acts_logits)
        durations = self.duration_activation(durations)
        log_prob_outputs = torch.cat((acts_log_probs, durations), dim=-1)
        prob_outputs = torch.cat((acts_probs, durations), dim=-1)

        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return log_prob_outputs, prob_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        # input: [N, 1, 2]
        embedded, durations = torch.split(input, [1, 1], dim=-1)
        embedded = self.dropout(self.embedding(embedded.int())).squeeze(-2)
        embedded = torch.cat((embedded, durations), dim=-1)

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

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


class Transformer(nn.Module):
    def __init__(
        self,
        in_shape: tuple[int, int],
        latent_dim: int,
        hidden_layers: int,
        hidden_size: int,
        teacher_forcing_ratio: float,
        encodings: int,
        encoding_weights: Optional[Tensor],
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        """Seq to seq via VAE model.

        Args:
            in_shape (tuple[int, int]): [time_step, activity one-hot encoding].
            latent_dim (int): Latent space size.
            hidden_layers (int): Lstm  layers in encoder and decoder.
            hidden_size (int): Size of lstm layers.
        """
        super(Transformer, self).__init__()
        self.steps, self.width = in_shape
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encodings = encodings
        self.encoding_weights = encoding_weights

        if (
            kwargs.get("weighted_loss") is not False
        ):  # default to use weightings
            if encoding_weights is None:
                raise ValueError(
                    "weighted_loss is True but encoding_weights is None"
                )
            self.NLLL = nn.NLLLoss(weight=encoding_weights)
        else:
            self.NLLL = nn.NLLLoss(weight=None)
        self.MSE = nn.MSELoss()
        self.hamming = MulticlassHammingDistance(
            num_classes=encodings, average="micro"
        )

        self.use_mask = kwargs.get("use_mask", True)  # deafult to use mask

        self.encoder = Encoder(
            input_size=encodings,
            hidden_size=self.hidden_size,
            num_layers=hidden_layers,
            dropout=dropout,
        )
        self.decoder = AttnDecoderRNN(
            input_size=encodings,
            hidden_size=self.hidden_size,
            output_size=encodings + 1,  # act encoding plus one for duration
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
        hidden = self.encoder(input)
        # flatten last hidden cell layer
        flat = torch.cat(hidden).permute(1, 0, 2).flatten(start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(flat)
        log_var = self.fc_var(flat)

        return [mu, log_var]

    def decode(self, z: Tensor, target=None, **kwargs) -> Tuple[Tensor, Tensor]:
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
            log_probs, probs = self.decoder(
                batch_size=batch_size, hidden=hidden, target=target
            )
        else:
            log_probs, probs = self.decoder(
                batch_size=batch_size, hidden=hidden, target=None
            )

        return log_probs, probs

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

    def forward(self, x: Tensor, target=None, **kwargs) -> List[Tensor]:
        """Forward pass, also return latent parameterization.

        Args:
            x (tensor): Input sequences [N, steps, acts].

        Returns:
            list[tensor]: [Output [N, steps, acts], Input [N, steps, acts], mu [N, latent_dims], var [N, latent_dims]].
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        log_prob_y, prob_y = self.decode(z, target=target)
        return [log_prob_y, prob_y, x, mu, log_var]

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

    def loss_function(
        self, log_probs, _, input, mu, log_var, mask, **kwargs
    ) -> dict:
        r"""Computes the VAE loss function.

        Splits the input into activity and duration, and the recons into activity and duration.

        Returns:
            dict: Losses.
        """

        kld_weight = kwargs["kld_weight"]
        duration_weight = kwargs["duration_weight"]

        # unpack act probs and durations
        target_acts, target_durations = self.unpack_encoding(input)
        pred_acts, pred_durations = self.unpack_encoding(log_probs)

        if self.use_mask:  # default is to use masking
            flat_mask = mask.view(-1).bool()
        else:
            flat_mask = torch.ones_like(target_acts).view(-1).bool()

        # activity encodng
        recon_acts_nlll = self.NLLL(
            pred_acts.view(-1, self.encodings)[flat_mask],
            target_acts.view(-1).long()[flat_mask],
        )

        # duration encodng
        recon_dur_mse = duration_weight * self.MSE(
            pred_durations.view(-1)[flat_mask],
            target_durations.view(-1)[flat_mask],
        )

        # combined
        recons_loss = recon_acts_nlll + recon_dur_mse

        recon_argmax = torch.argmax(pred_acts, dim=-1)
        recons_ham_loss = self.hamming(
            recon_argmax, target_acts.squeeze().long()
        )

        kld_loss = kld_weight * torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
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

    def predict_step(self, z: Tensor, current_device: int, **kwargs) -> Tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        z = z.to(current_device)
        prob_samples = self.decode(z)[1]
        return prob_samples

    def generate(self, x: Tensor, current_device: int, **kwargs) -> Tensor:
        """Given an encoder input, return reconstructed output.

        Args:
            x (tensor): [N, steps, acts].

        Returns:
            tensor: [N, steps, acts].
        """
        prob_samples = self.forward(x)[1]
        prob_samples = prob_samples.to(current_device)
        return prob_samples
