from typing import List, Optional, Tuple

import torch
from torch import Tensor, exp, nn

from caveat import current_device
from caveat.models import Base, CustomDurationEmbedding


class AutoSeqLSTM(Base):
    def __init__(self, *args, **kwargs):
        """RNN based encoder and decoder with encoder embedding layer and conditionality."""
        super().__init__(*args, **kwargs)
        if self.conditionals_size is None:
            raise UserWarning(
                "ConditionalLSTM requires conditionals_size, please check you have configures a compatible encoder and condition attributes"
            )

    def build(self, **config):
        self.latent_dim = 1  # dummy value for the predict dataloader
        self.hidden_size = config["hidden_size"]
        self.hidden_layers = config["hidden_layers"]
        self.dropout = config["dropout"]
        length, _ = self.in_shape

        self.decoder = Decoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            output_size=self.encodings + 1,
            num_layers=self.hidden_layers,
            max_length=length,
            dropout=self.dropout,
            sos=self.sos,
        )
        self.unflattened_shape = (2 * self.hidden_layers, self.hidden_size)
        flat_size_encode = self.hidden_layers * self.hidden_size * 2
        self.fc_hidden = nn.Linear(self.conditionals_size, flat_size_encode)

    def forward(
        self,
        x: Tensor,
        conditionals: Optional[Tensor] = None,
        target: Optional[Tensor] = None,
        **kwargs,
    ) -> List[Tensor]:

        log_probs = self.decode(z=x, conditionals=conditionals, target=target)
        return [log_probs, Tensor([]), Tensor([]), Tensor([])]

    def loss_function(
        self, log_probs: Tensor, target: Tensor, mask: Tensor, **kwargs
    ) -> dict:
        """Loss function for sequence encoding [N, L, 2]."""
        # unpack act probs and durations
        target_acts, target_durations = self.unpack_encoding(target)
        pred_acts, pred_durations = self.unpack_encoding(log_probs)
        pred_durations = torch.log(pred_durations)

        # activity loss
        recon_act_nlll = self.base_NLLL(
            pred_acts.view(-1, self.encodings), target_acts.view(-1).long()
        )
        recon_act_nlll = (recon_act_nlll * mask.view(-1)).sum() / mask.sum()

        # duration loss
        recon_dur_mse = self.duration_loss_weight * self.MSE(
            pred_durations, target_durations
        )
        recon_dur_mse = (recon_dur_mse * mask).sum() / mask.sum()

        # reconstruction loss
        recons_loss = recon_act_nlll + recon_dur_mse

        return {
            "loss": recons_loss,
            "recon_loss": recons_loss.detach(),
            "recon_act_nlll_loss": recon_act_nlll.detach(),
            "recon_time_mse_loss": recon_dur_mse.detach(),
            "recon_act_ratio": recon_act_nlll / recon_dur_mse,
        }

    def encode(self, input: Tensor):
        return None

    def decode(
        self,
        z: None,
        conditionals: Tensor,
        target: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        h = self.fc_hidden(conditionals)

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
            log_probs = self.decoder(
                batch_size=batch_size, hidden=hidden, target=target
            )
        else:
            log_probs = self.decoder(
                batch_size=batch_size, hidden=hidden, target=None
            )

        return log_probs

    def predict(
        self, z: Tensor, conditionals: Tensor, device: int, **kwargs
    ) -> Tensor:
        z = z.to(device)
        conditionals = conditionals.to(device)
        return exp(self.decode(z=z, conditionals=conditionals, kwargs=kwargs))


class Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        max_length,
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
        self.max_length = max_length
        self.sos = sos

        self.embedding = CustomDurationEmbedding(
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

    def forward(self, batch_size, hidden, target=None, **kwargs):
        hidden, cell = hidden
        decoder_input = torch.zeros(batch_size, 1, 2, device=hidden.device)
        decoder_input[:, :, 0] = self.sos  # set as SOS
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
                decoder_input = self.sample(decoder_output)

        outputs = torch.stack(outputs).permute(1, 0, 2)  # [N, steps, acts]

        acts_logits, durations = torch.split(
            outputs, [self.output_size - 1, 1], dim=-1
        )
        acts_log_probs = self.activity_logprob_activation(acts_logits)
        durations = torch.log(self.duration_activation(durations))
        log_prob_outputs = torch.cat((acts_log_probs, durations), dim=-1)

        return log_prob_outputs

    def forward_step(self, x, hidden):
        # [N, 1, 2]
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        prediction = self.fc(output)
        # [N, 1, encodings+1]
        return prediction, hidden

    def sample(self, x):
        # [N, 1, encodings+1]
        acts, duration = torch.split(x, [self.output_size - 1, 1], dim=-1)
        act = torch.multinomial(
            self.activity_prob_activation(acts.squeeze()), 1
        ).unsqueeze(-2)
        # _, topi = acts.topk(1)
        # act = (
        #     topi.squeeze(-1).detach().unsqueeze(-1)
        # )  # detach from history as input
        duration = self.duration_activation(duration)
        outputs = torch.cat((act, duration), dim=-1)
        # [N, 1, 2]
        return outputs
