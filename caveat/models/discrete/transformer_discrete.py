from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from caveat import current_device
from caveat.models.base import BaseVAE


class AttentionDiscrete(BaseVAE):
    def __init__(self, *args, **kwargs):
        """RNN based encoder and decoder with encoder embedding layer."""
        super().__init__(*args, **kwargs)

    def build(self, **config):
        self.latent_dim = config["latent_dim"]
        self.hidden_size = config["hidden_size"]
        self.heads = config["heads"]
        self.hidden_layers = config["hidden_layers"]
        self.dropout = config["dropout"]
        self.length = self.in_shape[0]

        self.encoder = AttentionEncoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            length=self.length,
            n_head=self.heads,
            n_layer=self.hidden_layers,
            dropout=self.dropout,
        )
        self.decoder = AttentionDecoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            output_size=self.encodings,
            num_heads=self.heads,
            num_layers=self.hidden_layers,
            length=self.length,
            dropout=self.dropout,
        )
        self.unflattened_shape = (self.length, self.hidden_size)
        flat_size_encode = self.length * self.hidden_size
        self.fc_mu = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_var = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_hidden = nn.Linear(self.latent_dim, flat_size_encode)

        if config.get("share_embed", False):
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def forward(self, x: Tensor, teacher=None, **kwargs) -> List[Tensor]:
        """Forward pass, also return latent parameterization.

        Args:
            x (tensor): Input sequences [N, L, Cin].

        Returns:
            list[tensor]: [Log probs, Probs [N, L, Cout], Input [N, L, Cin], mu [N, latent], var [N, latent]].
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)

        if teacher is not None:  # training
            log_prob_y, prob_y = self.decode(z, teacher=teacher)
            return [log_prob_y, prob_y, mu, log_var]

        # no target so assume generating
        log_prob, prob = self.predict(z, current_device=z.device)
        return [log_prob, prob, mu, log_var]

    def decode(
        self, z: Tensor, teacher=None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        # initialize hidden state as inputs
        hidden = self.fc_hidden(z)
        hidden = hidden.unflatten(1, self.unflattened_shape)
        log_probs, probs = self.decoder(hidden, teacher)

        return log_probs, probs

    def loss_function(
        self,
        log_probs: Tensor,
        probs: Tensor,
        mu: Tensor,
        log_var: Tensor,
        target: Tensor,
        mask: Tensor,
        **kwargs,
    ) -> dict:
        return self.discretized_loss(
            log_probs, probs, mu, log_var, target, mask, **kwargs
        )

    def predict_step(self, z: Tensor, current_device: int, **kwargs) -> Tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        _, prob_samples = self.predict(z, current_device)
        return prob_samples

    def predict(
        self, z: Tensor, current_device: int, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        z = z.to(current_device)
        B = z.shape[0]
        log_outputs = []
        outputs = []
        sequences = torch.zeros(B, 1, device=z.device)
        for _ in range(
            self.length
        ):  # todo: need a SOS encoding!!!!!!!!!!!!!!!!
            # get the predictions
            log_probs, probs = self.decode(z, teacher=sequences)
            log_outputs.append(log_probs)
            outputs.append(probs)
            # focus only on the last time step
            last_probs = probs[:, -1, :]  # becomes (B, C)
            # sample from the distribution
            next = torch.multinomial(last_probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            sequences = torch.cat((sequences, next), dim=1)  # (B, T+1)

        # prob_samples = prob_samples.unsqueeze(1)  # to match cnn for decoder
        return log_probs, probs

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


class AttentionHead(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd=10, block_size=128, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd=10, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(head_size=head_size, n_embd=n_embd)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MaskedAttentionHead(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd=10, block_size=128, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadMaskedAttention(nn.Module):
    """multiple heads of masked self-attention in parallel"""

    def __init__(
        self, num_heads, head_size, block_size, n_embd=10, dropout=0.0
    ):
        super().__init__()
        self.masked_heads = nn.ModuleList(
            [
                MaskedAttentionHead(
                    head_size=head_size, n_embd=n_embd, block_size=block_size
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.masked_heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class CrossAttentionHead(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd=10, block_size=128, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_encode, x_decode):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        k = self.key(x_encode)  # (B,T,hs)
        q = self.query(x_decode)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x_encode)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadCrossAttention(nn.Module):
    """multiple heads of masked self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd=10, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CrossAttentionHead(head_size=head_size, n_embd=n_embd)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_encode, x):
        out = torch.cat([h(x_encode, x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            num_heads=n_head,
            head_size=head_size,
            n_embd=n_embd,
            dropout=dropout,
        )
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadMaskedAttention(
            num_heads=n_head,
            head_size=head_size,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
        )
        self.ca = MultiHeadCrossAttention(
            num_heads=n_head,
            head_size=head_size,
            n_embd=n_embd,
            dropout=dropout,
        )
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.ln4 = nn.LayerNorm(n_embd)

    def forward(self, x_encode, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ca(self.ln2(x_encode), self.ln3(x))
        x = x + self.ffwd(self.ln4(x))
        return x


class AttentionEncoder(nn.Module):
    def __init__(
        self, input_size, hidden_size, length, n_head, n_layer, dropout
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embed_dropout = nn.Dropout(dropout)
        self.position_embedding = nn.Embedding(length, hidden_size)

        self.blocks = nn.Sequential(
            *[
                EncoderBlock(hidden_size, n_head=n_head, dropout=dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(hidden_size)  # final layer norm
        # self.lm_head = nn.Linear(hidden_size, input_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        _, L = x.shape  # batch size and sequence encoding length

        # idx and targets are both (B,T) tensor of integers
        embedding = self.embedding(x.long())  # (B,T,C)
        pos_emb = self.position_embedding(
            torch.arange(L, device=current_device())
        )  # (T,C)
        x = embedding + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        x = x.flatten(1)

        return x


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_heads,
        num_layers,
        length,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embed_dropout = nn.Dropout(dropout)
        self.position_embedding = nn.Embedding(length, hidden_size)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    hidden_size,
                    n_head=num_heads,
                    dropout=dropout,
                    block_size=length,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, output_size)
        self.activity_prob_activation = nn.Softmax(dim=-1)
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x_encode, x):
        N, L = x.shape  # batch size and sequence encoding length

        # idx and targets are both (B,T) tensor of integers
        embedding = self.embedding(x.long())  # (B,T,C)
        pos_emb = self.position_embedding(
            torch.arange(L, device=current_device())
        )  # (T,C)

        x = embedding + pos_emb  # (B,T,C)

        for layer in self.blocks:
            x = layer(x_encode, x)

        x = self.ln_f(x)  # (B,T,C)
        x = self.lm_head(x)
        # todo get ride of this ^, needs to be done for cnn too

        acts_probs = self.activity_prob_activation(x)
        acts_log_probs = self.activity_logprob_activation(x)

        return acts_log_probs, acts_probs
