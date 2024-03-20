from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torchmetrics.classification import MulticlassHammingDistance


class BaseEncoder(nn.Module):
    def __init__(self, **kwargs):
        raise NotImplementedError


class BaseDecoder(nn.Module):
    def __init__(self, **kwargs):
        raise NotImplementedError


class OneHotEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, dropout: float = 0.1):
        """Embedding that combines activity onehot embedding and duration."""
        super().__init__()
        if hidden_size != input_size + 1:
            raise ValueError("Hidden size must be equal to input size plus 1.")
        self.classes = input_size
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded, durations = torch.split(x, [1, 1], dim=-1)
        embedded = self.dropout(
            nn.functional.one_hot(embedded.int(), self.classes)
        )
        embedded = torch.cat((embedded, durations), dim=-1)
        return embedded


class OneHotPlusLinearEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, dropout: float = 0.1):
        """Embedding that combines activity onehot embedding and duration and linear layer."""
        super().__init__()
        self.classes = input_size
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded, durations = torch.split(x, [1, 1], dim=-1)
        embedded = self.dropout(
            nn.functional.one_hot(embedded.int(), self.classes)
        )
        embedded = torch.cat((embedded, durations), dim=-1)
        embedded = self.fc(embedded)
        return embedded


class CustomDurationEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, dropout: float = 0.1):
        """Embedding that combines activity embedding layer and duration."""
        super().__init__()
        if hidden_size < 2:
            raise ValueError("Hidden size must be greater than 1.")
        self.embedding = nn.Embedding(input_size, hidden_size - 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded, durations = torch.split(x, [1, 1], dim=-1)
        embedded = self.dropout(self.embedding(embedded.int())).squeeze(-2)
        embedded = torch.cat((embedded, durations), dim=-1)
        return embedded


class CustomCombinedEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, dropout: float = 0.1):
        """Embedding that combines activity embedding layer and duration and end time."""
        super().__init__()
        if hidden_size < 3:
            raise ValueError("Hidden size must be at least 3.")
        self.embedding = nn.Embedding(input_size, hidden_size - 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded, durations = torch.split(x, [1, 1], dim=-1)
        ends = torch.cumsum(durations, dim=-1)
        embedded = self.dropout(self.embedding(embedded.int())).squeeze(-2)
        embedded = torch.cat((embedded, durations, ends), dim=-1)
        return embedded


class CustomLinearEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, dropout: float = 0.1):
        """Embedding that combines activity embedding layer and duration using a linear layer."""
        super().__init__()
        if hidden_size < 2:
            raise ValueError("Hidden size must be greater than 1.")
        self.embedding = nn.Embedding(input_size, hidden_size - 1)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded, durations = torch.split(x, [1, 1], dim=-1)
        embedded = self.dropout(self.embedding(embedded.int())).squeeze(-2)
        embedded = torch.cat((embedded, durations), dim=-1)
        embedded = self.fc(embedded)
        return embedded


class BaseVAE(nn.Module):
    def __init__(
        self,
        in_shape: tuple,
        encodings: int,
        encoding_weights: Optional[Tensor] = None,
        sos: int = 0,
        **config,
    ) -> None:
        """Base VAE.

        Args:
            in_shape (tuple[int, int]): [time_step, activity one-hot encoding].
            encodings (int): Number of activity encodings.
            encoding_weights (tensor): Weights for activity encodings.
            sos (int, optional): Start of sequence token. Defaults to 0.
            config: Additional arguments from config.
        """
        super(BaseVAE, self).__init__()

        self.in_shape = in_shape
        self.encodings = encodings
        self.encoding_weights = encoding_weights

        self.sos = sos
        self.teacher_forcing_ratio = config.get("teacher_forcing_ratio", 0)
        print(f"Using teacher forcing ratio: {self.teacher_forcing_ratio}")
        self.kld_weight = config.get("kld_weight", 0.0001)
        print(f"Using KLD weight: {self.kld_weight}")
        self.duration_weight = config.get("duration_weight", 1)
        print(f"Using duration weight: {self.duration_weight}")
        self.use_mask = config.get("use_mask", True)  # defaults to True
        print(f"Using mask: {self.use_mask}")
        self.use_weighted_loss = config.get(
            "weighted_loss", True
        )  # defaults to True
        print(f"Using weighted loss: {self.use_weighted_loss}")

        if self.use_weighted_loss:
            self.NLLL = nn.NLLLoss(weight=encoding_weights)
        else:
            self.NLLL = nn.NLLLoss()

        self.base_NLLL = nn.NLLLoss(reduction="none")
        self.MSE = nn.MSELoss()
        self.hamming = MulticlassHammingDistance(
            num_classes=encodings, average="micro"
        )

        self.build(**config)

    def build(self, **config):
        self.latent_dim = config["latent_dim"]
        self.hidden_size = config["hidden_size"]
        self.hidden_layers = config["hidden_layers"]
        self.dropout = config["dropout"]
        length, _ = self.in_shape

        self.encoder = BaseEncoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_layers,
            dropout=self.dropout,
        )
        self.decoder = BaseDecoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            output_size=self.encodings + 1,
            num_layers=self.hidden_layers,
            max_length=length,
            sos=self.sos,
        )
        self.unflattened_shape = (self.hidden_layers, self.hidden_size)
        flat_size_encode = self.hidden_layers * self.hidden_size
        self.fc_mu = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_var = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_hidden = nn.Linear(self.latent_dim, flat_size_encode)

    def encode(self, input: Tensor) -> list[Tensor]:
        """Encodes the input by passing through the encoder network.

        Args:
            input (tensor): Input sequence batch [N, steps, acts].

        Returns:
            list[tensor]: Latent layer input (means and variances) [N, latent_dims].
        """
        # [N, L, C]
        hidden = self.encoder(input)
        # [N, flatsize]

        # Split the result into mu and var components
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)

        return [mu, log_var]

    def decode(self, z: Tensor, target=None, **kwargs) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output batch as tuple of log probs and probs ([N, L, C]).
        """
        hidden = self.fc_hidden(z)
        hidden = hidden.unflatten(1, self.unflattened_shape).permute(
            1, 0, 2
        )  # ([2xhidden, N, layers])
        hidden = hidden.split(
            self.hidden_layers
        )  # ([hidden, N, layers, [hidden, N, layers]])
        batch_size = z.shape[0]

        if target is not None and torch.rand(1) < self.teacher_forcing_ratio:
            # attempt to use teacher forcing by passing target
            log_probs, probs = self.decoder(
                batch_size=batch_size, hidden=hidden, target=target
            )
        else:
            log_probs, probs = self.decoder(
                batch_size=batch_size, hidden=hidden, target=None
            )

        return log_probs, probs

    def forward(
        self,
        x: Tensor,
        conditionals: Optional[Tensor] = None,
        target=None,
        **kwargs,
    ) -> List[Tensor]:
        """Forward pass, also return latent parameterization.

        Args:
            x (tensor): Input sequences [N, L, Cin].

        Returns:
            list[tensor]: [Log probs, Probs [N, L, Cout], Input [N, L, Cin], mu [N, latent], var [N, latent]].
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        log_prob_y, prob_y = self.decode(
            z, conditionals=conditionals, target=target
        )
        return [log_prob_y, prob_y, mu, log_var]

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
        """Computes the loss function. Different models are expected to need different loss functions
        depending on the data structure. Typically it will either be a sequence encoding [N, L, 2],
        or discretized encoding [N, L, C] or [N, L].

        The default is to use the sequence loss function. But child classes can override this method.

        Returns losses as a dictionary. Which must include the keys "loss" and "recon_loss".

        Args:
            log_probs (Tensor): Log probabilities of the output.
            probs (Tensor): Probabilities of the output.
            mu (Tensor): Latent layer means.
            log_var (Tensor): Latent layer log variances.
            target (Tensor): Target sequences.
            mask (Tensor): Input mask.

        Returns:
            dict: Losses.
        """

        return self.weighted_seq_loss(
            log_probs, probs, mu, log_var, target, mask, **kwargs
        )

    def unweighted_seq_loss(
        self, log_probs, probs, mu, log_var, target, mask, **kwargs
    ) -> dict:
        """Loss function for sequence encoding [N, L, 2]."""

        # unpack act probs and durations
        target_acts, target_durations = self.unpack_encoding(target)
        pred_acts, pred_durations = self.unpack_encoding(log_probs)

        if self.use_mask:  # default is to use masking
            flat_mask = mask.view(-1).bool()
        else:
            flat_mask = torch.ones_like(target_acts).view(-1).bool()

        # activity loss
        recon_act_nlll = self.NLLL(
            pred_acts.view(-1, self.encodings)[flat_mask],
            target_acts.view(-1).long()[flat_mask],
        )

        # duration loss
        recon_dur_mse = self.duration_weight * self.MSE(
            pred_durations.view(-1)[flat_mask],
            target_durations.view(-1)[flat_mask],
        )

        # reconstruction loss
        recons_loss = recon_act_nlll + recon_dur_mse

        # # hamming distance
        # recon_argmax = torch.argmax(pred_acts, dim=-1)
        # recon_act_ham = self.hamming(recon_argmax, target_acts.squeeze().long())

        # kld loss
        norm_kld_weight = self.kld_weight * self.latent_dim

        kld_loss = norm_kld_weight * torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0,
        )

        return {
            "loss": recons_loss + kld_loss,
            "KLD": kld_loss.detach(),
            "recon_loss": recons_loss.detach(),
            "recon_act_nlll_loss": recon_act_nlll.detach(),
            "recon_time_mse_loss": recon_dur_mse.detach(),
            "norm_kld_weight": torch.tensor([norm_kld_weight]).float(),
            "recon_act_ratio": recon_act_nlll / recon_dur_mse,
            "recon_kld_ratio": recons_loss / kld_loss,
        }

    def weighted_seq_loss(
        self, log_probs, probs, mu, log_var, target, weights, **kwargs
    ) -> dict:
        """Loss function for sequence encoding [N, L, 2]."""
        # unpack act probs and durations
        target_acts, target_durations = self.unpack_encoding(target)
        pred_acts, pred_durations = self.unpack_encoding(log_probs)

        # activity loss
        recon_act_nlll = self.base_NLLL(
            pred_acts.view(-1, self.encodings), target_acts.view(-1).long()
        )
        recon_act_nlll = (
            recon_act_nlll * weights.view(-1)
        ).sum() / weights.sum()

        # duration loss
        recon_dur_mse = self.duration_weight * self.MSE(
            pred_durations, target_durations
        )
        recon_dur_mse = (recon_dur_mse * weights).sum() / weights.sum()

        # reconstruction loss
        recons_loss = recon_act_nlll + recon_dur_mse

        # kld loss
        norm_kld_weight = self.kld_weight * self.latent_dim

        kld_loss = norm_kld_weight * torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0,
        )

        return {
            "loss": recons_loss + kld_loss,
            "KLD": kld_loss.detach(),
            "recon_loss": recons_loss.detach(),
            "recon_act_nlll_loss": recon_act_nlll.detach(),
            "recon_time_mse_loss": recon_dur_mse.detach(),
            "norm_kld_weight": torch.tensor([norm_kld_weight]).float(),
            "recon_act_ratio": recon_act_nlll / recon_dur_mse,
            "recon_kld_ratio": recons_loss / kld_loss,
        }

    def end_time_seq_loss(
        self, log_probs, probs, mu, log_var, target, weights, **kwargs
    ) -> dict:
        """Loss function for sequence encoding [N, L, 2]."""
        # unpack act probs and durations
        target_acts, target_durations = self.unpack_encoding(target)
        pred_acts, pred_durations = self.unpack_encoding(log_probs)

        # activity loss
        recon_act_nlll = self.base_NLLL(
            pred_acts.view(-1, self.encodings), target_acts.view(-1).long()
        )
        recon_act_nlll = (
            recon_act_nlll * weights.view(-1)
        ).sum() / weights.sum()

        # ends loss
        target_ends = torch.cumsum(target_durations, dim=-1)
        pred_ends = torch.cumsum(pred_durations, dim=-1)

        recon_end_mse = self.duration_weight * self.MSE(pred_ends, target_ends)
        recon_end_mse = (recon_end_mse * weights).sum() / weights.sum()

        # reconstruction loss
        recons_loss = recon_act_nlll + recon_end_mse

        # kld loss
        norm_kld_weight = self.kld_weight * self.latent_dim

        kld_loss = norm_kld_weight * torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0,
        )

        return {
            "loss": recons_loss + kld_loss,
            "KLD": kld_loss.detach(),
            "recon_loss": recons_loss.detach(),
            "recon_act_nlll_loss": recon_act_nlll.detach(),
            "recon_time_mse_loss": recon_end_mse.detach(),
            "norm_kld_weight": torch.tensor([norm_kld_weight]).float(),
            "recon_act_ratio": recon_act_nlll / recon_end_mse,
            "recon_kld_ratio": recons_loss / kld_loss,
        }

    def combined_seq_loss(
        self, log_probs, probs, mu, log_var, target, weights, **kwargs
    ) -> dict:
        """Loss function for sequence encoding [N, L, 2]."""
        # unpack act probs and durations
        target_acts, target_durations = self.unpack_encoding(target)
        pred_acts, pred_durations = self.unpack_encoding(log_probs)

        # activity loss
        recon_act_nlll = self.base_NLLL(
            pred_acts.view(-1, self.encodings), target_acts.view(-1).long()
        )
        recon_act_nlll = (
            recon_act_nlll * weights.view(-1)
        ).sum() / weights.sum()

        # duration loss
        recon_dur_mse = self.duration_weight * self.MSE(
            pred_durations, target_durations
        )
        recon_dur_mse = (recon_dur_mse * weights).sum() / weights.sum()

        # ends loss
        target_ends = torch.cumsum(target_durations, dim=-1)
        pred_ends = torch.cumsum(pred_durations, dim=-1)

        recon_end_mse = self.duration_weight * self.MSE(pred_ends, target_ends)
        recon_end_mse = (recon_end_mse * weights).sum() / weights.sum()

        # combined time loss
        recon_time_mse = (0.5 * recon_dur_mse) + (0.5 * recon_end_mse)

        # reconstruction loss
        recons_loss = recon_act_nlll + recon_time_mse

        # kld loss
        norm_kld_weight = self.kld_weight * self.latent_dim

        kld_loss = norm_kld_weight * torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0,
        )

        return {
            "loss": recons_loss + kld_loss,
            "KLD": kld_loss.detach(),
            "recon_loss": recons_loss.detach(),
            "recon_act_nlll_loss": recon_act_nlll.detach(),
            "recon_time_mse_loss": recon_time_mse.detach(),
            "norm_kld_weight": torch.tensor([norm_kld_weight]).float(),
            "recon_act_ratio": recon_act_nlll / recon_time_mse,
            "recon_kld_ratio": recons_loss / kld_loss,
        }

    def discretized_loss(
        self, log_probs, probs, mu, log_var, target, mask, **kwargs
    ) -> dict:
        """Loss function for discretized encoding [N, L]."""
        # activity loss
        recon_act_nlll = self.NLLL(
            log_probs.squeeze().permute(0, 2, 1), target.long()
        )

        # recon_argmax = probs.squeeze().argmax(dim=-1)
        # recon_act_ham = self.hamming(recon_argmax, input.long())

        # kld loss
        norm_kld_weight = self.kld_weight
        kld_loss = norm_kld_weight * torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0,
        )

        # loss
        loss = recon_act_nlll + kld_loss

        return {
            "loss": loss,
            "recon_loss": recon_act_nlll,
            "recon_act_nlll_loss": recon_act_nlll,
            "KLD": kld_loss,
            "norm_kld_weight": torch.tensor([norm_kld_weight]),
            "recon_kld_ratio": recon_act_nlll / kld_loss,
        }

    def discretized_loss_encoded(
        self, log_probs, probs, mu, log_var, target, mask, **kwargs
    ) -> dict:
        """Computes the loss function for discretized encoding [N, L, C]."""

        target_argmax = target.squeeze().argmax(dim=-1)
        return self.discretized_loss(
            log_probs, probs, mu, log_var, target_argmax, mask, **kwargs
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Re-parameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu (tensor): Mean of the latent Gaussian [N x latent_dims].
            logvar (tensor): Standard deviation of the latent Gaussian [N x latent_dims].

        Returns:
            tensor: [N x latent_dims].
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def unpack_encoding(self, input: Tensor) -> tuple[Tensor, Tensor]:
        """Split the input into activity and duration.

        Args:
            input (tensor): Input sequences [N, steps, acts].

        Returns:
            tuple[tensor, tensor]: [activity [N, steps, acts], duration [N, steps, 1]].
        """
        acts = input[:, :, :-1].contiguous()
        durations = input[:, :, -1:].squeeze(-1).contiguous()
        return acts, durations

    def pack_encoding(self, acts: Tensor, durations: Tensor) -> Tensor:
        """Pack the activity and duration into input.

        Args:
            acts (tensor): Activity [N, steps, acts].
            durations (tensor): Duration [N, steps, 1].

        Returns:
            tensor: Input sequences [N, steps, acts].
        """
        if len(durations.shape) == 2:
            durations = durations.unsqueeze(-1)
        return torch.cat((acts, durations), dim=-1)

    def predict_step(self, z: Tensor, current_device: int, **kwargs) -> Tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        z = z.to(current_device)
        prob_samples = self.decode(z, **kwargs)[1]
        return prob_samples

    def generate(self, x: Tensor, current_device: int, **kwargs) -> Tensor:
        """Given an encoder input, return reconstructed output.

        Args:
            x (tensor): [N, steps, acts].

        Returns:
            tensor: [N, steps, acts].
        """
        prob_samples = self.forward(x, **kwargs)[1]
        prob_samples = prob_samples.to(current_device)
        return prob_samples
