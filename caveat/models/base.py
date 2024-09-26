from typing import List, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, exp, nn

from caveat.experiment import Experiment


class BaseEncoder(LightningModule):
    def __init__(self, **kwargs):
        raise NotImplementedError

    def forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        raise NotImplementedError


class BaseDecoder(LightningModule):
    def __init__(self, **kwargs):
        raise NotImplementedError

    def forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        raise NotImplementedError


class Base(Experiment):
    def __init__(
        self,
        in_shape: tuple,
        encodings: int,
        encoding_weights: Optional[Tensor] = None,
        conditionals_size: Optional[tuple] = None,
        sos: int = 0,
        **kwargs,
    ) -> None:
        """Base VAE.

        Args:
            in_shape (tuple[int, int]): [time_step, activity one-hot encoding].
            encodings (int): Number of activity encodings.
            encoding_weights (tensor): Weights for activity encodings.
            conditionals_size (int, optional): Size of conditionals encoding. Defaults to None.
            sos (int, optional): Start of sequence token. Defaults to 0.
            config: Additional arguments from config.
        """
        super(Base, self).__init__(**kwargs)

        self.in_shape = in_shape
        print(f"Found input shape: {self.in_shape}")
        self.encodings = encodings
        self.encoding_weights = encoding_weights
        self.conditionals_size = conditionals_size
        if self.conditionals_size is not None:
            print(f"Found conditionals size: {self.conditionals_size}")
        self.sos = sos

        self.teacher_forcing_ratio = kwargs.get("teacher_forcing_ratio", 0.5)
        print(f"Found teacher forcing ratio: {self.teacher_forcing_ratio}")
        self.kld_weight = kwargs.get("kld_weight", 0.0001)
        print(f"Found KLD weight: {self.kld_weight}")
        self.duration_weight = kwargs.get("duration_weight", 1)
        print(f"Found duration weight: {self.duration_weight}")
        self.use_mask = kwargs.get("use_mask", True)
        print(f"Using mask: {self.use_mask}")
        self.use_weighted_loss = kwargs.get("weighted_loss", True)

        if self.use_weighted_loss and encoding_weights is not None:
            print("Using weighted loss function")
            self.NLLL = nn.NLLLoss(weight=encoding_weights)
        else:
            self.NLLL = nn.NLLLoss()

        self.base_NLLL = nn.NLLLoss(reduction="none")
        self.MSE = nn.MSELoss()

        self.build(**kwargs)

    def build(self, **config):
        self.latent_dim = config["latent_dim"]
        self.hidden_size = config["hidden_size"]
        self.hidden_layers = config["hidden_layers"]
        self.dropout = config.get("dropout", 0)
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
        mu, log_var = self.encode(x, conditionals)
        z = self.reparameterize(mu, log_var)
        log_probs_x = self.decode(z, conditionals=conditionals, target=target)
        return [log_probs_x, mu, log_var, z]

    def encode(
        self, input: Tensor, conditionals: Optional[Tensor]
    ) -> list[Tensor]:
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
        return (eps * std) + mu

    def kld(self, mu: Tensor, log_var: Tensor) -> Tensor:
        # from https://kvfrans.com/deriving-the-kl/
        return torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

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
            log_probs = self.decoder(
                batch_size=batch_size, hidden=hidden, target=target
            )
        else:
            log_probs = self.decoder(
                batch_size=batch_size, hidden=hidden, target=None
            )

        return log_probs

    def loss_function(
        self,
        log_probs: Tensor,
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
            mu (Tensor): Latent layer means.
            log_var (Tensor): Latent layer log variances.
            target (Tensor): Target sequences.
            mask (Tensor): Input mask.

        Returns:
            dict: Losses.
        """

        return self.weighted_seq_loss(
            log_probs=log_probs,
            mu=mu,
            log_var=log_var,
            target=target,
            mask=mask,
            **kwargs,
        )

    def predict(self, z: Tensor, device: int, **kwargs) -> Tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        z = z.to(device)
        prob_samples = exp(self.decode(z, **kwargs))
        return prob_samples

    def infer(self, x: Tensor, device: int, **kwargs) -> Tensor:
        """Given an encoder input, return reconstructed output and z samples.

        Args:
            x (tensor): [N, steps, acts].

        Returns:
            (tensor: [N, steps, acts], tensor: [N, latent_dims]).
        """
        log_probs_x, _, _, z = self.forward(x, **kwargs)
        prob_samples = exp(log_probs_x)
        prob_samples = prob_samples.to(device)
        z = z.to(device)
        return prob_samples, z

    def unweighted_seq_loss(
        self, log_probs, mu, log_var, target, mask, **kwargs
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
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
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
        self, log_probs, mu, log_var, target, mask, **kwargs
    ) -> dict:
        """Loss function for sequence encoding [N, L, 2]."""
        # unpack act probs and durations
        target_acts, target_durations = self.unpack_encoding(target)
        pred_acts, pred_durations = self.unpack_encoding(log_probs)

        # activity loss
        recon_act_nlll = self.base_NLLL(
            pred_acts.view(-1, self.encodings), target_acts.view(-1).long()
        )
        recon_act_nlll = (recon_act_nlll * mask.view(-1)).sum() / mask.sum()

        # duration loss
        recon_dur_mse = self.duration_weight * self.MSE(
            pred_durations, target_durations
        )
        recon_dur_mse = (recon_dur_mse * mask).sum() / mask.sum()

        # reconstruction loss
        recons_loss = recon_act_nlll + recon_dur_mse

        # kld loss
        norm_kld_weight = self.kld_weight

        unweighted_kld = self.kld(mu, log_var)
        kld_loss = norm_kld_weight * unweighted_kld

        return {
            "loss": recons_loss + kld_loss,
            "KLD": unweighted_kld.detach(),
            "betaKLD": kld_loss.detach(),
            "recon_loss": recons_loss.detach(),
            "recon_act_nlll_loss": recon_act_nlll.detach(),
            "recon_time_mse_loss": recon_dur_mse.detach(),
            "norm_kld_weight": torch.tensor([norm_kld_weight]).float(),
            "recon_act_ratio": recon_act_nlll / recon_dur_mse,
            "recon_kld_ratio": recons_loss / kld_loss,
        }

    def end_time_seq_loss(
        self, log_probs, mu, log_var, target, mask, **kwargs
    ) -> dict:
        """Loss function for sequence encoding [N, L, 2]."""
        # unpack act probs and durations
        target_acts, target_durations = self.unpack_encoding(target)
        pred_acts, pred_durations = self.unpack_encoding(log_probs)

        # activity loss
        recon_act_nlll = self.base_NLLL(
            pred_acts.view(-1, self.encodings), target_acts.view(-1).long()
        )
        recon_act_nlll = (recon_act_nlll * mask.view(-1)).sum() / mask.sum()

        # ends loss
        target_ends = torch.cumsum(target_durations, dim=-1)
        pred_ends = torch.cumsum(pred_durations, dim=-1)

        recon_end_mse = self.duration_weight * self.MSE(pred_ends, target_ends)
        recon_end_mse = (recon_end_mse * mask).sum() / mask.sum()

        # reconstruction loss
        recons_loss = recon_act_nlll + recon_end_mse

        # kld loss
        norm_kld_weight = self.kld_weight * self.latent_dim
        unweighted_kld = self.kld(mu, log_var)
        kld_loss = norm_kld_weight * unweighted_kld

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
        self, log_probs, mu, log_var, target, mask, **kwargs
    ) -> dict:
        """Loss function for sequence encoding [N, L, 2]."""
        # unpack act probs and durations
        target_acts, target_durations = self.unpack_encoding(target)
        pred_acts, pred_durations = self.unpack_encoding(log_probs)

        # activity loss
        recon_act_nlll = self.base_NLLL(
            pred_acts.view(-1, self.encodings), target_acts.view(-1).long()
        )
        recon_act_nlll = (recon_act_nlll * mask.view(-1)).sum() / mask.sum()

        # duration loss
        recon_dur_mse = self.duration_weight * self.MSE(
            pred_durations, target_durations
        )
        recon_dur_mse = (recon_dur_mse * mask).sum() / mask.sum()

        # ends loss
        target_ends = torch.cumsum(target_durations, dim=-1)
        pred_ends = torch.cumsum(pred_durations, dim=-1)

        recon_end_mse = self.duration_weight * self.MSE(pred_ends, target_ends)
        recon_end_mse = (recon_end_mse * mask).sum() / mask.sum()

        # combined time loss
        recon_time_mse = (0.5 * recon_dur_mse) + (0.5 * recon_end_mse)

        # reconstruction loss
        recons_loss = recon_act_nlll + recon_time_mse

        # kld loss
        norm_kld_weight = self.kld_weight * self.latent_dim
        unweighted_kld = self.kld(mu, log_var)
        kld_loss = norm_kld_weight * unweighted_kld

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
        self, log_probs, mu, log_var, target, mask, **kwargs
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
        unweighted_kld = self.kld(mu, log_var)
        kld_loss = norm_kld_weight * unweighted_kld

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
        self, log_probs, mu, log_var, target, mask, **kwargs
    ) -> dict:
        """Computes the loss function for discretized encoding [N, L, C]."""

        target_argmax = target.squeeze().argmax(dim=-1)
        return self.discretized_loss(
            log_probs, mu, log_var, target_argmax, mask, **kwargs
        )

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
