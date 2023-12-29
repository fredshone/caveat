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
        self.kld_weight = config.get("kld_weight", 0.1)
        self.duration_weight = config.get("duration_weight", 1)
        self.use_mask = config.get("use_mask", True)  # defaults to True
        self.use_weighted_loss = config.get(
            "use_weighted_loss", True
        )  # defaults to True

        self.NLLL = nn.NLLLoss(weight=encoding_weights)
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

    def forward(self, x: Tensor, target=None, **kwargs) -> List[Tensor]:
        """Forward pass, also return latent parameterization.

        Args:
            x (tensor): Input sequences [N, L, Cin].

        Returns:
            list[tensor]: [Log probs, Probs [N, L, Cout], Input [N, L, Cin], mu [N, latent], var [N, latent]].
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        log_prob_y, prob_y = self.decode(z, target=target)
        return [log_prob_y, prob_y, x, mu, log_var]

    def loss_function(
        self, log_probs, _, input, mu, log_var, mask, **kwargs
    ) -> dict:
        r"""Computes the VAE loss function.

        Splits the input into activity and duration, and the recons into activity and duration.

        Returns:
            dict: Losses.
        """

        # unpack act probs and durations
        target_acts, target_durations = self.unpack_encoding(input)
        pred_acts, pred_durations = self.unpack_encoding(log_probs)

        if self.use_mask:  # default is to use masking
            flat_mask = mask.view(-1).bool()
        else:
            flat_mask = torch.ones_like(target_acts).view(-1).bool()

        # activity encodng
        recon_act_nlll = self.NLLL(
            pred_acts.view(-1, self.encodings)[flat_mask],
            target_acts.view(-1).long()[flat_mask],
        )

        # duration encodng
        recon_dur_mse = self.duration_weight * self.MSE(
            pred_durations.view(-1)[flat_mask],
            target_durations.view(-1)[flat_mask],
        )

        # combined
        recons_loss = recon_act_nlll + recon_dur_mse

        recon_argmax = torch.argmax(pred_acts, dim=-1)
        recon_act_ham = self.hamming(recon_argmax, target_acts.squeeze().long())

        output_size = log_probs.shape[-1] * log_probs.shape[-2]
        norm_kld_weight = self.kld_weight * self.latent_dim / output_size

        kld_loss = norm_kld_weight * torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0,
        )

        loss = recons_loss + kld_loss
        return {
            "loss": loss,
            "recon_loss": recons_loss.detach(),
            "recon_act_nlll_loss": recon_act_nlll.detach(),
            "recon_dur_mse_loss": recon_dur_mse.detach(),
            "recon_act_ham_loss": recon_act_ham.detach(),
            "KLD": kld_loss.detach(),
            "norm_kld_weight": torch.tensor([norm_kld_weight]),
        }

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

    def predict_step(self, z: Tensor, current_device: int, **kwargs) -> Tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        # z = z.to(current_device)
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
        # prob_samples = prob_samples.to(current_device)
        return prob_samples
