import torch

from caveat.models.discrete.embed_conv import Conv
from caveat.models.discrete.lstm_discrete import LSTM_Discrete
from caveat.models.discrete.transformer_discrete import AttentionDiscrete
from caveat.models.seq2seq.lstm import Seq2SeqLSTM
from caveat.models.sequence.cond_gen_lstm import CVAE_LSTM
from caveat.models.sequence.gen_lstm import VAE_LSTM


def test_conv_embed_cov_forward():
    x = torch.randn(3, 1, 144, 5)  # (batch, channels, steps, acts)
    x_max = x.argmax(dim=-1).squeeze()
    model = Conv(
        in_shape=x_max[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        **{"hidden_layers": [1], "latent_dim": 2, "dropout": 0.1},
    )
    log_prob_y, prob_y, mu, log_var = model(x_max)
    assert log_prob_y.shape == torch.Size([3, 144, 5])
    assert prob_y.shape == torch.Size([3, 144, 5])
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    losses = model.loss_function(
        log_prob_y, prob_y, mu, log_var, x_max, mask=None
    )
    assert "loss" in losses
    assert "recon_loss" in losses


def test_discrete_lstm_forward():
    x = torch.randn(3, 144, 5)  # (batch, channels, steps, acts)
    x_max = x.argmax(dim=-1).squeeze()
    model = LSTM_Discrete(
        in_shape=x_max[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        **{
            "hidden_layers": 1,
            "hidden_size": 2,
            "latent_dim": 2,
            "dropout": 0.1,
        },
    )
    log_prob_y, prob_y, mu, log_var = model(x_max)
    assert log_prob_y.shape == x.shape
    assert prob_y.shape == x.shape
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    losses = model.loss_function(
        log_prob_y, prob_y, mu, log_var, x_max, mask=None
    )
    assert "loss" in losses
    assert "recon_loss" in losses


def test_discrete_transformer_forward():
    x = torch.randn(3, 144, 5)  # (batch, channels, steps, acts)
    x_max = x.argmax(dim=-1).squeeze()
    model = AttentionDiscrete(
        in_shape=x_max[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        **{
            "hidden_layers": 1,
            "hidden_size": 2,
            "heads": 1,
            "latent_dim": 2,
            "dropout": 0.1,
        },
    )
    log_prob_y, prob_y, mu, log_var = model(x_max)
    assert log_prob_y.shape == x.shape
    assert prob_y.shape == x.shape
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    losses = model.loss_function(
        log_prob_y, prob_y, mu, log_var, x_max, mask=None
    )
    assert "loss" in losses
    assert "recon_loss" in losses


def test_lstm_forward():
    x = torch.randn(3, 10, 6)  # (batch, channels, steps, acts+1)
    weights = torch.ones((3, 10))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    model = VAE_LSTM(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        **{
            "hidden_layers": 1,
            "hidden_size": 2,
            "latent_dim": 2,
            "dropout": 0.1,
        },
    )
    log_prob_y, prob_y, mu, log_var = model(x_encoded)
    assert log_prob_y.shape == x.shape
    assert prob_y.shape == x.shape
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    losses = model.loss_function(
        log_prob_y, prob_y, mu, log_var, x_encoded, mask=weights
    )
    assert "loss" in losses
    assert "recon_loss" in losses


def test_conditional_vae_lstm_forward():
    x = torch.randn(3, 10, 6)  # (batch, channels, steps, acts+1)
    weights = torch.ones((3, 10))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    conditionals = torch.randn(3, 10)  # (batch, channels)
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    model = CVAE_LSTM(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        conditionals_size=10,
        **{
            "hidden_layers": 1,
            "hidden_size": 2,
            "latent_dim": 2,
            "dropout": 0.1,
        },
    )
    log_prob_y, prob_y, mu, log_var = model(
        x_encoded, conditionals=conditionals
    )
    assert log_prob_y.shape == x.shape
    assert prob_y.shape == x.shape
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    losses = model.loss_function(
        log_prob_y, prob_y, mu, log_var, x_encoded, mask=weights
    )
    assert "loss" in losses
    assert "recon_loss" in losses


def test_seq2seq_forward():
    N = 10
    L = 16
    A = 5
    M = 5
    C = A + M + 2
    x = torch.randn(N, L, C)
    weights = torch.ones((N, L))
    acts, durations, modes, distances = x.split([A, 1, M, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    modes_max = modes.argmax(dim=-1).unsqueeze(-1)
    conditionals = torch.randn(N, 4)
    x_encoded = torch.cat([acts_max, durations, modes_max, distances], dim=-1)
    model = Seq2SeqLSTM(
        in_shape=x_encoded[0].shape,
        encodings=(A, M),
        encoding_weights=None,
        conditionals_size=4,
        **{
            "hidden_layers": 1,
            "hidden_size": 12,
            "latent_dim": 2,
            "dropout": 0.1,
        },
    )
    (
        log_act_probs,
        act_probs,
        durations,
        log_mode_probs,
        mode_probs,
        distances,
    ) = model(x_encoded, conditionals=conditionals)
    assert log_act_probs.shape == (N, L, A)
    assert act_probs.shape == (N, L, A)
    assert durations.shape == (N, L, 1)
    assert log_mode_probs.shape == (N, L, M)
    assert mode_probs.shape == (N, L, M)
    assert distances.shape == (N, L, 1)
    losses = model.loss_function(
        log_act_probs,
        act_probs,
        durations,
        log_mode_probs,
        mode_probs,
        distances,
        x_encoded,
        mask=weights,
    )
    assert "loss" in losses
    assert "recon_loss" in losses
