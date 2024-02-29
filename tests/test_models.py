import torch

from caveat.models.discrete.conv2d import ConvOneHot
from caveat.models.discrete.embed_conv import Conv
from caveat.models.discrete.lstm_discrete import LSTM_Discrete
from caveat.models.discrete.transformer_discrete import AttentionDiscrete
from caveat.models.sequence.gru import GRU
from caveat.models.sequence.lstm import LSTM


def test_conv_one_hot_forward():
    x = torch.randn(3, 1, 144, 5)  # (batch, channels, steps, acts)
    model = ConvOneHot(
        in_shape=x[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        **{"hidden_layers": [1], "latent_dim": 2, "dropout": 0.1},
    )
    log_prob_y, prob_y, mu, log_var = model(x)
    assert log_prob_y.shape == x.shape
    assert prob_y.shape == x.shape
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    losses = model.loss_function(log_prob_y, prob_y, mu, log_var, x, mask=None)
    assert "loss" in losses
    assert "recon_loss" in losses


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


def test_gru_forward():
    x = torch.randn(3, 10, 6)  # (batch, steps, acts+1)
    weights = torch.ones((3, 10))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    model = GRU(
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


def test_lstm_forward():
    x = torch.randn(3, 10, 6)  # (batch, channels, steps, acts+1)
    weights = torch.ones((3, 10))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    model = LSTM(
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
