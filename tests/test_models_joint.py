import torch

from caveat.models.joint_vaes.jvae_sequence import JVAESeqLSTM


def test_cvae_lstm_nudger_forward():
    # schedules
    x = torch.randn(3, 10, 6)  # (batch, channels, steps, acts+1)
    weights = torch.ones((3, 10))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    x_encoded = torch.cat([acts_max, durations], dim=-1)

    # attributes
    conditionals = torch.Tensor([[1, 0], [1, 5], [2, 0]])

    model = JVAESeqLSTM(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=None,
        conditionals_size=10,
        **{
            "hidden_layers": 1,
            "hidden_size": 2,
            "latent_dim": 2,
            "dropout": 0.1,
            "attribute_embed_sizes": [2, 5],
        },
    )
    log_prob_x, log_prob_y, mu, log_var, z = model(
        x_encoded, conditionals=conditionals
    )
    assert log_prob_x.shape == (3, 10, 6)
    assert len(log_prob_y) == 2
    assert log_prob_y[0].shape == (3, 2)
    assert log_prob_y[1].shape == (3, 5)
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert z.shape == (3, 2)
    # losses = model.loss_function(
    #     log_probs=log_prob_y,
    #     probs=prob_y,
    #     mu=mu,
    #     log_var=log_var,
    #     target=x_encoded,
    #     mask=weights,
    # )
    # assert "loss" in losses
    # assert "recon_loss" in losses
