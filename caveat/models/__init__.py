from .seq.base import SEQVAE
from .seq.lstm import LSTM
from .seq.lstm2d import LSTM2d
from .vae import VAE2D

library = {
    "VAE": VAE2D,
    "VAE2D": VAE2D,
    "SEQVAE": SEQVAE,
    "LSTM": LSTM,
    "LSTM2D": LSTM2d,
}
