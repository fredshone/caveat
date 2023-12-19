from .seq.lstm import SEQVAE
from .seq.lstm_2 import SEQVAESEQ
from .vae import VAE2D

library = {
    "VAE": VAE2D,
    "VAE2D": VAE2D,
    "SEQVAE": SEQVAE,
    "SEQVAESEQ": SEQVAESEQ,
}
