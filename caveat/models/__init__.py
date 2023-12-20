from .seq.base import SEQVAE
from .seq.lstm import SEQVAESEQ
from .vae import VAE2D

library = {
    "VAE": VAE2D,
    "VAE2D": VAE2D,
    "SEQVAE": SEQVAE,
    "SEQVAESEQ": SEQVAESEQ,
}
