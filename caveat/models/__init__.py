from .base_VAE import Base
from .discrete.embed_conv import Conv
from .discrete.lstm_discrete import LSTM_Discrete
from .discrete.transformer_discrete import AttentionDiscrete
from .embed import CustomDurationEmbedding
from .sequence.conditional_LSTM import ConditionalLSTM
from .sequence.conditional_VAE_LSTM import ConditionalVAE_LSTM
from .sequence.lstm import LSTM

library = {
    "VAE_Conv_Discrete": Conv,
    "VAE_LSTM": LSTM,
    "CVAE_LSTM": ConditionalVAE_LSTM,
    "C_LSTM": ConditionalLSTM,
    "VAE_LSTM_Discrete": LSTM_Discrete,
    "Attention_Discrete": AttentionDiscrete,
}
