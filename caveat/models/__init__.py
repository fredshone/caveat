from .base_VAE import Base
from .discrete.embed_conv import Conv
from .discrete.lstm_discrete import LSTM_Discrete
from .discrete.transformer_discrete import AttentionDiscrete
from .embed import CustomDurationEmbedding, CustomDurationModeDistanceEmbedding
from .sequence.cond_lstm import ConditionalLSTM
from .sequence.cond_gen_lstm import CVAE_LSTM
from .sequence.gen_lstm import VAE_LSTM
from .seq2seq.lstm import Seq2SeqLSTM
from .seq2score.lstm import Seq2ScoreLSTM

library = {
    "VAE_Conv_Discrete": Conv,
    "VAE_LSTM": VAE_LSTM,
    "CVAE_LSTM": CVAE_LSTM,
    "C_LSTM": ConditionalLSTM,
    "VAE_LSTM_Discrete": LSTM_Discrete,
    "Attention_Discrete": AttentionDiscrete,
    "Seq2Seq_LSTM": Seq2SeqLSTM,
    "Seq2Score_LSTM": Seq2ScoreLSTM,
}
