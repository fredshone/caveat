from .discrete.embed_conv import Conv
from .discrete.lstm_discrete import LSTM_Discrete
from .discrete.transformer_discrete import AttentionDiscrete
from .sequence.conditional_lstm import ConditionalLSTM
from .sequence.lstm import LSTM

library = {
    "conv": Conv,
    "LSTM": LSTM,
    "conditional_LSTM": ConditionalLSTM,
    "LSTM_Discrete": LSTM_Discrete,
    "Attention_Discrete": AttentionDiscrete,
}
