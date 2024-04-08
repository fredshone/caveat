from .ss.conv2d import ConvOneHot
from .discrete.embed_conv import Conv
from .discrete.lstm_discrete import LSTM_Discrete
from .discrete.transformer_discrete import AttentionDiscrete
from .sequence.conditional_lstm import ConditionalLSTM
from .sequence.gru import GRU
from .sequence.lstm import LSTM, LSTM_Unweighted
from .sequence.lstm_bi import LSTM_BI
from .sequence.lstm_deep import LSTM_Deep

library = {
    "conv": Conv,
    "convOH": ConvOneHot,
    "LSTM": LSTM,
    "conditional_LSTM": ConditionalLSTM,
    "GRU": GRU,
    "LSTM_BI": LSTM_BI,
    "LSTM_Deep": LSTM_Deep,
    "LSTM_Discrete": LSTM_Discrete,
    "LSTM_Unweighted": LSTM_Unweighted,
    "Attention_Discrete": AttentionDiscrete,
}
