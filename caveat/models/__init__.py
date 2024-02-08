from .conv.conv2d import ConvOneHot
from .conv.embed_conv import Conv
from .seq.gru import GRU
from .seq.lstm import LSTM, LSTM_Unweighted
from .seq.lstm_bi import LSTM_BI
from .seq.lstm_deep import LSTM_Deep

library = {
    "conv": Conv,
    "convOH": ConvOneHot,
    "LSTM": LSTM,
    "GRU": GRU,
    "LSTM_BI": LSTM_BI,
    "LSTM_Deep": LSTM_Deep,
    "LSTM_Unweighted": LSTM_Unweighted,
}
