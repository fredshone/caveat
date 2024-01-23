from .conv.conv2d import Conv2d
from .conv.embed_conv import EmbedConv
from .seq.gru import GRU
from .seq.lstm import LSTM
from .seq.lstm_bi import LSTM_BI
from .seq.lstm_deep import LSTM_Deep
from .seq.seq2seq import SEQ2SEQ

library = {
    "EmbedConv": EmbedConv,
    "Conv2d": Conv2d,
    "LSTM": LSTM,
    "GRU": GRU,
    "LSTM_BI": LSTM_BI,
    "LSTM_Deep": LSTM_Deep,
    "SEQ2SEQ": SEQ2SEQ,
}
