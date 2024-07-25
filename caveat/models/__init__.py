from .base import Base
from .discrete.auto_discrete_lstm import AutoDiscLSTM
from .discrete.cond_discrete_conv import CondDiscConv
from .discrete.cond_discrete_lstm import CondDiscLSTM
from .discrete.vae_discrete_conv import VAEDiscConv
from .discrete.vae_discrete_lstm import VAEDiscLSTM
from .discrete.vae_discrete_transformer import VAEDiscTrans
from .embed import CustomDurationEmbedding, CustomDurationModeDistanceEmbedding
from .seq2score.lstm import Seq2ScoreLSTM
from .seq2seq.lstm import Seq2SeqLSTM
from .sequence.auto_sequence_lstm import AutoSeqLSTM
from .sequence.cond_sequence_lstm import CondSeqLSTM
from .sequence.cvae_sequence_lstm import CVAESeqLSTM
from .sequence.cvae_sequence_lstm_2 import CVAESeqLSTM2
from .sequence.vae_sequence_lstm import VAESeqLSTM

library = {
    "Koushik": CondDiscLSTM,
    "CondDiscLSTM": CondDiscLSTM,
    "CondDiscConv": CondDiscConv,
    "CondSeqLSTM": CondSeqLSTM,
    "AutoDiscLSTM": AutoDiscLSTM,
    "AutoSeqLSTM": AutoSeqLSTM,
    "VAEDiscConv": VAEDiscConv,
    "VAESeqLSTM": VAESeqLSTM,
    "VAEDiscLSTM": VAEDiscLSTM,
    "VAEDiscTrans": VAEDiscTrans,
    "CVAESeqLSTM": CVAESeqLSTM,
    "CVAESeqLSTM2": CVAESeqLSTM2,
    "Seq2SeqLSTM": Seq2SeqLSTM,
    "Seq2ScoreLSTM": Seq2ScoreLSTM,
}
