from caveat.encoding.attributes import AttributeEncoder
from caveat.encoding.base import (
    BaseDataset,
    BaseEncoder,
    PaddedDatatset,
    LHS2RHSDataset,
    StaggeredDataset,
)
from caveat.encoding.discrete import DiscreteEncoder, DiscreteEncoderPadded
from caveat.encoding.sequence import SequenceEncoder, SequenceEncoderStaggered
from caveat.encoding.seq2seq import Seq2SeqEncoder
from caveat.encoding.seq2score import Seq2ScoreEncoder

library = {
    "discrete": DiscreteEncoder,
    "discrete_padded": DiscreteEncoderPadded,
    "sequence": SequenceEncoder,
    "sequence_staggered": SequenceEncoderStaggered,
    "seq2seq": Seq2SeqEncoder,
    "seq2score": Seq2ScoreEncoder,
}
