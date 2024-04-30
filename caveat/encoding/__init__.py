from caveat.encoding.attributes import AttributeEncoder
from caveat.encoding.base import (
    BaseDataset,
    BaseEncoder,
    PaddedDatatset,
    StaggeredDataset,
)
from caveat.encoding.discrete import DiscreteEncoder, DiscreteEncoderPadded
from caveat.encoding.sequence import SequenceEncoder, SequenceEncoderStaggered

library = {
    "discrete": DiscreteEncoder,
    "discrete_padded": DiscreteEncoderPadded,
    "sequence": SequenceEncoder,
    "sequence_staggered": SequenceEncoderStaggered,
}
