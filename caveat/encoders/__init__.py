from caveat.encoders.attributes import AttributeEncoder
from caveat.encoders.base import (
    BaseDataset,
    BaseEncoder,
    PaddedDatatset,
    StaggeredDataset,
)
from caveat.encoders.discrete import DiscreteEncoder, DiscreteEncoderPadded
from caveat.encoders.sequence import SequenceEncoder, SequenceEncoderStaggered

library = {
    "discrete": DiscreteEncoder,
    "discrete_padded": DiscreteEncoderPadded,
    "sequence": SequenceEncoder,
    "sequence_staggered": SequenceEncoderStaggered,
}
