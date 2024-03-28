from caveat.encoders.attributes import AttributeEncoder
from caveat.encoders.base import BaseDataset, BaseEncoder, PaddedDatatset
from caveat.encoders.discrete import DiscreteEncoder, DiscreteEncoderPadded
from caveat.encoders.sequence import SequenceEncoder

library = {
    "discrete": DiscreteEncoder,
    "discrete_padded": DiscreteEncoderPadded,
    "sequence": SequenceEncoder,
}
