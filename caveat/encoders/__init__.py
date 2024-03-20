from caveat.encoders.attributes import AttributeEncoder
from caveat.encoders.base import BaseEncoded, BaseEncoder
from caveat.encoders.discrete import DiscreteEncoder, DiscreteWithPadEncoder
from caveat.encoders.discrete_one_hot import DiscreteOneHotEncoder
from caveat.encoders.seq import UnweightedSequenceEncoder
from caveat.encoders.seq_weighted import SequenceEncoder

library = {
    "discrete_one_hot": DiscreteOneHotEncoder,
    "discrete": DiscreteEncoder,
    "discrete_padded": DiscreteWithPadEncoder,
    "unweighted_sequence": UnweightedSequenceEncoder,
    "sequence": SequenceEncoder,
}
