from re import S

from .base import BaseEncodedPlans, BaseEncoder
from .descrete import DescreteEncoder
from .descrete_one_hot import DescreteEncoderOneHot
from .seq import SequenceEncoder

library = {
    "one_hot": DescreteEncoderOneHot,
    "descrete": DescreteEncoder,
    "sequence": SequenceEncoder,
}
