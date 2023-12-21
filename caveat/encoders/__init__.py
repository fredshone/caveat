from re import S

from .base import BaseEncoder
from .descrete2d import DescreteEncoder2D
from .descrete3d import DescreteEncoder3D
from .seq import Sequence
from .seq2d import Sequence2D

library = {
    "descrete": DescreteEncoder3D,
    "3d": DescreteEncoder3D,
    "2d": DescreteEncoder2D,
    "sequence": Sequence,
    "sequence2d": Sequence2D,
}
