from .base import BaseEncoder
from .descrete2d import DescreteEncoder2D
from .descrete3d import DescreteEncoder3D

library = {
    "descrete": DescreteEncoder3D,
    "3d": DescreteEncoder3D,
    "2d": DescreteEncoder2D,
}
