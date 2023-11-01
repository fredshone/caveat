from abc import ABC

from pandas import DataFrame
from torch import tensor


class BaseEncoder(ABC):
    def encode(self, input: DataFrame) -> tensor:
        raise NotImplementedError

    def decode(self, input: tensor) -> DataFrame:
        raise NotImplementedError
