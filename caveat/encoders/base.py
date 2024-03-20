from abc import ABC
from typing import Optional

from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset


class BaseEncoded(Dataset):
    def __init__(self):
        """Base encoded sequence Dataset."""
        super(BaseEncoded, self).__init__()
        self.encodings: int
        self.encoding_weights: Tensor
        self.masks: Tensor

    def shape(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class BaseEncoder(ABC):
    def __init__(self) -> None:
        super(BaseEncoder, self).__init__()
        self.encodings = None

    def encode(
        self, schedules: DataFrame, conditionals: Optional[DataFrame]
    ) -> BaseEncoded:
        raise NotImplementedError

    def decode(self, schedules: Tensor) -> DataFrame:
        raise NotImplementedError
