from abc import ABC
from typing import Optional

from pandas import DataFrame
from torch import Tensor
from torch.nn.functional import pad
from torch.utils.data import Dataset

from caveat.data import ScheduleAugment

from .attributes import AttributeEncoder


class BaseEncoder(ABC):
    conditionals_encoder = AttributeEncoder

    def __init__(self, schedules: DataFrame, **kwargs) -> None:
        raise NotImplementedError

    def encode(
        self, schedules: DataFrame, conditionals: Optional[Tensor]
    ) -> Dataset:
        raise NotImplementedError

    def decode(self, schedules: Tensor) -> DataFrame:
        raise NotImplementedError


class BaseDataset(Dataset):
    def __init__(
        self,
        schedules: Tensor,
        masks: Optional[Tensor],
        activity_encodings: int,
        activity_weights: Optional[Tensor],
        augment: Optional[ScheduleAugment],
        conditionals: Optional[Tensor],
    ):
        super(BaseDataset, self).__init__()
        self.schedules = schedules
        self.masks = masks
        self.activity_encodings = activity_encodings
        self.encoding_weights = activity_weights
        self.augment = augment
        self.conditionals = conditionals
        self.conditionals_shape = (
            conditionals.shape[-1] if conditionals is not None else None
        )

    def shape(self):
        return self.schedules[0].shape

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, idx):
        sample = self.schedules[idx]
        if self.augment:
            sample = self.augment(sample)

        if self.masks is not None:
            mask = self.masks[idx]
        else:
            mask = None

        if self.conditionals is not None:
            conditionals = self.conditionals[idx]
        else:
            conditionals = Tensor([])

        return (sample, mask), (sample, mask), conditionals


class PaddedDatatset(BaseDataset):

    def shape(self):
        return self.schedules[0].shape[0] + 1

    def __getitem__(self, idx):
        sample = self.schedules[idx]
        print(sample.shape)
        if self.augment:
            sample = self.augment(sample)

        if self.masks is not None:
            mask = self.masks[idx]
        else:
            mask = None

        if self.conditionals is not None:
            conditionals = self.conditionals[idx]
        else:
            conditionals = Tensor([])

        pad_left = pad(sample, (1, 0))
        pad_right = pad(sample, (0, 1))
        return (pad_left, mask), (pad_right, mask), conditionals


class StaggeredDataset(BaseDataset):

    def shape(self):
        return len(self.schedules[0]) - 1, 2

    def __getitem__(self, idx):
        sample = self.schedules[idx]
        if self.augment:
            sample = self.augment(sample)

        if self.masks is not None:
            mask = self.masks[idx]
        else:
            mask = None

        if self.conditionals is not None:
            conditionals = self.conditionals[idx]
        else:
            conditionals = Tensor([])

        return (
            (sample[:-1, :], mask[:-1]),
            (sample[1:, :], mask[1:]),
            conditionals,
        )
