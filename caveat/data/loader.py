from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DescreteSequenceDataset(Dataset):
    def __init__(self, path: str, length: int, step: int):
        df = pd.read_csv(path)
        self.size = df.pid.nunique()
        self.index_to_acts = {i: a for i, a in enumerate(df.act.unique())}
        self.acts_to_index = {a: i for i, a in self.index_to_acts.items()}
        self.encoded = descretise_population(
            df,
            samples=self.size,
            length=length,
            step=step,
            class_map=self.acts_to_index,
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.encoded[idx]


class VAEDataset(LightningDataModule):
    def __init__(
        self,
        path: str,
        length: int = 1440,
        step: int = 10,
        val_split: float = 0.1,
        seed: int = 1234,
        train_batch_size: int = 128,
        val_batch_size: int = 128,
        test_batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.path = path
        self.length = length
        self.step = step
        self.steps = length // step
        self.val_split = val_split
        self.generator = torch.manual_seed(seed)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mapping = None

    def setup(self, stage: Optional[str] = None) -> None:
        data = DescreteSequenceDataset(self.path, self.length, self.step)
        self.mapping = data.index_to_acts
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            data, [1 - self.val_split, self.val_split], generator=self.generator
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


def descretise_population(
    data: pd.DataFrame, samples: int, length: int, step: int, class_map: dict
) -> torch.tensor:
    num_classes = len(class_map)  # bit dodgy
    steps = length // step
    encoded = np.zeros((samples, steps, num_classes, 1), dtype=np.float32)
    # todo: we keep the last dimension so we look like an image, remove?

    for pid, trace in data.groupby("pid"):
        trace_encoding = descretise_trace(
            acts=trace.act,
            starts=trace.start,
            ends=trace.end,
            length=length,
            class_map=class_map,
        )
        trace_encoding = down_sample(trace_encoding, step)
        assert len(trace_encoding) == steps
        trace_encoding = one_hot(trace_encoding, num_classes)
        trace_encoding = trace_encoding.reshape(steps, num_classes, 1)  # todo
        encoded[pid] = trace_encoding
    encoded = encoded.transpose(0, 3, 2, 1)  # [B, C, H, W]
    return torch.from_numpy(encoded)


def descretise_trace(
    acts: Iterable[str],
    starts: Iterable[int],
    ends: Iterable[int],
    length: int,
    class_map: dict,
) -> np.array:
    """
    Create categorical encoding from ranges with step of 1.
    """
    encoding = np.zeros((length), dtype=np.int8)
    for act, start, end in zip(acts, starts, ends):
        encoding[start:end] = class_map[act]
    return encoding


def down_sample(array: np.array, step: int) -> np.array:
    """
    todo:
    Methodology will down sample based on first classification.
    If we are down sampling a lot (for example from minutes to hours),
    we would be better of, samplig based on majority class.
    """
    return array[::step]


def one_hot(target: np.array, num_classes: int) -> np.array:
    return np.eye(num_classes)[target]
