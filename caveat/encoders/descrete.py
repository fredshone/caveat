from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from caveat.encoders import BaseEncoder


class DescreteEncoder(BaseEncoder):
    def __init__(self, duration: int = 1440, step_size: int = 10, **kwargs):
        self.duration = duration
        self.step_size = step_size
        self.steps = duration // step_size
        self.index_to_acts = None
        self.acts_to_index = None

    def encode(self, data: pd.DataFrame) -> Dataset:
        self.index_to_acts = {i: a for i, a in enumerate(data.act.unique())}
        self.acts_to_index = {a: i for i, a in self.index_to_acts.items()}
        return DescreteSequenceDataset(
            descretise_population(
                data,
                duration=self.duration,
                step_size=self.step_size,
                class_map=self.acts_to_index,
            )
        )

    def decode(self, encoded: Tensor) -> pd.DataFrame:
        """Decode decretised a sequences ([B, C, T, A]) into DataFrame of 'traces', eg:

        pid | act | start | end

        pid is taken as sample enumeration.

        Args:
            encoded (Tensor): _description_
            mapping (dict): _description_
            length (int): Length of plan in minutes.

        Returns:
            pd.DataFrame: _description_
        """
        encoded = torch.argmax(encoded, axis=-1)
        decoded = []

        for pid in range(len(encoded)):
            current_act = None
            act_start = 0

            for step, act_idx in enumerate(encoded[pid, 0]):
                if int(act_idx) != current_act and current_act is not None:
                    decoded.append(
                        [
                            pid,
                            self.index_to_acts[current_act],
                            int(act_start * self.step_size),
                            int(step * self.step_size),
                        ]
                    )
                    act_start = step
                current_act = int(act_idx)
            decoded.append(
                [
                    pid,
                    self.index_to_acts[current_act],
                    int(act_start * self.step_size),
                    self.duration,
                ]
            )

        return pd.DataFrame(decoded, columns=["pid", "act", "start", "end"])


class DescreteSequenceDataset(Dataset):
    def __init__(self, encoded: Tensor):
        """Torch Dataset for descretised sequence data.

        Args:
            data (Tensor): Population of sequences.
        """
        self.encoded = encoded
        self.size = len(encoded)

    def shape(self):
        return self.encoded[0].shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.encoded[idx]


def descretise_population(
    data: pd.DataFrame, duration: int, step_size: int, class_map: dict
) -> torch.tensor:
    """Convert given population of activity traces into vector [P, C, H, W].
    P is the population size.
    C (channel) is length 1.
    H is time steps.
    W is a one-hot encoding of activity type.

    Args:
        data (pd.DataFrame): _description_
        duration (int): _description_
        step_size (int): _description_
        class_map (dict): _description_

    Returns:
        torch.tensor: [P, C, H, W]
    """
    persons = data.pid.nunique()
    num_classes = len(class_map)
    steps = duration // step_size
    encoded = np.zeros((persons, steps, num_classes, 1), dtype=np.float32)

    for pid, (_, trace) in enumerate(data.groupby("pid")):
        trace_encoding = descretise_trace(
            acts=trace.act,
            starts=trace.start,
            ends=trace.end,
            length=duration,
            class_map=class_map,
        )
        trace_encoding = down_sample(trace_encoding, step_size)
        trace_encoding = one_hot(trace_encoding, num_classes)
        trace_encoding = trace_encoding.reshape(steps, num_classes, 1)
        encoded[pid] = trace_encoding  # [B, H, W, C]
    encoded = encoded.transpose(0, 3, 1, 2)  # [B, C, H, W]
    return torch.from_numpy(encoded)


def descretise_trace(
    acts: Iterable[str],
    starts: Iterable[int],
    ends: Iterable[int],
    length: int,
    class_map: dict,
) -> np.array:
    """Create categorical encoding from ranges with step of 1.

    Args:
        acts (Iterable[str]): _description_
        starts (Iterable[int]): _description_
        ends (Iterable[int]): _description_
        length (int): _description_
        class_map (dict): _description_

    Returns:
        np.array: _description_
    """
    encoding = np.zeros((length), dtype=np.int8)
    for act, start, end in zip(acts, starts, ends):
        encoding[start:end] = class_map[act]
    return encoding


def down_sample(array: np.array, step: int) -> np.array:
    """Down-sample by steppiong through given array.
    todo:
    Methodology will down sample based on first classification.
    If we are down sampling a lot (for example from minutes to hours),
    we would be better of, samplig based on majority class.

    Args:
        array (np.array): _description_
        step (int): _description_

    Returns:
        np.array: _description_
    """
    return array[::step]


def one_hot(target: np.array, num_classes: int) -> np.array:
    """One hot encoding of given categorical array.

    Args:
        target (np.array): _description_
        num_classes (int): _description_

    Returns:
        np.array: _description_
    """
    return np.eye(num_classes)[target]
