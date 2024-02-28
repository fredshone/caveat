from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from caveat.encoders import BaseEncoded, BaseEncoder


class DiscreteEncoder(BaseEncoder):
    def __init__(self, duration: int = 1440, step_size: int = 10, **kwargs):
        self.duration = duration
        self.step_size = step_size
        self.steps = duration // step_size

    def encode(self, data: pd.DataFrame) -> BaseEncoded:
        self.index_to_acts = {i: a for i, a in enumerate(data.act.unique())}
        self.acts_to_index = {a: i for i, a in self.index_to_acts.items()}
        return DiscreteEncoded(
            data,
            duration=self.duration,
            step_size=self.step_size,
            class_map=self.acts_to_index,
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
        encoded = torch.argmax(encoded, dim=-1)
        decoded = []

        for pid in range(len(encoded)):
            current_act = None
            act_start = 0

            for step, act_idx in enumerate(encoded[pid]):
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


class DiscreteEncoded(BaseEncoded):
    def __init__(
        self, data: pd.DataFrame, duration: int, step_size: int, class_map: dict
    ):
        """Torch Dataset for descretised sequence data.

        Args:
            data (Tensor): Population of sequences.
        """
        data = data.copy()
        data.act = data.act.map(class_map)
        self.encodings = data.act.nunique()
        # calc weightings
        weights = data.groupby("act", observed=True).duration.sum().to_dict()
        weights = np.array([weights[k] for k in range(len(weights))])
        self.encoding_weights = torch.from_numpy(1 / weights).float()
        self.encoded = discretise_population(
            data, duration=duration, step_size=step_size
        )
        self.mask = torch.ones((1, self.encoded.shape[-1]))
        self.size = len(self.encoded)

    def shape(self):
        return self.encoded[0].shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = self.encoded[idx]
        return (sample, self.mask), (sample, self.mask)


class DiscreteWithPadEncoder(BaseEncoder):
    def __init__(self, duration: int = 1440, step_size: int = 10, **kwargs):
        self.duration = duration
        self.step_size = step_size
        self.steps = duration // step_size

    def encode(self, data: pd.DataFrame) -> BaseEncoded:
        self.index_to_acts = {i + 1: a for i, a in enumerate(data.act.unique())}
        self.index_to_acts[0] = "<PAD>"
        self.acts_to_index = {a: i for i, a in self.index_to_acts.items()}
        return DiscretePadEncoded(
            data,
            duration=self.duration,
            step_size=self.step_size,
            class_map=self.acts_to_index,
        )

    def decode(self, encoded: Tensor) -> pd.DataFrame:
        """Decode disretised a sequences ([B, C, T, A]) into DataFrame of 'traces', eg:

        pid | act | start | end

        pid is taken as sample enumeration.

        Args:
            encoded (Tensor): _description_
            mapping (dict): _description_
            length (int): Length of plan in minutes.

        Returns:
            pd.DataFrame: _description_
        """
        encoded = torch.argmax(encoded, dim=-1)
        decoded = []

        for pid in range(len(encoded)):
            current_act = None
            act_start = 0

            for step, act_idx in enumerate(encoded[pid]):
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


class DiscretePadEncoded(BaseEncoded):
    def __init__(
        self, data: pd.DataFrame, duration: int, step_size: int, class_map: dict
    ):
        """Torch Dataset for discretised sequence data with padding.

        Args:
            data (Tensor): Population of sequences.
        """
        super().__init__()
        data = data.copy()
        data.act = data.act.map(class_map)
        self.encodings = data.act.nunique() + 1  # <PAD>
        # calc weightings
        weights = data.groupby("act", observed=True).duration.sum().to_dict()
        weights[0] = data.pid.nunique() * 60  # pad weight is equal to 1 hour
        weights = np.array([weights[k] for k in range(len(weights))])
        self.encoding_weights = torch.from_numpy(1 / (weights)).float()
        encoded = discretise_population(
            data, duration=duration, step_size=step_size
        )
        self.size = len(encoded)
        self.encoded_pad_left = torch.concatenate(
            (torch.zeros((self.size, 1)), encoded), dim=-1
        )
        self.encoded_pad_right = torch.concatenate(
            (encoded, torch.zeros((self.size, 1))), dim=-1
        )
        self.mask = torch.ones((1, self.encoded_pad_left.shape[-1]))

    def shape(self):
        return self.encoded_pad_left[0].shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (
            (self.encoded_pad_left[idx], self.mask),
            (self.encoded_pad_right[idx], self.mask),
        )


def discretise_population(
    data: pd.DataFrame, duration: int, step_size: int
) -> torch.Tensor:
    """Convert given population of activity traces into vector [N, L] of classes.
    N is the population size.
    l is time steps.

    Args:
        data (pd.DataFrame): _description_
        duration (int): _description_
        step_size (int): _description_

    Returns:
        torch.tensor: [N, L]
    """
    persons = data.pid.nunique()
    steps = duration // step_size
    encoded = np.zeros((persons, steps), dtype=np.int8)

    for pid, (_, trace) in enumerate(data.groupby("pid")):
        trace_encoding = discretise_trace(
            acts=trace.act, starts=trace.start, ends=trace.end, length=duration
        )
        trace_encoding = down_sample(trace_encoding, step_size)
        encoded[pid] = trace_encoding  # [N, L]
    return torch.from_numpy(encoded)


def discretise_trace(
    acts: Iterable[str], starts: Iterable[int], ends: Iterable[int], length: int
) -> np.ndarray:
    """Create categorical encoding from ranges with step of 1.

    Args:
        acts (Iterable[str]): _description_
        starts (Iterable[int]): _description_
        ends (Iterable[int]): _description_
        length (int): _description_

    Returns:
        np.array: _description_
    """
    encoding = np.zeros((length), dtype=np.int8)
    for act, start, end in zip(acts, starts, ends):
        encoding[start:end] = act
    return encoding


def down_sample(array: np.ndarray, step: int) -> np.ndarray:
    """Down-sample by steppiong through given array.
    todo:
    Methodology will down sample based on first classification.
    If we are down sampling a lot (for example from minutes to hours),
    we would be better of, sampling based on majority class.

    Args:
        array (np.array): _description_
        step (int): _description_

    Returns:
        np.array: _description_
    """
    return array[::step]
