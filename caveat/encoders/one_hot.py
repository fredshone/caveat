from typing import Iterable

import numpy as np
import pandas as pd
import torch


def descretise_population(
    data: pd.DataFrame, duration: int, step_size: int, class_map: dict
) -> torch.Tensor:
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
) -> np.ndarray:
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


def down_sample(array: np.ndarray, step: int) -> np.ndarray:
    """Down-sample by stepping through given array.
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


def one_hot(target: np.ndarray, num_classes: int) -> np.ndarray:
    """One hot encoding of given categorical array.

    Args:
        target (np.array): _description_
        num_classes (int): _description_

    Returns:
        np.array: _description_
    """
    return np.eye(num_classes)[target]
