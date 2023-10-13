import numpy as np
import pandas as pd
import pytest
import torch

from caveat.data import loader


@pytest.mark.parametrize(
    "target,num_classes,expected",
    [
        (np.array([0, 1, 0]), 2, np.array([[1, 0], [0, 1], [1, 0]])),
        (np.array([0, 1, 0]), 3, np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])),
    ],
)
def test_one_hot(target, num_classes, expected):
    result = loader.one_hot(target, num_classes)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "target,step,expected",
    [
        (np.array([0, 0, 1, 1, 1, 0]), 1, np.array([0, 0, 1, 1, 1, 0])),
        (np.array([0, 0, 1, 1, 1, 0]), 2, np.array([0, 1, 1])),
        (np.array([0, 0, 1, 1, 1, 0]), 3, np.array([0, 1])),
        (np.array([0, 0, 1, 1, 1, 0]), 4, np.array([0, 1])),
        (np.array([0, 0, 1, 1, 1, 0]), 5, np.array([0, 0])),
        (np.array([0, 0, 1, 1, 1, 0]), 6, np.array([0])),
    ],
)
def test_down_sample(target, step, expected):
    result = loader.down_sample(target, step)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "acts,starts,ends,length,expected",
    [
        (
            ["a", "b", "a"],
            [0, 2, 5],
            [2, 5, 6],
            6,
            np.array([0, 0, 1, 1, 1, 0]),
        ),
        (["a", "b", "a"], [0, 2, 5], [2, 5, 6], 5, np.array([0, 0, 1, 1, 1])),
        (["b", "a"], [2, 5], [5, 6], 6, np.array([0, 0, 1, 1, 1, 0])),
    ],
)
def test_descretise_trace(acts, starts, ends, length, expected):
    class_map = {"a": 0, "b": 1}
    result = loader.descretise_trace(acts, starts, ends, length, class_map)
    np.testing.assert_array_equal(result, expected)


def test_descretise_population():
    traces = pd.DataFrame(
        [
            [0, "a", 0, 2],
            [0, "b", 2, 5],
            [0, "a", 5, 6],
            [1, "a", 0, 3],
            [1, "b", 3, 5],
            [1, "a", 5, 6],
        ],
        columns=["pid", "act", "start", "end"],
    )
    length = 6
    step = 2
    expected = np.array(
        [[[[1, 0, 0], [0, 1, 1]]], [[[1, 1, 0], [0, 0, 1]]]], dtype=np.int8
    )
    samples = traces.pid.nunique()
    class_map = {"a": 0, "b": 1}
    result = loader.descretise_population(
        traces, samples, length, step, class_map
    )
    expected = torch.from_numpy(expected)
    assert torch.equal(result, expected)


def test_large_downsample_descretise_population():
    traces = pd.DataFrame(
        [[0, "a", 0, 360], [0, "b", 360, 840], [0, "a", 840, 1440]],
        columns=["pid", "act", "start", "end"],
    )
    length = 1440
    step = 180
    expected = np.array(
        [[[[1, 1, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0, 0]]]], dtype=np.int8
    )
    samples = traces.pid.nunique()
    class_map = {"a": 0, "b": 1}
    result = loader.descretise_population(
        traces, samples, length, step, class_map
    )
    expected = torch.from_numpy(expected)
    assert torch.equal(result, expected)
