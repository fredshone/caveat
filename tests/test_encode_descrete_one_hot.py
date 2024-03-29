import numpy as np
import pandas as pd
import pytest
import torch
from torch import tensor

from caveat.encoders import discrete_one_hot


@pytest.mark.parametrize(
    "target,num_classes,expected",
    [
        (np.array([0, 1, 0]), 2, np.array([[1, 0], [0, 1], [1, 0]])),
        (np.array([0, 1, 0]), 3, np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])),
    ],
)
def test_one_hot(target, num_classes, expected):
    result = discrete_one_hot.one_hot(target, num_classes)
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
    result = discrete_one_hot.down_sample(target, step)
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
    result = discrete_one_hot.descretise_trace(
        acts, starts, ends, length, class_map
    )
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
        [[[[1, 0], [0, 1], [0, 1]]], [[[1, 0], [1, 0], [0, 1]]]], dtype=np.int8
    )
    class_map = {"a": 0, "b": 1}
    result = discrete_one_hot.descretise_population(
        traces, length, step, class_map
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
        [[[[1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0]]]],
        dtype=np.int8,
    )
    class_map = {"a": 0, "b": 1}
    result = discrete_one_hot.descretise_population(
        traces, length, step, class_map
    )
    expected = torch.from_numpy(expected)
    assert torch.equal(result, expected)


def test_encoded_weights():
    traces = pd.DataFrame(
        [
            [0, "a", 0, 2, 2],
            [0, "b", 2, 5, 3],
            [0, "a", 5, 6, 1],
            [1, "a", 0, 3, 3],
            [1, "b", 3, 5, 2],
            [1, "a", 5, 6, 1],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    length = 6
    step = 2
    expected_weights = torch.tensor([1 / 7, 1 / 5])
    encoder = discrete_one_hot.DiscreteOneHotEncoder(length, step)
    encoded = encoder.encode(traces)
    assert torch.equal(encoded.encoding_weights, expected_weights)


def test_encode_with_jitter():
    traces = pd.DataFrame(
        [
            [0, "a", 0, 2, 2],
            [0, "b", 2, 5, 3],
            [0, "a", 5, 6, 1],
            [1, "a", 0, 3, 3],
            [1, "b", 3, 5, 2],
            [1, "a", 5, 6, 1],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    length = 6
    step = 2
    encoder = discrete_one_hot.DiscreteOneHotEncoder(length, step, jitter=0.3)
    encoded = encoder.encode(traces)
    for _ in range(10):
        for i in range(len(encoded)):
            (left, left_mask), (right, right_mask) = encoded[i]
            assert left.shape == (1, 3, 2)
            assert left_mask.shape == (1, 3)
            assert right.shape == (1, 3, 2)
            assert right_mask.shape == (1, 3)


@pytest.mark.parametrize(
    "encoded,length,step_size,expected",
    [
        (
            tensor(
                [
                    [[[0.8, 0.1, 0.1], [0.7, 0.0, 0.0], [0.1, 0.5, 0.1]]],
                    [[[1, 0, 0], [0, 0, 1], [1, 0, 0]]],
                ]
            ),
            144,
            48,
            pd.DataFrame(
                [
                    [0, "a", 0, 96],
                    [0, "b", 96, 144],
                    [1, "a", 0, 48],
                    [1, "c", 48, 96],
                    [1, "a", 96, 144],
                ],
                columns=["pid", "act", "start", "end"],
            ),
        ),
        (
            tensor([[[[1, 0, 0], [1, 0, 0], [0, 1, 0]]]]),
            3,
            1,
            pd.DataFrame(
                [[0, "a", 0, 2], [0, "b", 2, 3]],
                columns=["pid", "act", "start", "end"],
            ),
        ),
    ],
)
def test_decode_descretised(encoded, length, step_size, expected):
    encoder = discrete_one_hot.DiscreteOneHotEncoder(
        duration=length, step_size=step_size
    )
    encoder.index_to_acts = {0: "a", 1: "b", 2: "c"}
    result = encoder.decode(encoded)
    pd.testing.assert_frame_equal(expected, result)
