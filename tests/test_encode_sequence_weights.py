import numpy as np
import pandas as pd
import pytest
import torch
from torch import tensor

from caveat.encoders import seq_weighted as seq


@pytest.mark.parametrize(
    "acts,durations,act_weights,expected,expects_weights",
    [
        (
            [2, 3, 2],
            [0.3, 0.2, 0.5],
            {0: 0.1, 1: 0.1, 2: 0.3, 3: 0.5},
            np.array(
                [[0, 0.0], [2, 0.3], [3, 0.2], [2, 0.5], [1, 0.0], [1, 0.0]],
                dtype=np.float32,
            ),
            np.array([0.1, 0.3, 0.5, 0.3, 0.1, 0.0], dtype=np.float32),
        )
    ],
)
def test_encode_sequence(
    acts, durations, act_weights, expected, expects_weights
):
    result, weights = seq.encode_sequence(
        acts,
        durations,
        max_length=6,
        encoding_width=2,
        act_weights=act_weights,
        sos=0,
        eos=1,
    )
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(weights, expects_weights)


def test_encode_population():
    traces = pd.DataFrame(
        [
            [0, "a", 0, 4, 4],
            [0, "b", 4, 10, 6],
            [1, "a", 0, 6, 6],
            [1, "b", 6, 10, 4],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    map = {"<SOS>": 0, "<EOS>": 1, "a": 2, "b": 3}
    encoded = seq.SequenceDurationsWeighted(
        traces, max_length=6, acts_to_index=map, norm_duration=10
    )
    expected = np.array(
        [
            [[0, 0.0], [2, 0.4], [3, 0.6], [1, 0.0], [1, 0.0], [1, 0.0]],
            [[0, 0.0], [2, 0.6], [3, 0.4], [1, 0.0], [1, 0.0], [1, 0.0]],
        ],
        dtype=np.float32,
    )
    expected_weights = np.array(
        [[0.5, 1, 1, 0.5, 0.0, 0.0], [0.5, 1, 1, 0.5, 0.0, 0.0]],
        dtype=np.float32,
    )
    expected = torch.from_numpy(expected)
    expected_weights = torch.from_numpy(expected_weights)

    assert torch.equal(encoded.encoded, expected)
    assert torch.equal(encoded.encoding_weights, expected_weights)


def test_encoded():
    traces = pd.DataFrame(
        [
            [0, 0, 0, 4, 4],
            [0, 1, 4, 8, 4],
            [0, 0, 8, 10, 2],
            [1, 0, 0, 3, 3],
            [1, 1, 3, 7, 4],
            [1, 0, 7, 10, 3],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    length = 6
    duration = 10
    expected = torch.tensor(
        [[0, 0.0], [2, 0.4], [3, 0.4], [2, 0.2], [1, 0.0], [1, 0.0]]
    )
    expected_weights = torch.tensor(
        [1 / 2, 1 / 1.2, 1 / 0.8, 1 / 1.2, 1 / 2, 0]
    )
    encoder = seq.SequenceEncoder(max_length=length, duration=duration)
    encoded = encoder.encode(traces)
    (left, left_mask), (right, right_mask) = encoded[0]
    assert torch.equal(left, expected)
    assert torch.equal(left_mask, expected_weights)
    assert torch.equal(right, expected)
    assert torch.equal(right_mask, expected_weights)


def test_encoded_with_jitter():
    traces = pd.DataFrame(
        [
            [0, 0, 0, 4, 4],
            [0, 1, 4, 8, 4],
            [0, 0, 8, 10, 2],
            [1, 0, 0, 3, 3],
            [1, 1, 3, 7, 4],
            [1, 0, 7, 10, 3],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    length = 6
    duration = 10
    encoder = seq.SequenceEncoder(
        max_length=length, duration=duration, jitter=0.1
    )
    encoded = encoder.encode(traces)
    for _ in range(10):
        for i in range(len(encoded)):
            (left, left_mask), (right, right_mask) = encoded[i]
            assert torch.equal(left, right)
            assert torch.equal(left_mask, right_mask)
            assert left.shape == (6, 2)
            assert left_mask.shape == (6,)


@pytest.mark.parametrize(
    "encoded,length,norm_duration,expected",
    [
        (
            tensor(
                [
                    [
                        [0.9, 0, 0, 0, 0],
                        [0, 0.1, 0.8, 0.1, 0.25],
                        [0.2, 0.2, 0.1, 0.5, 0.75],
                        [0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                    ],
                    [
                        [1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0.25],
                        [0, 0, 0, 1, 0.25],
                        [0, 0, 1, 0, 0.5],
                        [0, 1, 0, 0, 0],
                    ],
                ]
            ),
            5,
            144,
            pd.DataFrame(
                [
                    [0, "a", 0, 36, 36],
                    [0, "b", 36, 144, 108],
                    [1, "a", 0, 36, 36],
                    [1, "b", 36, 72, 36],
                    [1, "a", 72, 144, 72],
                ],
                columns=["pid", "act", "start", "end", "duration"],
            ),
        )
    ],
)
def test_decode_descretised(encoded, length, norm_duration, expected):
    encoder = seq.SequenceEncoder(max_length=length, duration=norm_duration)
    _ = encoder.encode(expected)
    result = encoder.decode(encoded)
    result["duration"] = result["end"] - result["start"]
    pd.testing.assert_frame_equal(expected, result)
