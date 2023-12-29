import numpy as np
import pandas as pd
import pytest
import torch
from torch import tensor

from caveat.encoders import seq


@pytest.mark.parametrize(
    "acts,durations,expected,expected_mask",
    [
        (
            [2, 3, 2],
            [0.3, 0.2, 0.5],
            np.array(
                [[0, 0.0], [2, 0.3], [3, 0.2], [2, 0.5], [1, 0.0], [1, 0.0]],
                dtype=np.float32,
            ),
            np.array([1, 1, 1, 1, 1, 0]),
        )
    ],
)
def test_encode_sequence(acts, durations, expected, expected_mask):
    result, mask = seq.encode_sequence(
        acts, durations, max_length=6, encoding_width=2, sos=0, eos=1
    )
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(mask, expected_mask)


def test_encode_population():
    traces = pd.DataFrame(
        [
            [0, "a", 0, 2, 2],
            [0, "b", 2, 5, 3],
            [0, "a", 5, 10, 5],
            [1, "a", 0, 3, 3],
            [1, "b", 3, 5, 2],
            [1, "a", 5, 10, 5],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    map = {"<SOS>": 0, "<EOS>": 1, "a": 2, "b": 3}
    encoded = seq.SequenceEncodedPlans(
        traces, max_length=6, acts_to_index=map, norm_duration=10
    )
    expected = np.array(
        [
            [[0.0, 0.0], [2, 0.2], [3, 0.3], [2, 0.5], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [2, 0.3], [3, 0.2], [2, 0.5], [1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    expected = torch.from_numpy(expected)
    assert torch.equal(encoded.encoded, expected)


def test_encoded_weights():
    traces = pd.DataFrame(
        [
            [0, "a", 0, 2, 2],
            [0, "b", 2, 5, 3],
            [0, "a", 5, 10, 5],
            [1, "a", 0, 3, 3],
            [1, "b", 3, 5, 2],
            [1, "a", 5, 10, 5],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    expected_weights = torch.tensor([1 / 120, 1 / 120, 1 / 15, 1 / 5])
    expected_mask = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    encoder = seq.SequenceEncoder(max_length=10, norm_duration=10)
    encoded = encoder.encode(traces)
    assert torch.equal(encoded.encoding_weights, expected_weights)
    assert torch.equal(encoded[0][1], expected_mask)


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
    encoder = seq.SequenceEncoder(
        max_length=length, norm_duration=norm_duration
    )
    _ = encoder.encode(expected)
    result = encoder.decode(encoded)
    result["duration"] = result["end"] - result["start"]
    pd.testing.assert_frame_equal(expected, result)
