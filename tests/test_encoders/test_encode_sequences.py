import numpy as np
import pandas as pd
import pytest
import torch

from caveat.encoding import sequence as seq


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


def test_encoder():
    schedules = pd.DataFrame(
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
    attributes = torch.tensor([[0, 0], [1, 1]])
    attributes_weights = torch.tensor([[1, 2], [3, 4]])
    encoder = seq.SequenceEncoder(max_length=length, norm_duration=duration)
    encoded_data = encoder.encode(schedules, attributes, attributes_weights)
    encoded_schedule = encoded_data.schedules
    masks = encoded_data.schedule_weights
    labels = encoded_data.labels
    labels_weights = encoded_data.label_weights

    assert torch.equal(encoded_schedule[0], expected)
    assert torch.equal(masks[0], expected_weights)
    assert torch.equal(labels[0], torch.tensor([0, 0]))
    assert torch.equal(labels_weights[0], torch.tensor([1, 2]))
