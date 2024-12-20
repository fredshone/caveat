import numpy as np
import pandas as pd
import pytest
import torch

from caveat.encoding import seq2seq as seq


@pytest.mark.parametrize(
    "acts,durations,modes,distances,act_weights,expected,expects_weights",
    [
        (
            [2, 3, 2],
            [0.3, 0.2, 0.5],
            [0, 1, 0],
            [0.1, 0.2, 0.3],
            {0: 0.1, 1: 0.1, 2: 0.3, 3: 0.5},
            np.array(
                [
                    [0, 0.0, 0, 0.0],
                    [2, 0.3, 0, 0.1],
                    [3, 0.2, 1, 0.2],
                    [2, 0.5, 0, 0.3],
                    [1, 0.0, 0, 0],
                    [1, 0.0, 0, 0],
                ],
                dtype=np.float32,
            ),
            np.array([0.1, 0.3, 0.5, 0.3, 0.1, 0.0], dtype=np.float32),
        )
    ],
)
def test_encode_sequence(
    acts, durations, modes, distances, act_weights, expected, expects_weights
):
    lhs, rhs, weights = seq.encode_sequences(
        acts,
        durations,
        modes,
        distances,
        target_acts=acts,
        target_durations=durations,
        target_modes=modes,
        target_distances=distances,
        max_length=6,
        encoding_width=4,
        act_weights=act_weights,
        sos=0,
        eos=1,
    )
    np.testing.assert_array_equal(lhs, expected)
    np.testing.assert_array_equal(rhs, expected)
    np.testing.assert_array_equal(weights, expects_weights)


def test_encoder():
    lhs = pd.DataFrame(
        [
            [0, 0, 0, 4, 4, 0, 0.1],
            [0, 1, 4, 8, 4, 1, 0.1],
            [0, 0, 8, 10, 2, 0, 0.1],
            [1, 0, 0, 3, 3, 0, 1],
            [1, 1, 3, 7, 4, 1, 1],
            [1, 0, 7, 10, 3, 0, 1],
        ],
        columns=["pid", "act", "start", "end", "duration", "mode", "distance"],
    )
    rhs = pd.DataFrame(
        [
            [0, 0, 4, 4, 0, 0.1],
            [1, 4, 8, 4, 1, 0.1],
            [0, 8, 10, 2, 0, 0.1],
            [0, 0, 3, 3, 0, 1],
            [1, 3, 7, 4, 1, 1],
            [0, 7, 10, 3, 0, 1],
        ],
        columns=[
            "target_act",
            "target_start",
            "target_end",
            "target_duration",
            "target_mode",
            "target_distance",
        ],
    )
    schedules = pd.concat([lhs, rhs], axis=1)
    length = 6
    duration = 10
    expected = torch.tensor(
        [
            [0, 0.0, 0, 0],
            [2, 0.4, 0, 0.1],
            [3, 0.4, 1, 0.1],
            [2, 0.2, 0, 0.1],
            [1, 0.0, 0, 0],
            [1, 0.0, 0, 0],
        ]
    )
    expected_weights = torch.tensor(
        [1 / 2, 1 / 1.2, 1 / 0.8, 1 / 1.2, 1 / 2, 0]
    )
    encoder = seq.Seq2SeqEncoder(max_length=length, norm_duration=duration)
    encoded = encoder.encode(schedules, None)

    assert torch.equal(encoded.lhs[0], expected)
    assert torch.equal(encoded.rhs[0], expected)
    assert torch.equal(encoded.lhs_weights[0], expected_weights)
