import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from torch import Tensor

from caveat.data import decode


@pytest.mark.parametrize(
    "encoded,length,expected",
    [
        (
            Tensor(
                [
                    [[[0.8, 0.1, 0.1], [0.7, 0.0, 0.0], [0.1, 0.5, 0.1]]],
                    [[[1, 0, 0], [0, 0, 1], [1, 0, 0]]],
                ]
            ),
            144,
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
            Tensor([[[[1, 0, 0], [1, 0, 0], [0, 1, 0]]]]),
            3,
            pd.DataFrame(
                [[0, "a", 0, 2], [0, "b", 2, 3]],
                columns=["pid", "act", "start", "end"],
            ),
        ),
    ],
)
def test_decode_descretised(encoded, length, expected):
    mapping = {0: "a", 1: "b", 2: "c"}
    result = decode.decode_descretised(encoded, mapping, length)
    assert_frame_equal(expected, result)
