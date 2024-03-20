import pandas as pd
import pytest
from torch import Tensor
from torch.testing import assert_close

from caveat.encoders.attributes import (
    AttributeEncoder,
    nominal_encode,
    ordinal_encode,
)


def test_encode_ordinal():
    data = pd.Series([34, 96, 15])
    min, max = 0, 100
    encoded = ordinal_encode(data, min, max)
    expected = Tensor([[0.34], [0.96], [0.15]]).float()
    assert_close(encoded, expected)


def test_encode_nominal():
    data = pd.Series(["M", "F", "F"])
    encoded = nominal_encode(data)
    expected = Tensor([[0, 1], [1, 0], [1, 0]]).float()
    assert_close(encoded, expected)


def test_encoder_ordinal():
    config = {"age": {"ordinal": (0, 100)}}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = AttributeEncoder(config=config)
    encoded = encoder.encode(data)
    expected = Tensor([[0.34], [0.96], [0.15]]).float()
    assert_close(encoded, expected)


def test_encoder_nominal():
    config = {"gender": "nominal"}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = AttributeEncoder(config=config)
    encoded = encoder.encode(data)
    expected = Tensor([[0, 1], [1, 0], [1, 0]]).float()
    assert_close(encoded, expected)


def test_encoder_missing_columns():
    config = {"unknown": "nominal"}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = AttributeEncoder(config=config)
    with pytest.raises(UserWarning):
        encoder.encode(data)


def test_encoder_bad_config():
    bad_config = {"age": {"nominal": (0, 10)}}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = AttributeEncoder(config=bad_config)
    with pytest.raises(UserWarning):
        encoder.encode(data)


def test_encoder_mixed():
    config = {"gender": "nominal", "age": {"ordinal": (0, 100)}}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = AttributeEncoder(config=config)
    encoded = encoder.encode(data)
    expected = Tensor([[0, 1, 0.34], [1, 0, 0.96], [1, 0, 0.15]]).float()
    assert_close(encoded, expected)
