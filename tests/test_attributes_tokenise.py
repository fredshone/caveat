import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from torch import Tensor
from torch.testing import assert_close

from caveat.encoding.attributes import TokenAttributeEncoder, tokenize


def test_encode_no_encodings():
    data = pd.Series(["M", "F", "F"])
    encoded, encodings = tokenize(data, None)
    assert_close(encoded, Tensor([1, 0, 0]).int())
    assert encodings == {"M": 1, "F": 0}


def test_encode_with_encodings():
    data = pd.Series(["M", "F", "F"])
    encodings = {"M": 0, "F": 1}
    encoded, encodings = tokenize(data, encodings)
    assert_close(encoded, Tensor([0, 1, 1]).int())
    assert encodings == {"M": 0, "F": 1}


def test_encoder_ordinal():
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [10, 50, 20], "gender": ["M", "F", "F"]}
    )
    encoder = TokenAttributeEncoder(config={"age": "ordinal"})
    with pytest.raises(UserWarning) as w:
        _ = encoder.encode(data)
        assert w.message.contains("Ordinal encoding not supported")


def test_encoder_nominal():
    config = {"gender": "nominal"}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = TokenAttributeEncoder(config=config)
    encoded = encoder.encode(data)
    expected = Tensor([[1], [0], [0]]).int()
    assert_close(encoded, expected)
    assert encoder.config["gender"] == {
        "nominal": {"M": 1, "F": 0},
        "location": 0,
        "type": "object",
    }
    assert encoder.sizes == [2]


def test_re_encoder_nominal():
    config = {"gender": "nominal"}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = TokenAttributeEncoder(config=config)
    _ = encoder.encode(data)
    new = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "M", "F"]}
    )
    new_encoded = encoder.encode(new)
    assert_close(new_encoded, Tensor([[1], [1], [0]]).int())
    assert encoder.config["gender"] == {
        "nominal": {"M": 1, "F": 0},
        "location": 0,
        "type": "object",
    }
    assert encoder.sizes == [2]


def test_re_encoder_new_cat_nominal():
    config = {"gender": "nominal"}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = TokenAttributeEncoder(config=config)
    _ = encoder.encode(data)
    new = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "X", "F"]}
    )
    with pytest.raises(UserWarning) as w:
        encoder.encode(new)
        assert w.message.contains("X")


def test_encoder_missing_columns():
    config = {"unknown": "nominal"}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = TokenAttributeEncoder(config=config)
    with pytest.raises(UserWarning):
        encoder.encode(data)


def test_encoder_bad_config():
    bad_config = {"age": {"nominal": (0, 10)}}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = TokenAttributeEncoder(config=bad_config)
    with pytest.raises(UserWarning):
        encoder.encode(data)


def test_encoder_multi():
    config = {"gender": "nominal", "age": "nominal"}
    data = pd.DataFrame(
        {
            "pid": [0, 1, 2],
            "age": ["old", "old", "young"],
            "gender": ["M", "F", "F"],
        }
    )
    encoder = TokenAttributeEncoder(config=config)
    encoded = encoder.encode(data)
    expected = Tensor([[1, 0], [0, 0], [0, 1]]).int()
    assert_close(encoded, expected)
    assert encoder.config["gender"] == {
        "nominal": {"M": 1, "F": 0},
        "location": 0,
        "type": "object",
    }
    assert encoder.config["age"] == {
        "nominal": {"old": 0, "young": 1},
        "location": 1,
        "type": "object",
    }
    assert encoder.sizes == [2, 2]


def test_re_encoder_mixed():
    config = {"gender": "nominal", "age": "nominal"}
    data = pd.DataFrame(
        {
            "pid": [0, 1, 2],
            "age": ["old", "old", "young"],
            "gender": ["M", "F", "F"],
        }
    )
    encoder = TokenAttributeEncoder(config=config)
    _ = encoder.encode(data)
    new_data = pd.DataFrame(
        {
            "pid": [0, 1, 2],
            "age": ["young", "old", "young"],
            "gender": ["M", "M", "F"],
        }
    )
    new_encoded = encoder.encode(new_data)
    expected = Tensor([[1, 1], [1, 0], [0, 1]]).int()
    assert_close(new_encoded, expected)
    assert encoder.config["gender"] == {
        "nominal": {"M": 1, "F": 0},
        "location": 0,
        "type": "object",
    }
    assert encoder.config["age"] == {
        "nominal": {"old": 0, "young": 1},
        "location": 1,
        "type": "object",
    }
    assert encoder.sizes == [2, 2]


def test_decode_attributes():
    config = {"gender": "nominal", "age": "nominal"}
    data = pd.DataFrame(
        {
            "pid": [0, 1, 2],
            "age": ["old", "old", "young"],
            "gender": ["M", "F", "F"],
        }
    )
    encoder = TokenAttributeEncoder(config=config)
    encoded = encoder.encode(data)
    expected = Tensor([[1, 0], [0, 0], [0, 1]]).int()
    assert_close(encoded, expected)

    preds = Tensor(
        [
            [[0.1, 0.5], [0.9, 0.3]],
            [[0.9, 0.5], [0.9, 0.3]],
            [[0.5, 0.2], [0.2, 0.5]],
        ]
    )
    decoded_data = encoder.decode(preds)[data.columns]
    assert_frame_equal(data, decoded_data, check_dtype=True, check_exact=True)
