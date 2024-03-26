import pytest
from pandas import DataFrame, testing

from caveat.data.loaders import validate_attributes_index, validate_attributes, load_and_validate_attributes


def test_validate_attributes_index():
    attributes = DataFrame(
        {
            "pid": [0, 1],
            "gender": ["F", "M"],
            "age": [25, 30],
        }
    )
    schedules = DataFrame(
        [
            [1, "a", 0, 2, 2],
            [1, "b", 2, 5, 3],
            [1, "a", 5, 10, 5],
            [2, "a", 0, 3, 3],
            [2, "b", 3, 5, 2],
            [2, "a", 5, 10, 5],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    with pytest.raises(UserWarning):
        attributes = validate_attributes_index(attributes, schedules)


test_data = [
    ({"pid": [0, 1], "gender": ["F", "M"], "age": [25, 30]}, True),
    ({"pid": [0, 1], "age": [25, 30]}, False),
    ({"id": [0, 1], "gender": ["F", "M"], "age": [25, 30]}, False),
]
ids = ["valid", "missing_column", "wrong_column"]

@pytest.mark.parametrize("data,valid", test_data, ids=ids)
def test_validate_attributes(data, valid):
    attributes = DataFrame(data)
    config = {
        "conditionals": {
            "gender": "nominal",
            "age": {"ordinal": (0, 100)},
            }
        }
    if not valid:
        with pytest.raises(UserWarning):
            validate_attributes(attributes, config)
    else:
        validate_attributes(attributes, config)


def test_load_and_validate_attributes_none(schedules):
    config = {}
    assert load_and_validate_attributes(config, schedules) == (None, None)
    

def test_load_and_validate_attributes(schedules):
    config = {
        "attributes_path": "tests/fixtures/attributes.csv",
        "conditionals": {
            "gender": "nominal",
            "age": {"ordinal": (0, 100)},
            }
        }
    attributes, synthetic_attributes = load_and_validate_attributes(config, schedules)
    assert len(attributes) == 2
    testing.assert_frame_equal(attributes, synthetic_attributes)


def test_load_and_validate_synthetic_attributes(schedules):
    config = {
        "attributes_path": "tests/fixtures/attributes.csv",
        "synthetic_attributes_path": "tests/fixtures/synthetic_attributes.csv",
        "conditionals": {
            "gender": "nominal",
            "age": {"ordinal": (0, 100)},
            }
        }
    attributes, synthetic_attributes = load_and_validate_attributes(config, schedules)
    assert len(attributes) == 2
    assert len(synthetic_attributes) == 4
