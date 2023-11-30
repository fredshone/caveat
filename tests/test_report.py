from pandas import DataFrame, Series

from caveat.features.times import average_start_times
from caveat.report import describe, report_diff


def test_report_average_start_times():
    observed = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "work", "start": 2},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
            {"pid": 1, "act": "work", "start": 1},
        ]
    )
    ys = {
        "y1": DataFrame(
            [
                {"pid": 0, "act": "home", "start": 0},
                {"pid": 0, "act": "work", "start": 2},
                {"pid": 0, "act": "home", "start": 6},
                {"pid": 1, "act": "home", "start": 0},
                {"pid": 1, "act": "work", "start": 1},
            ]
        ),
        "y2": DataFrame(
            [
                {"pid": 0, "act": "home", "start": 0},
                {"pid": 0, "act": "work", "start": 2},
                {"pid": 0, "act": "home", "start": 6},
                {"pid": 1, "act": "home", "start": 0},
                {"pid": 1, "act": "work", "start": 3},
            ]
        ),
    }
    data = {
        "observed": {"home": 2.0, "work": 1.5},
        "y1": {"home": 2.0, "work": 1.5},
        "y1 delta": {"home": 0.0, "work": 0.0},
        "y2": {"home": 2.0, "work": 2.5},
        "y2 delta": {"home": 0.0, "work": 1.0},
    }
    expected = DataFrame(
        {
            key: Series(
                {
                    ("average start time", act): value
                    for act, value in values.items()
                }
            )
            for key, values in data.items()
        }
    )
    result = report_diff(observed, ys, average_start_times)
    assert result.equals(expected)


def test_describe():
    observed = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "work", "start": 2},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
            {"pid": 1, "act": "work", "start": 1},
        ]
    )
    ys = {
        "y0": DataFrame(
            [
                {"pid": 0, "act": "home", "start": 0},
                {"pid": 0, "act": "work", "start": 2},
                {"pid": 0, "act": "home", "start": 6},
                {"pid": 1, "act": "home", "start": 0},
                {"pid": 1, "act": "work", "start": 1},
            ]
        ),
        "y1": DataFrame(
            [
                {"pid": 0, "act": "home", "start": 0},
                {"pid": 0, "act": "work", "start": 0},
                {"pid": 0, "act": "home", "start": 6},
                {"pid": 1, "act": "home", "start": 0},
                {"pid": 1, "act": "work", "start": 1},
            ]
        ),
        "y2": DataFrame(
            [
                {"pid": 0, "act": "home", "start": 0},
                {"pid": 0, "act": "work", "start": 2},
                {"pid": 0, "act": "home", "start": 6},
                {"pid": 1, "act": "home", "start": 0},
                {"pid": 1, "act": "work", "start": 3},
            ]
        ),
    }
    data = {
        "observed": {"home": 2.0, "work": 1.5},
        "y0": {"home": 2.0, "work": 1.5},
        "y1": {"home": 2.0, "work": 0.5},
        "y2": {"home": 2.0, "work": 2.5},
        "mean": {"home": 2.0, "work": 1.5},
        "mean delta": {"home": 0.0, "work": 0.0},
        "std": {"home": 0.0, "work": 1},
    }
    expected = DataFrame(
        {
            key: Series(
                {
                    ("average start time", act): value
                    for act, value in values.items()
                }
            )
            for key, values in data.items()
        }
    )
    result = describe(observed, ys, average_start_times)
    assert result.equals(expected)
