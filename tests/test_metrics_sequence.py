from pandas import DataFrame, Series

from caveat import report
from caveat.features.sequence import sequence_probs


def test_sequence_probs():
    population = DataFrame(
        [
            {"pid": 0, "act": "home"},
            {"pid": 0, "act": "work"},
            {"pid": 0, "act": "home"},
            {"pid": 1, "act": "home"},
            {"pid": 1, "act": "work"},
        ]
    )
    expected = Series(
        {
            ("sequence rate", "home>work>home"): 0.5,
            ("sequence rate", "home>work"): 0.5,
        }
    )
    result = sequence_probs(population)
    assert result.equals(expected)


def test_report_sequence_probs():
    x = DataFrame(
        [
            {"pid": 0, "act": "home"},
            {"pid": 0, "act": "work"},
            {"pid": 0, "act": "home"},
            {"pid": 1, "act": "home"},
            {"pid": 1, "act": "work"},
        ]
    )
    ys = {
        "y1": DataFrame(
            [
                {"pid": 0, "act": "home"},
                {"pid": 0, "act": "work"},
                {"pid": 0, "act": "home"},
                {"pid": 1, "act": "home"},
                {"pid": 1, "act": "work"},
            ]
        ),
        "y2": DataFrame(
            [
                {"pid": 0, "act": "home"},
                {"pid": 0, "act": "work"},
                {"pid": 0, "act": "home"},
                {"pid": 1, "act": "home"},
                {"pid": 1, "act": "home"},
            ]
        ),
    }
    data = {
        "observed": {"home>work>home": 0.5, "home>work": 0.5, "home>home": 0.0},
        "y1": {"home>work>home": 0.5, "home>work": 0.5, "home>home": 0.0},
        "y1 delta": {"home>work>home": 0.0, "home>work": 0.0, "home>home": 0.0},
        "y2": {"home>work>home": 0.5, "home>work": 0.0, "home>home": 0.5},
        "y2 delta": {
            "home>work>home": 0.0,
            "home>work": -0.5,
            "home>home": 0.5,
        },
    }
    expected = DataFrame(
        {
            key: Series({("sequence rate", k): v for k, v in value.items()})
            for key, value in data.items()
        }
    )
    result = report.report_diff(x, ys, sequence_probs)
    assert result.equals(expected)