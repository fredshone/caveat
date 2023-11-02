from pandas import DataFrame, Series

from caveat.metrics.sequence import report_sequence_probs, sequence_probs


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
            ("sequence prob.", "home>work>home"): 0.5,
            ("sequence prob.", "home>work"): 0.5,
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
            key: Series({("sequence prob.", k): v for k, v in value.items()})
            for key, value in data.items()
        }
    )
    result = report_sequence_probs(x, ys)
    assert result.equals(expected)
