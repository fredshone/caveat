from pandas import DataFrame, Series

from caveat import report
from caveat.features import transitions


def test_transitions():
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
            ("transition rate", "home->work"): 1.0,
            ("transition rate", "work->home"): 0.5,
        }
    )
    result = transitions.transition_rates(population)
    assert result.equals(expected)


def test_compare_transitions():
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
        "observed": {"home->work": 1.0, "work->home": 0.5, "home->home": 0.0},
        "y1": {"home->work": 1.0, "work->home": 0.5, "home->home": 0.0},
        "y1 delta": {"home->work": 0.0, "work->home": 0.0, "home->home": 0.0},
        "y2": {"home->work": 0.5, "work->home": 0.5, "home->home": 0.5},
        "y2 delta": {"home->work": -0.5, "work->home": 0.0, "home->home": 0.5},
    }
    expected = DataFrame(
        {
            key: Series({("transition rate", k): v for k, v in value.items()})
            for key, value in data.items()
        }
    )
    result = report.report_diff(x, ys, transitions.transition_rates)
    assert result.equals(expected)
