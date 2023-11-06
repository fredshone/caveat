from pandas import DataFrame, Series

from caveat import report
from caveat.features.structural import start_and_end_acts


def test_start_and_end_acts():
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
            ("structural", "first act is home"): 1.0,
            ("structural", "last act is home"): 0.5,
        }
    )
    result = start_and_end_acts(population, target="home")
    assert result.equals(expected)


def test_compare_activity_start_and_end_acts():
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
    expected = DataFrame(
        {
            "observed": Series(
                {
                    ("structural", "first act is home"): 1.0,
                    ("structural", "last act is home"): 0.5,
                }
            ),
            "y1": Series(
                {
                    ("structural", "first act is home"): 1.0,
                    ("structural", "last act is home"): 0.5,
                }
            ),
            "y1 delta": Series(
                {
                    ("structural", "first act is home"): 0.0,
                    ("structural", "last act is home"): 0.0,
                }
            ),
            "y2": Series(
                {
                    ("structural", "first act is home"): 1.0,
                    ("structural", "last act is home"): 1.0,
                }
            ),
            "y2 delta": Series(
                {
                    ("structural", "first act is home"): 0.0,
                    ("structural", "last act is home"): 0.5,
                }
            ),
        }
    )
    result = report.report_diff(x, ys, start_and_end_acts)
    assert result.equals(expected)
