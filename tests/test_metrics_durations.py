from pandas import DataFrame, Series

from caveat import report
from caveat.features import durations


def test_average_durations():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 3},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 1},
        ]
    )
    expected = Series(
        {("average duration", "home"): 10.0, ("average duration", "work"): 2.0}
    )
    result = durations.average_activity_durations(population)
    assert result.equals(expected)


def test_average_activity_plan_seq_durations():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 3},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 8},
            {"pid": 1, "act": "work", "duration": 1},
        ]
    )
    expected = Series(
        {
            ("average act plan seq duration", "home0"): 9.0,
            ("average act plan seq duration", "work1"): 2.0,
            ("average act plan seq duration", "home2"): 10.0,
        }
    )
    result = durations.average_activity_plan_seq_durations(population)
    assert result.equals(expected)


def test_average_activity_seq_durations():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 3},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 8},
            {"pid": 1, "act": "work", "duration": 1},
        ]
    )
    expected = Series(
        {
            ("average act seq duration", "home0"): 9.0,
            ("average act seq duration", "work0"): 2.0,
            ("average act seq duration", "home1"): 10.0,
        }
    )
    result = durations.average_activity_seq_durations(population)
    assert result.equals(expected)


def test_report_average_durations():
    observed = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 3},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 1},
        ]
    )
    ys = {
        "y1": DataFrame(
            [
                {"pid": 0, "act": "home", "duration": 10},
                {"pid": 0, "act": "work", "duration": 3},
                {"pid": 0, "act": "home", "duration": 10},
                {"pid": 1, "act": "home", "duration": 10},
                {"pid": 1, "act": "work", "duration": 1},
            ]
        ),
        "y2": DataFrame(
            [
                {"pid": 0, "act": "home", "duration": 10},
                {"pid": 0, "act": "work", "duration": 3},
                {"pid": 0, "act": "home", "duration": 10},
                {"pid": 1, "act": "home", "duration": 10},
                {"pid": 1, "act": "work", "duration": 3},
            ]
        ),
    }
    data = {
        "observed": {"home": 10.0, "work": 2.0},
        "y1": {"home": 10.0, "work": 2.0},
        "y1 delta": {"home": 0.0, "work": 0.0},
        "y2": {"home": 10.0, "work": 3.0},
        "y2 delta": {"home": 0.0, "work": 1.0},
    }
    expected = DataFrame(
        {
            key: Series(
                {
                    ("average duration", act): value
                    for act, value in values.items()
                }
            )
            for key, values in data.items()
        }
    )
    result = report.report_diff(
        observed, ys, durations.average_activity_durations
    )
    assert result.equals(expected)
