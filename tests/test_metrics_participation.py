from pandas import DataFrame, Series

from caveat import report
from caveat.features import participation


def test_participation():
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
            ("participation rate", "home"): 1.5,
            ("participation rate", "work"): 1.0,
        }
    )
    result = participation.participation_rates(population)
    assert result.equals(expected)


def test_act_plan_seq_participation_rates():
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
            ("act plan seq participation rate", "home0"): 1.0,
            ("act plan seq participation rate", "work1"): 1.0,
            ("act plan seq participation rate", "home2"): 0.5,
        }
    )
    result = participation.act_plan_seq_participation_rates(population)
    assert result.equals(expected)


def test_act_seq_participation_rates():
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
            ("act seq participation rate", "home0"): 1.0,
            ("act seq participation rate", "work0"): 1.0,
            ("act seq participation rate", "home1"): 0.5,
        }
    )
    result = participation.act_seq_participation_rates(population)
    assert result.equals(expected)


def test_compare_participation_rates():
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
        "observed": {"home": 1.5, "work": 1.0},
        "y1": {"home": 1.5, "work": 1.0},
        "y1 delta": {"home": 0.0, "work": 0.0},
        "y2": {"home": 2.0, "work": 0.5},
        "y2 delta": {"home": 0.5, "work": -0.5},
    }
    expected = DataFrame(
        {
            key: Series(
                {("participation rate", k): v for k, v in value.items()}
            )
            for key, value in data.items()
        }
    )
    result = report.report_diff(x, ys, participation.participation_rates)
    assert result.equals(expected)


def test_combinations_with_replacement():
    array = ["a", "b", "c"]
    tuple_length = 2
    expected = [
        ["a", "a"],
        ["a", "b"],
        ["a", "c"],
        ["b", "b"],
        ["b", "c"],
        ["c", "c"],
    ]
    result = participation.combinations_with_replacement(array, tuple_length)
    assert result == expected


def test_calc_pair_rate():
    act_counts = DataFrame(
        {
            "home": [1, 2, 2, 2, 3],
            "work": [1, 1, 1, 1, 0],
            "school": [1, 0, 0, 0, 0],
        }
    )
    assert participation.calc_pair_rate(act_counts, ("home", "home")) == 0.8
    assert participation.calc_pair_rate(act_counts, ("home", "work")) == 0.8


def test_participation_pairs():
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
            ("joint participation rate", "home+work"): 1,
            ("joint participation rate", "home+home"): 0.5,
            ("joint participation rate", "work+work"): 0.0,
        }
    )
    result = participation.joint_participation_rates(population)
    assert result.equals(expected)


def test_report_participation_pairs():
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
        "observed": {"home+work": 1, "home+home": 0.5, "work+work": 0.0},
        "y1": {"home+work": 1, "home+home": 0.5, "work+work": 0.0},
        "y1 delta": {"home+work": 0.0, "home+home": 0.0, "work+work": 0.0},
        "y2": {"home+work": 0.5, "home+home": 1.0, "work+work": 0.0},
        "y2 delta": {"home+work": -0.5, "home+home": 0.5, "work+work": 0.0},
    }
    expected = DataFrame(
        {
            key: Series(
                {("joint participation rate", k): v for k, v in value.items()}
            )
            for key, value in data.items()
        }
    )
    result = report.report_diff(x, ys, participation.joint_participation_rates)
    assert result.equals(expected)
