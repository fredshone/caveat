from pandas import DataFrame, Series

from caveat.features.times import average_end_times, average_start_times


def test_average_start_times():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "work", "start": 2},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
            {"pid": 1, "act": "work", "start": 1},
        ]
    )
    expected = Series(
        {
            ("average start time", "home"): 2.0,
            ("average start time", "work"): 1.5,
        }
    )
    result = average_start_times(population)
    assert result.equals(expected)


def test_average_end_times():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "end": 1},
            {"pid": 0, "act": "work", "end": 3},
            {"pid": 0, "act": "home", "end": 4},
            {"pid": 1, "act": "home", "end": 1},
            {"pid": 1, "act": "work", "end": 2},
        ]
    )
    expected = Series(
        {("average end time", "home"): 2.0, ("average end time", "work"): 2.5}
    )
    result = average_end_times(population)
    assert result.equals(expected)
