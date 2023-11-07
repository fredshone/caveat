from pandas import DataFrame, Series

from caveat.features import times


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
    result = times.average_start_times(population)
    assert result.equals(expected)


def test_average_start_times_plan_seq():
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
            ("average start time plan seq", "home0"): 0.0,
            ("average start time plan seq", "work1"): 1.5,
            ("average start time plan seq", "home2"): 6.0,
        }
    )
    result = times.average_start_times_plan_seq(population)
    assert result.equals(expected)


def test_average_start_times_act_seq():
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
            ("average start time act seq", "home0"): 0.0,
            ("average start time act seq", "work0"): 1.5,
            ("average start time act seq", "home1"): 6.0,
        }
    )
    result = times.average_start_times_act_seq(population)
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
    result = times.average_end_times(population)
    assert result.equals(expected)


def test_average_end_times_plan_seq():
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
        {
            ("average end time plan seq", "home0"): 1.0,
            ("average end time plan seq", "work1"): 2.5,
            ("average end time plan seq", "home2"): 4.0,
        }
    )
    result = times.average_end_times_plan_seq(population)
    assert result.equals(expected)


def test_average_end_times_act_seq():
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
        {
            ("average end time act seq", "home0"): 1.0,
            ("average end time act seq", "work0"): 2.5,
            ("average end time act seq", "home1"): 4.0,
        }
    )
    result = times.average_end_times_act_seq(population)
    assert result.equals(expected)
