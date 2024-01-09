from pandas import DataFrame

from caveat.data.samplers import biased_sample, random_sample, sample_observed


def test_random_sample():
    data = DataFrame(
        [
            [0, "a", 0, 2, 2],
            [0, "b", 2, 5, 3],
            [0, "a", 5, 10, 5],
            [1, "a", 0, 3, 3],
            [1, "b", 3, 5, 2],
            [1, "a", 5, 10, 5],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    assert len(random_sample(data, p=1)) == 6
    assert random_sample(data, p=1).pid.nunique() == 2
    assert len(random_sample(data, p=0.5)) == 3
    assert random_sample(data, p=0.5).pid.nunique() == 1
    assert random_sample(data, p=0).empty


def test_biased_sample():
    data = DataFrame(
        [
            [0, "a", 0, 2, 3],
            [0, "b", 2, 5, 3],
            [0, "a", 5, 10, 4],
            [1, "a", 0, 3, 3],
            [1, "b", 3, 5, 2],
            [1, "a", 5, 10, 5],
            [2, "a", 0, 3, 3],
            [2, "b", 3, 5, 2],
            [2, "a", 5, 10, 5],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    assert len(biased_sample(data, p=1, threshold=3)) == 9
    assert biased_sample(data, p=1, threshold=3).pid.nunique() == 3
    assert len(biased_sample(data, p=0.5, threshold=3)) == 6
    assert biased_sample(data, p=0.5, threshold=3).pid.nunique() == 2
    assert len(biased_sample(data, p=0, threshold=3)) == 3
    assert biased_sample(data, p=0, threshold=3).pid.nunique() == 1


def test_configuration_none_sample():
    config = {}
    data = DataFrame(
        [
            [0, "a", 0, 2, 2],
            [0, "b", 2, 5, 3],
            [0, "a", 5, 10, 5],
            [1, "a", 0, 3, 3],
            [1, "b", 3, 5, 2],
            [1, "a", 5, 10, 5],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    assert (sample_observed(data, config) == data).all().all()


def test_configuration_random_sample():
    config = {"sampler_params": {"type": "random", "p": 0.0}}
    data = DataFrame(
        [
            [0, "a", 0, 2, 2],
            [0, "b", 2, 5, 3],
            [0, "a", 5, 10, 5],
            [1, "a", 0, 3, 3],
            [1, "b", 3, 5, 2],
            [1, "a", 5, 10, 5],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    assert sample_observed(data, config).empty
    assert not data.empty


def test_configuration_biased_sample():
    config = {"sampler_params": {"type": "biased", "p": 0.0, "threshold": 10}}
    data = DataFrame(
        [
            [0, "a", 0, 2, 2],
            [0, "b", 2, 5, 3],
            [0, "a", 5, 10, 5],
            [1, "a", 0, 3, 3],
            [1, "b", 3, 5, 2],
            [1, "a", 5, 10, 5],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    assert sample_observed(data, config).empty
    assert not data.empty
