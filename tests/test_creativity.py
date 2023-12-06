from pandas import DataFrame

from caveat.features.uniqueness import external, hash_population, internal


def test_hash_population():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    assert hash_population(population) == {"home10work10", "home10work10home10"}


def test_internal_uniqueness_full():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    hashed = hash_population(population)
    assert internal(population, hashed) == 1


def test_internal_uniqueness_half():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
        ]
    )
    hashed = hash_population(population)
    assert internal(population, hashed) == 0.5


def test_external_uniqueness_full():
    a = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    b = DataFrame(
        [
            {"pid": 2, "act": "home", "duration": 10},
            {"pid": 2, "act": "work", "duration": 15},
            {"pid": 2, "act": "home", "duration": 5},
            {"pid": 3, "act": "home", "duration": 10},
            {"pid": 3, "act": "work", "duration": 10},
            {"pid": 3, "act": "shop", "duration": 10},
        ]
    )
    assert external(a, hash_population(a), b, hash_population(b)) == 1


def test_external_uniqueness_partial():
    a = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    b = DataFrame(
        [
            {"pid": 2, "act": "home", "duration": 10},
            {"pid": 2, "act": "work", "duration": 10},
            {"pid": 2, "act": "home", "duration": 10},
            {"pid": 3, "act": "home", "duration": 10},
            {"pid": 3, "act": "work", "duration": 10},
            {"pid": 3, "act": "shop", "duration": 10},
        ]
    )
    assert external(a, hash_population(a), b, hash_population(b)) == 0.75
