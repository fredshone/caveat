from numpy import array
from pandas import DataFrame

from caveat.features import transitions
from caveat.features.utils import equals


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
    expected = {
        "home>work": (array([1]), array([2])),
        "work>home": (array([0, 1]), array([1, 1])),
    }
    result = transitions.transitions_by_act(population)
    assert equals(result, expected)


def test_transition_3s():
    population = DataFrame(
        [
            {"pid": 0, "act": "home"},
            {"pid": 0, "act": "work"},
            {"pid": 0, "act": "home"},
            {"pid": 1, "act": "home"},
            {"pid": 1, "act": "work"},
        ]
    )
    expected = {"home>work>home": (array([1]), array([1]))}
    result = transitions.transition_3s_by_act(population)
    assert equals(result, expected)
