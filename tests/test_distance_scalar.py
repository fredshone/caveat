from numpy import array

from caveat.distances import scalar


def test_ape_binary():
    a = (array([0, 1]), array([0, 1]))
    assert scalar.ape(a, a) == 0
    b = (array([0, 1]), array([0, 100]))
    assert scalar.ape(a, b) == 0
    b = (array([0, 1]), array([1, 0]))
    assert scalar.ape(a, b) == 1
    b = (array([0, 1]), array([0.5, 0.5]))
    assert scalar.ape(a, b) == 1
    b = (array([0, 1]), array([5, 5]))
    assert scalar.ape(a, b) == 1
    # todo


def test_mse():
    a = (array([0, 1]), array([0, 1]))
    assert scalar.mse(a, a) == 0
    b = (array([0, 1]), array([1, 0]))
    assert scalar.mse(a, b) == 1
    # todo
