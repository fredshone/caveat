from pandas import Series
from pandas.testing import assert_series_equal
from numpy import array

from caveat.evaluate.describe import features

def test_describe_actual():
    d = {
        "a": 0,
        "b": 1,
    }
    expected = Series(d)
    assert_series_equal(features.actual(d), expected, check_dtype=False)


def test_feature_length():
    d = {
        "a": (array([0,1]), array([10,10])),
        "b": (array([0,1,2]), array([10,5,3]))
    }
    expected = Series({
        "a": 2, "b": 3
    })
    assert_series_equal(features.feature_length(d), expected, check_dtype=False)


def test_feature_weight():
    d = {
        "a": (array([0,1]), array([10,10])),
        "b": (array([0,1,2]), array([10,5,3]))
    }
    expected = Series({
        "a": 20, "b": 18
    })
    assert_series_equal(features.feature_weight(d), expected, check_dtype=False)


def test_average_weight():
    d = {
        "a": (array([0,1]), array([10,10])),
        "b": (array([0,1,2]), array([10,5,3]))
    }
    expected = Series({
        "a": 10, "b": 6
    })
    assert_series_equal(features.average_weight(d), expected, check_dtype=False)


def test_average():
    d = {
        "a": (array([0,1]), array([10,10])),
        "b": (array([0,1,2]), array([2,2,2]))
    }
    expected = Series({
        "a": 0.5, "b": 1
    })
    assert_series_equal(features.average(d), expected, check_dtype=False)


def test_average2d():
    d = {
        "a": (array([[0,0],[1,1]]), array([10,10])),
        "b": (array([[0,0],[0,1],[0,2]]), array([2,2,2]))
    }
    expected = Series({
        "a": 1, "b": 1
    })
    assert_series_equal(features.average2d(d), expected, check_dtype=False)