from numpy import array
from pandas import DataFrame, Series

from caveat.describe.features import average
from caveat.distances.scalar import mae
from caveat.features.times import start_times_by_act
from caveat.report import describe_feature, extract_default, score_features


def test_describe_feature():
    observed = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "work", "start": 2},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
            {"pid": 1, "act": "work", "start": 1},
        ]
    )
    # ys = {
    #     "y1": DataFrame(
    #         [
    #             {"pid": 0, "act": "home", "start": 0},
    #             {"pid": 0, "act": "work", "start": 2},
    #             {"pid": 0, "act": "home", "start": 6},
    #             {"pid": 1, "act": "home", "start": 0},
    #             {"pid": 1, "act": "work", "start": 1},
    #         ]
    #     ),
    #     "y2": DataFrame(
    #         [
    #             {"pid": 0, "act": "home", "start": 0},
    #             {"pid": 0, "act": "work", "start": 2},
    #             {"pid": 0, "act": "home", "start": 6},
    #             {"pid": 1, "act": "home", "start": 0},
    #             {"pid": 1, "act": "work", "start": 3},
    #         ]
    #     ),
    # }
    # data = {
    #     "observed": {"home": 2.0, "work": 1.5},
    #     "y1": {"home": 2.0, "work": 1.5},
    #     "y1 delta": {"home": 0.0, "work": 0.0},
    #     "y2": {"home": 2.0, "work": 2.5},
    #     "y2 delta": {"home": 0.0, "work": 1.0},
    # }
    expected = Series({"home": 2.0, "work": 1.5}, name="test")
    start_times = start_times_by_act(observed)
    result = describe_feature("test", start_times, average)
    assert result.equals(expected)


def test_create_default():
    feature = {"home": (array([0, 1, 2, 3]), array([10, 0, 2, 3]))}
    default = extract_default(feature)
    assert (default[0] == array([0])).all()
    assert (default[1] == array([1])).all()
    feature = {"home": (array([[0, 0], [10, 10]]), array([10, 3]))}
    default = extract_default(feature)
    assert (default[0] == array([[0, 0]])).all()
    assert (default[1] == array([1])).all()


def test_score_features():
    observed = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "work", "start": 2},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
            {"pid": 1, "act": "work", "start": 1},
        ]
    )
    y = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "work", "start": 2},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
            {"pid": 1, "act": "work", "start": 1},
        ]
    )
    expected = Series({"home": 0.0, "work": 0.0}, name="test").sort_index()
    x = start_times_by_act(observed)
    y = start_times_by_act(y)
    result = score_features(
        "test", x, y, mae, (array([0]), array([1]))
    ).sort_index()
    assert result.equals(expected)


def test_score_features_with_default():
    observed = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "work", "start": 1},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
            {"pid": 1, "act": "work", "start": 1},
        ]
    )
    y = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
        ]
    )
    expected = Series({"home": 0.0, "work": 1.0}, name="test").sort_index()
    x = start_times_by_act(observed)
    y = start_times_by_act(y)
    result = score_features(
        "test", x, y, mae, (array([0]), array([1]))
    ).sort_index()
    assert result.equals(expected)
