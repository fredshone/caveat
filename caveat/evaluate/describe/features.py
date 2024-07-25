import numpy as np
from numpy import ndarray
from pandas import Series


def actual(features: dict[str, float]) -> Series:
    return Series(features)


def feature_length(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series({k: len(v) for k, (v, w) in features.items()})


def feature_weight(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series({k: w.sum() for k, (v, w) in features.items()})


def average_weight(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series({k: w.mean() for k, (v, w) in features.items()})


def average(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series(
        {
            k: np.average(v, axis=0, weights=w).sum()
            for k, (v, w) in features.items()
        }
    )


def weighted_variance(values, weights):
    """
    Return the weighted standard deviation.

    They weights are in effect first normalized so that they
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return variance


def variance(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series(
        {
            k: weighted_variance(v, weights=w).sum()
            for k, (v, w) in features.items()
        }
    )


def sd(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series(
        {
            k: np.sqrt(weighted_variance(v, weights=w)).sum()
            for k, (v, w) in features.items()
        }
    )


def average2d(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series(
        {
            k: np.average(v, axis=0, weights=w).sum().sum()
            for k, (v, w) in features.items()
            if w.sum() > 0
        }
    )
