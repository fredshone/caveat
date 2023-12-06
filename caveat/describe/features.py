import numpy as np
from numpy import ndarray
from pandas import Series


def actual(features: dict[str, float]) -> Series:
    return Series(features)


def feature_length(features: dict[str, ndarray]) -> Series:
    return Series({k: len(v) for k, (v, w) in features.items()})


def feature_weight(features: dict[str, ndarray]) -> Series:
    return Series({k: w.sum() for k, (v, w) in features.items()})


def average_weight(features: dict[str, ndarray]) -> Series:
    return Series({k: w.mean() for k, (v, w) in features.items()})


def average(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series(
        {
            k: np.average(v, axis=0, weights=w).sum()
            for k, (v, w) in features.items()
        }
    )


def average2d(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series(
        {
            k: np.average(v, axis=0, weights=w).sum().sum()
            for k, (v, w) in features.items()
        }
    )
