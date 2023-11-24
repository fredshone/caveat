from typing import Optional

import numpy as np
from numpy import array, ndarray, unique
from pandas import Series


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


def bin_values(values: array, bin_size: int) -> ndarray:
    """
    Bins the input values based on the given bin size.

    Args:
        values (array): Input values to be binned.
        bin_size (int): Size of each bin.

    Returns:
        array: Binned values.
    """
    return (values // bin_size * bin_size) + (bin_size / 2)


def compress_feature(
    feature: list, bin_size: Optional[int] = None
) -> tuple[ndarray, ndarray]:
    """
    Compresses a feature by optionally binning its values and returning unique values with counts.

    Args:
        feature (list): The feature to compress.
        bin_size (int, optional): The size of each bin. If None, no binning is performed.

    Returns:
        tuple: A tuple containing two arrays and the total weight. The first array contains the unique
            values, and the second  array contains the counts of each value.
    """
    s = array(feature)
    if bin_size is not None:
        s = bin_values(s, bin_size)
    ks, ws = unique(s, axis=0, return_counts=True)
    return ks, ws


def weighted_features(
    features: dict[str, ndarray], bin_size: Optional[int] = None
) -> dict[str, tuple[ndarray, ndarray]]:
    """
    Apply optional binning and value counting to dictionary of features.

    Args:
        features (dict[array): A dictionary of features to compress.
        bin_size (Optional[int]): The size of the bin to use for compression. Defaults to None.

    Returns:
        dict[str, tuple[array, array[int]]]: A dictionary of features and weights.
    """
    return {
        k: compress_feature(values, bin_size) for k, values in features.items()
    }
