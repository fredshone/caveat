import numpy as np
from pandas import Series


def ape(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> float:
    # unpack
    ak, aw = a
    bk, bw = b
    # alc weighted average
    akw = (ak * aw).sum() / aw.sum()
    bkw = (bk * bw).sum() / bw.sum()
    diff = np.abs(akw - bkw)
    if diff == 0:
        return 0.0
    if bkw == 0:
        return np.abs(akw - bkw) / akw
    return np.abs(akw - bkw) / bkw


def mape_scalar(a, b):
    return np.abs((a - b) / a).mean()


def actual(features: dict[str, float]) -> Series:
    return Series(features)
