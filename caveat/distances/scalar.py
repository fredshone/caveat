import numpy as np


def ape(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> float:
    # unpack
    ak, aw = a
    bk, bw = b
    # calc weighted average
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


def mse(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> float:
    # unpack
    ak, aw = a
    bk, bw = b
    return ((aw - bw) ** 2).mean()


def mae(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> float:
    # unpack
    ak, aw = a
    bk, bw = b
    return np.abs(aw - bw).mean()
