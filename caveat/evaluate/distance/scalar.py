import numpy as np


def mape(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> float:
    """Calculate mean average percentage error between distributions a and b.

    Clipped at 1.0.

    Args:
        a (tuple[np.ndarray, np.ndarray]): Distribution a.
        b (tuple[np.ndarray, np.ndarray]): Distribution b.

    Returns:
        float: MAPE.
    """
    # TODO test this
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
        return clamp(diff / akw)
    return clamp(diff / bkw)


def clamp(x):
    if x > 1.0:
        return 1.0
    return x


def mape_scalar(a, b):
    return np.abs((a - b) / a).mean()


def mse(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> float:
    # requires and b have same support.
    # unpack
    _, aw = a
    _, bw = b
    return ((aw - bw) ** 2).mean()


def mae(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> float:
    # TODO test this
    # requires and b have same support.
    # unpack
    _, aw = a
    _, bw = b
    return (np.abs(aw - bw)).mean()


def abs_av_diff(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> float:
    # TODO test this
    # unpack
    ak, aw = a
    bk, bw = b
    a_average = (ak * aw).sum() / aw.sum()
    b_average = (bk * bw).sum() / bw.sum()

    return np.abs(a_average - b_average)
