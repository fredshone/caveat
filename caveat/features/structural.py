from numpy import array, ndarray
from pandas import DataFrame


def start_and_end_acts(
    population: DataFrame, target: str = "home"
) -> dict[str, tuple[ndarray, ndarray]]:
    n = len(population.pid.unique())
    first = (population.groupby("pid").first().act == target).sum()
    last = (population.groupby("pid").last().act == target).sum()
    return {
        f"first act {target}": (array([0, 1]), array([(n - first), first])),
        f"last act {target}": (array([0, 1]), array([(n - last), last])),
    }


def time_consistency(
    population: DataFrame, target: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    n = len(population.pid.unique())
    starts = (population.groupby("pid").first().start == 0).sum()
    ends = (population.groupby("pid").last().end == target).sum()
    duration = (population.groupby("pid").duration.sum() == target).sum()
    return {
        "starts at 0": (array([0, 1]), array([(n - starts), starts])),
        f"ends at {target}": (array([0, 1]), array([(n - ends), ends])),
        f"duration is {target}": (
            array([0, 1]),
            array([(n - duration), duration]),
        ),
    }


def trip_consisteny(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    raise NotImplementedError
