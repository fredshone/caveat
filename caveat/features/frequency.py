from numpy import array, ndarray
from pandas import DataFrame

from caveat.encoders.descrete3d import descretise_population


def activity_bins(
    population: DataFrame, class_map: dict, duration: int = 1440, step: int = 30
) -> ndarray:
    return (
        descretise_population(
            population, duration=duration, step_size=step, class_map=class_map
        )
        .sum(0)
        .squeeze()
        .numpy()
    )


def activity_frequencies(
    population: DataFrame, duration: int = 1440, step: int = 10
) -> dict[str, tuple[ndarray, ndarray]]:
    index_to_acts = {i: a for i, a in enumerate(population.act.unique())}
    class_map = {a: i for i, a in index_to_acts.items()}
    bins = (
        descretise_population(
            population, duration=duration, step_size=step, class_map=class_map
        )
        .mean(0)
        .squeeze()
        .numpy()
    )
    support = array([i for i in range(0, duration, step)])
    return {act: (support, bins[:, i]) for act, i in class_map.items()}
