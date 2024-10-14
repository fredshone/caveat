from numpy import arange, ndarray
from pandas import DataFrame

from caveat.encoding.one_hot import descretise_population


def binned_activity_count(
    population: DataFrame, class_map: dict, duration: int = 1440, step: int = 15
) -> ndarray:
    return (
        descretise_population(
            population, duration=duration, step_size=step, class_map=class_map
        )
        .sum(0)
        .numpy()
    )[0, :, :]


def binned_activity_density(
    population: DataFrame, class_map: dict, duration: int = 1440, step: int = 15
) -> ndarray:
    return (
        descretise_population(
            population, duration=duration, step_size=step, class_map=class_map
        )
        .mean(0)
        .numpy()
    )[0, :, :]


def activity_frequencies(
    population: DataFrame, duration: int = 1440, step: int = 15
) -> dict[str, tuple[ndarray, ndarray]]:
    index_to_acts = {i: a for i, a in enumerate(population.act.unique())}
    class_map = {a: i for i, a in index_to_acts.items()}
    bins = binned_activity_count(
        population=population, class_map=class_map, duration=duration, step=step
    )
    support = arange(0, 1, step / duration)
    freqs = {act: (support, bins[:, i]) for act, i in class_map.items()}
    return freqs


def activity_densities(
    population: DataFrame, duration: int = 1440, step: int = 15
) -> dict[str, tuple[ndarray, ndarray]]:
    index_to_acts = {i: a for i, a in enumerate(population.act.unique())}
    class_map = {a: i for i, a in index_to_acts.items()}
    bins = binned_activity_density(
        population=population, class_map=class_map, duration=duration, step=step
    )
    support = arange(0, 1, step / duration)
    return {act: (support, bins[:, i]) for act, i in class_map.items()}
