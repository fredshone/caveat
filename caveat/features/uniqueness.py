from pandas import DataFrame


def hash_population(population: DataFrame) -> set[str]:
    act_hash = population.act.astype(str) + population.duration.astype(str)
    return set(act_hash.groupby(population.pid).apply("".join))


def internal(population: DataFrame, hashed: set[str]) -> float:
    n = population.pid.nunique()
    unique = len(hashed)
    return unique / n


def external(
    observed: DataFrame,
    obseved_hashed: set[str],
    synthetic: DataFrame,
    synthetic_hashed: set[str],
) -> float:
    n = observed.pid.nunique() + synthetic.pid.nunique()
    unique = len(obseved_hashed | synthetic_hashed)
    return unique / n
