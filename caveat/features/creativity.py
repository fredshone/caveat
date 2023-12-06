from pandas import DataFrame


def hash_population(population: DataFrame) -> set[str]:
    """Hash a population of sequences. We first create strings of combined activities and durations.
    Then create a python set of these strings. This will remove duplicates.

    Args:
        population (DataFrame): Input population of sequences.

    Returns:
        set[str]: set of hashed sequences.
    """
    act_hash = population.act.astype(str) + population.duration.astype(str)
    return set(act_hash.groupby(population.pid).apply("".join))


def diversity(population: DataFrame, hashed: set[str]) -> float:
    """Measure the internal diversity of a population of sequences. This is the ratio of unique
    sequences to the total number of sequences.

    Args:
        population (DataFrame): Input population of sequences.
        hashed (set[str]): Hashed population of sequences.

    Returns:
        float: Diversity of the population.
    """
    n = population.pid.nunique()
    unique = len(hashed)
    return unique / n


def novelty(
    observed_hashed: set[str], synthetic: DataFrame, synthetic_hashed: set[str]
) -> float:
    """Measure the novelty of a population by comparing it to an observed population.

    Args:
        observed_hashed (set[str]): Hashed observed population.
        synthetic (DataFrame): Synthetic population.
        synthetic_hashed (set[str]): Hashed synthetic population.

    Returns:
        float: Novelty of the synthetic population.
    """
    unique = len(synthetic_hashed - observed_hashed)
    return unique / synthetic.pid.nunique()
