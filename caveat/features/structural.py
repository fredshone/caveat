from pandas import DataFrame, Series


def start_and_end_acts(population: DataFrame, target: str = "home") -> Series:
    """
    Calculates the proportion of individuals in the population who performed their first and last act at the specified target location.

    Args:
        population (DataFrame): A DataFrame containing the population data.
        target (str): The target location to calculate the proportion of individuals who performed their first and last act there. Defaults to "home".

    Returns:
        Series: A pandas Series containing the calculated metrics.
    """
    first = (population.groupby("pid").first().act == target).mean()
    last = (population.groupby("pid").last().act == target).mean()
    metrics = Series(
        {
            ("structural", f"first act is {target}"): first,
            ("structural", f"last act is {target}"): last,
        }
    )
    metrics.index.names = ["type", "metric"]
    return metrics


def time_consistency(population: DataFrame, target: int = 1440) -> Series:
    starts_at_zero = (population.groupby("pid").first().time == 0).mean()
    ends_at_target = (population.groupby("pid").last().time == target).mean()
    target_duration = (
        population.groupby("pid").duration.sum() == target
    ).mean()
    metrics = Series(
        {
            ("structural", "starts at 0"): starts_at_zero,
            ("structural", f"ends at {target}"): ends_at_target,
            ("structural", f"duration is {target}"): target_duration,
        }
    )
    metrics.index.names = ["type", "metric"]
    return metrics


def trip_consistency(population: DataFrame) -> Series:
    pass
