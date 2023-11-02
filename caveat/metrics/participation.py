from typing import Optional

from pandas import DataFrame, MultiIndex, Series


def participation_rates(population: DataFrame) -> Series:
    """
    Calculates the participation rates for each activity in the given population DataFrame.

    Args:
        population (DataFrame): A DataFrame containing the population data.

    Returns:
        Series: A Series containing the participation rates for each activity.
    """
    metrics = (
        population.groupby("act", observed=False).pid.count()
        / population.pid.nunique()
    )
    metrics.index = MultiIndex.from_tuples(
        [("participation rate", act) for act in metrics.index]
    )
    return metrics


def report_participation_rates(
    observed: DataFrame, ys: dict[str, DataFrame], head: Optional[int] = None
) -> DataFrame:
    """
    Generate a report of participation rates for observed and comparison groups.

    Args:
        observed (DataFrame): A DataFrame containing the observed participation rates.
        ys (dict[str, DataFrame]): A dictionary of DataFrames containing the comparison group participation rates.
        head (Optional[int], optional): The number of rows to include in the report. Defaults to None.

    Returns:
        DataFrame: A DataFrame containing the participation rates for the observed and comparison groups, as well as the delta between them.
    """
    x_report = participation_rates(observed)
    x_report.name = "observed"
    report = DataFrame(x_report)
    for name, y in ys.items():
        y_report = participation_rates(y)
        report[name] = y_report
        report = report.fillna(0)
        report[f"{name} delta"] = report[name] - report.observed
    if head is not None:
        report = report.head(head)
    return report


def combinations_with_replacement(
    targets: list, length: int, prev_array=[]
) -> list[list]:
    """
    Returns all possible combinations of elements in the input array with replacement,
    where each combination has a length of tuple_length.

    Args:
        targets (list): The input array to generate combinations from.
        length (int): The length of each combination.
        prev_array (list, optional): The previous array generated in the recursion. Defaults to [].

    Returns:
        list: A list of all possible combinations of elements in the input array with replacement.
    """
    if len(prev_array) == length:
        return [prev_array]
    combs = []
    for i, val in enumerate(targets):
        prev_array_extended = prev_array.copy()
        prev_array_extended.append(val)
        combs += combinations_with_replacement(
            targets[i:], length, prev_array_extended
        )
    return combs


def calc_pair_rate(act_counts: DataFrame, pair: tuple) -> float:
    """
    Calculates the participation rate for given activity pairs given activity counts.

    Parameters:
    act_counts (DataFrame): DataFrame of activity counts.
    pair (tuple): Pair of activities to calculate participation rate for.

    Returns:
    float: Participation rate of the pair of users.
    """
    a, b = pair
    if a == b:
        return (act_counts[a] > 1).mean()
    return ((act_counts[a] > 0) & (act_counts[b] > 0)).mean()


def participation_pairs(population: DataFrame) -> Series:
    """
    Calculate the participation rate for all pairs of activities in the given population.

    Args:
        population (pandas.DataFrame): A DataFrame containing the population data.

    Returns:
        pandas.Series: A Series containing the participation rate for all pairs of activities.
    """
    act_counts = (
        population.groupby("pid").act.value_counts().unstack(fill_value=0)
    )
    pairs = combinations_with_replacement(list(population.act.unique()), 2)
    idx = ["+".join(pair) for pair in pairs]
    p = [calc_pair_rate(act_counts, pair) for pair in pairs]
    report = Series(p, index=idx)
    report = report.sort_values(ascending=False)
    report.index = MultiIndex.from_tuples(
        [("participation rate", pair) for pair in report.index]
    )
    return report


def report_participation_pairs(
    observed: DataFrame, ys: dict[str, DataFrame], head: Optional[int] = None
) -> DataFrame:
    """
    Generate a report of participation pairs for the observed and comparison DataFrames.

    Args:
        observed (DataFrame): The observed DataFrame.
        ys (dict[str, DataFrame]): A dictionary of comparison DataFrames.
        head (Optional[int], optional): The number of rows to include in the report. Defaults to None.

    Returns:
        DataFrame: A report of participation pairs for the observed and comparison DataFrames.
    """
    x_report = participation_pairs(observed)
    x_report.name = "observed"
    report = DataFrame(x_report)
    for name, y in ys.items():
        y_report = participation_pairs(y)
        report[name] = y_report
        report = report.fillna(0)
        report[f"{name} delta"] = report[name] - report.observed
    if head is not None:
        report = report.head(head)
    return report
