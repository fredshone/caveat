from pandas import DataFrame, MultiIndex, Series


def average_activity_durations(population: DataFrame) -> Series:
    """
    Calculate the average duration for each activity type in the population.

    Args:
        population (pandas.DataFrame): A DataFrame containing the population data.

    Returns:
        pandas.Series: A Series containing the average duration for each activity.
    """
    metric = population.groupby("act", observed=False).duration.mean()
    order = (
        population.groupby("act", observed=False)
        .duration.count()
        .sort_values(ascending=False)
        .index
    )
    metric = metric[order]
    metric.index = MultiIndex.from_tuples(
        [("average duration", act) for act in metric.index]
    )
    return metric


def average_activity_plan_seq_durations(population: DataFrame) -> Series:
    """
    Calculate the average duration for each activity (indexed by sequence enumeration) in the population.

    Args:
        population (pandas.DataFrame): A DataFrame containing the population data.

    Returns:
        pandas.Series: A Series containing the average duration for each activity enumeration.
    """
    actseq = population.act.astype(str) + population.groupby(
        "pid", as_index=False
    ).cumcount().astype(str)

    metric = population.groupby(actseq).duration.mean()
    order = (
        population.groupby(actseq, observed=False)
        .duration.count()
        .sort_values(ascending=False)
        .index
    )
    metric = metric[order]
    metric.index = MultiIndex.from_tuples(
        [("average act plan seq duration", act) for act in metric.index]
    )
    return metric


def average_activity_seq_durations(population: DataFrame) -> Series:
    """
    Calculate the average duration for each activity (indexed by enumeration) in the population.

    Args:
        population (pandas.DataFrame): A DataFrame containing the population data.

    Returns:
        pandas.Series: A Series containing the average duration for each activity enumeration.
    """
    actseq = population.act.astype(str) + population.groupby(
        ["pid", "act"], as_index=False, observed=False
    ).cumcount().astype(str)

    metric = population.groupby(actseq).duration.mean()
    order = (
        population.groupby(actseq, observed=False)
        .duration.count()
        .sort_values(ascending=False)
        .index
    )
    metric = metric[order]
    metric.index = MultiIndex.from_tuples(
        [("average act seq duration", act) for act in metric.index]
    )
    return metric
