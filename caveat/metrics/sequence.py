from typing import Optional

from pandas import DataFrame, MultiIndex, Series, concat


def collect_sequence(acts: Series) -> str:
    return ">".join(acts)


def sequence_probs(population: DataFrame) -> DataFrame:
    """
    Calculates the sequence probabilities in the given population DataFrame.

    Args:
        population (DataFrame): A DataFrame containing the population data.

    Returns:
        DataFrame: A DataFrame containing the probability of each sequence.
    """
    metrics = (
        population.groupby("pid")
        .act.apply(collect_sequence)
        .value_counts(normalize=True)
    )
    metrics = metrics.sort_values(ascending=False)
    metrics.index = MultiIndex.from_tuples(
        [("sequence prob.", acts) for acts in metrics.index]
    )
    return metrics


def report_sequence_probs(
    observed: DataFrame, ys: dict[str, DataFrame], head: Optional[int] = None
) -> DataFrame:
    """
    Generate a report of sequence probabilities for the observed and comparison DataFrames.

    Args:
        observed (DataFrame): The observed DataFrame.
        ys (dict[str, DataFrame]): A dictionary of comparison DataFrames.
        head (Optional[int], optional): The number of rows to include in the report. Defaults to None.

    Returns:
        DataFrame: A report of sequence probabilities for the observed and comparison DataFrames.
    """
    x_report = sequence_probs(observed)
    x_report.name = "observed"
    report = DataFrame(x_report)
    for name, y in ys.items():
        y_report = sequence_probs(y)
        y_report.name = name
        print(y_report)
        report = concat([report, y_report], axis=1)
        report = report.fillna(0)
        report[f"{name} delta"] = report[name] - report.observed
    if head is not None:
        report = report.head(head)
    return report
