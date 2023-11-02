from typing import Optional

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
            ("structural", f"first act at {target}"): first,
            ("structural", f"last act at {target}"): last,
        }
    )
    metrics.index.names = ["type", "metric"]
    return metrics


def report_activity_start_and_end_acts(
    observed: DataFrame,
    ys: dict[str, DataFrame],
    target: str = "home",
    head: Optional[int] = None,
) -> DataFrame:
    """
    Generate a report of the start and end activities for the observed and other datasets.

    Args:
        observed (DataFrame): The observed dataset.
        ys (dict[str, DataFrame]): A dictionary of other datasets to compare against.
        target (str, optional): The target activity to report on. Defaults to "home".
        head (Optional[int], optional): The number of rows to include in the report. Defaults to None.

    Returns:
        DataFrame: A report of the start and end activities for the observed and other datasets.
    """
    x_report = start_and_end_acts(observed, target)
    x_report.name = "observed"
    report = DataFrame(x_report)
    for name, y in ys.items():
        y_report = start_and_end_acts(y, target)
        report[name] = y_report
        report[f"{name} delta"] = report[name] - report.observed
    if head is not None:
        report = report.head(head)
    return report
