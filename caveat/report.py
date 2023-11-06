from typing import Callable, Optional

from pandas import DataFrame, concat


def report_diff(
    observed: DataFrame,
    ys: dict[str, DataFrame],
    feature: Callable,
    head: Optional[int] = None,
) -> DataFrame:
    x_report = feature(observed)
    x_report.name = "observed"
    report = DataFrame(x_report)
    for name, y in ys.items():
        y_report = feature(y)
        y_report.name = name
        report = concat([report, y_report], axis=1)
        report = report.fillna(0)
        report[f"{name} delta"] = report[name] - report.observed
    if head is not None:
        report = report.head(head)
    return report


def describe(
    observed: DataFrame,
    ys: dict[str, DataFrame],
    metric: Callable,
    head: Optional[int] = None,
) -> DataFrame:
    x_report = metric(observed)
    x_report.name = "observed"
    report = DataFrame(x_report)
    for name, y in ys.items():
        y_report = metric(y)
        y_report.name = name
        report = concat([report, y_report], axis=1)
        report = report.fillna(0)
    report["mean"] = report[ys.keys()].mean(axis=1)
    report["mean delta"] = report.mean - report.observed
    report["std"] = report[ys.keys()].std(axis=1)
    if head is not None:
        report = report.head(head)
    return report
