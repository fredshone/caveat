from pathlib import Path
from typing import Callable, Optional

from pandas import DataFrame, MultiIndex, Series, concat, set_option
from scipy.stats import wasserstein_distance

from caveat import features
from caveat.metrics.wasserstein import sinkhorn


def score_synthesis(
    observed: DataFrame,
    sampled: dict[str, DataFrame],
    log_dir: Optional[Path] = None,
    n: int = 10,
    verbose: bool = True,
):
    participation_jobs = [
        (
            "participation rates",
            features.participation.raw_participation_rates_by_act,
            wasserstein_distance,
        ),
        (
            "enumerated participation rates",
            features.participation.raw_participation_rates_by_act_enum,
            wasserstein_distance,
        ),
    ]
    transition_jobs = [
        (
            "transition rates",
            features.transitions.raw_transition_rates,
            wasserstein_distance,
        )
    ]
    time_jobs = [
        (
            "start times",
            features.times.start_times_by_act,
            wasserstein_distance,
        ),
        ("end times", features.times.end_times_by_act, wasserstein_distance),
        ("durations", features.times.durations_by_act, wasserstein_distance),
        ("start-durations", features.times.start_durations_by_act, sinkhorn),
    ]
    reports = DataFrame()
    display_reports = DataFrame()

    for theme, jobs in [
        ("participation", participation_jobs),
        ("transitions", transition_jobs),
        ("scheduling", time_jobs),
    ]:
        for name, feature, distance in jobs:
            report = report_distances(
                observed, sampled, name, feature, distance, head=n
            )
            report.index = MultiIndex.from_tuples(
                [(theme, a, b) for (a, b) in report.index]
            )
            report.index.names = ["theme", "metric", "activity"]

            reports = concat([reports, report], axis=0)
            if n is not None:
                report = report.head(n)
            display_reports = concat([display_reports, report], axis=0)

    metric_scores = reports.groupby(["theme", "metric"]).apply(weighted_av)
    theme_scores = reports.groupby("theme").apply(weighted_av)

    if log_dir is not None:
        reports.to_csv(Path(log_dir, "report.csv"))
        metric_scores.to_csv(Path(log_dir, "metric_scores.csv"))
        theme_scores.to_csv(Path(log_dir, "theme_scores.csv"))

    set_option("display.precision", 2)
    if verbose:
        print(display_reports.to_markdown())
    print(metric_scores.to_markdown())
    print(theme_scores.to_markdown())

    return reports, metric_scores, theme_scores


def report_scalar_distance(
    observed: DataFrame,
    ys: dict[str, DataFrame],
    name: str,
    feature: Callable,
    distance: Callable,
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
        report[f"{name} score"] = report[name] - report.observed
    if head is not None:
        report = report.head(head)
    return report


def report_distances(
    observed: DataFrame,
    ys: dict[str, DataFrame],
    name: str,
    feature: Callable,
    distance: Callable,
    head: Optional[int] = None,
) -> DataFrame:
    x_features = feature(observed)
    counts = Series({k: len(v) for k, v in x_features.items()})
    counts.name = "feature count"
    counts.sort_values(ascending=False, inplace=True)
    report = DataFrame(counts)

    for model, y in ys.items():
        features = feature(y)
        col = Series(
            {
                k: distance(x_features[k], features.get(k, [0]))
                for k in report.index
            }
        )
        col.name = model
        report = concat([report, col], axis=1)

    report.index = MultiIndex.from_tuples([(name, f) for f in report.index])

    if head is not None:
        report = report.head(head)
    return report


def av(report: DataFrame, ignore_col: str = "feature count") -> Series:
    cols = list(report.columns)
    cols.remove(ignore_col)
    score = report[cols].mean(axis=0)
    score.name = report.index.get_level_values(0)[0]
    return score


def weighted_av(report: DataFrame, weight_col: str = "feature count") -> Series:
    weights = report[weight_col]
    total = sum(weights)
    cols = list(report.columns)
    cols.remove(weight_col)
    scores = DataFrame()
    for c in cols:
        scores[c] = report[c] * weights / total
    return scores.sum()


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
    report["mean"] = report[ys.keys()].mean(axis=1)
    report["mean delta"] = report["mean"] - report.observed
    report["std"] = report[ys.keys()].std(axis=1)
    if head is not None:
        report = report.head(head)
    return report
