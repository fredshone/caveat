from pathlib import Path
from typing import Callable, Optional

import numpy as np
from pandas import DataFrame, MultiIndex, Series, concat

from caveat.describe.features import (
    average,
    average2d,
    average_weight,
    feature_length,
    feature_weight,
)
from caveat.distances import ape, emd, mae
from caveat.features import (
    creativity,
    frequency,
    participation,
    structural,
    times,
    transitions,
)

structure_jobs = [
    (
        ("time consistency", structural.time_consistency),
        (feature_weight),
        ("prob. consistent", average),
        ("MAE", mae),
    ),
    (
        ("start and end acts", structural.start_and_end_acts),
        (feature_weight),
        ("prob.", average),
        ("MAE", mae),
    ),
]
frequency_jobs = [
    (
        ("agg. participation", frequency.activity_frequencies),
        (feature_length),
        ("average freq.", average_weight),
        ("MAE", mae),
    )
]
participation_prob_jobs = [
    (
        ("participation", participation.participation_prob_by_act),
        (feature_weight),
        ("prob.", average),
        ("MAPE", ape),
    ),
    (
        ("joint participation", participation.joint_participation_prob),
        (feature_weight),
        ("prob.", average),
        ("MAPE", ape),
    ),
]
participation_rate_jobs = [
    (
        ("participation rate", participation.participation_rates),
        (feature_weight),
        ("av. rate", average),
        ("EMD", emd),
    ),
    (
        (
            "activity participation rates",
            participation.participation_rates_by_act,
        ),
        (feature_weight),
        ("av. rate", average),
        ("EMD", emd),
    ),
    (
        (
            "enumerated participation rates",
            participation.participation_rates_by_act_enum,
        ),
        (feature_weight),
        ("av. rate", average),
        ("EMD", emd),
    ),
]
transition_jobs = [
    (
        ("transition pairs", transitions.transitions_by_act),
        (feature_weight),
        ("av. rate", average),
        ("EMD", emd),
    ),
    (
        ("transition 3s", transitions.transition_3s_by_act),
        (feature_weight),
        ("av. rate", average),
        ("EMD", emd),
    ),
    # (
    #     ("sequences", transitions.full_sequences),
    #     ("mean", average),
    #     ("EMD", emd),
    # ),
]
time_jobs = [
    (
        ("start times", times.start_times_by_act),
        (feature_weight),
        ("average", average),
        ("EMD", emd),
    ),
    (
        ("end times", times.end_times_by_act),
        (feature_weight),
        ("average", average),
        ("EMD", emd),
    ),
    (
        ("durations", times.durations_by_act),
        (feature_weight),
        ("average", average),
        ("EMD", emd),
    ),
    (
        ("start-durations", times.start_and_duration_by_act_bins),
        (feature_weight),
        ("average", average2d),
        ("EMD", emd),
    ),
]


def report(
    observed: DataFrame,
    sampled: dict[str, DataFrame],
    log_dir: Optional[Path] = None,
    report_description: bool = True,
    report_scores: bool = True,
    head: int = 10,
    verbose: bool = True,
    report_creativity: bool = True,
):
    descriptions = DataFrame()
    scores = DataFrame()

    if report_creativity:
        observed_hash = creativity.hash_population(observed)
        uniqueness_descriptions = DataFrame(
            {
                "feature count": [observed.pid.nunique()],
                "observed": [creativity.diversity(observed, observed_hash)],
            }
        )
        uniqueness_descriptions.index = MultiIndex.from_tuples(
            [("creativity", "novelty", "all")],
            names=["domain", "feature", "segment"],
        )
        uniqueness_scores = uniqueness_descriptions.copy()

        for model, y in sampled.items():
            y_hash = creativity.hash_population(y)
            uniqueness_descriptions[model] = creativity.diversity(y, y_hash)
            uniqueness_scores[model] = 1 - creativity.novelty(
                observed_hash, y, y_hash
            )
        uniqueness_descriptions["description"] = "prob. unique"
        uniqueness_scores["distance"] = "prob. novel"
        descriptions = concat([descriptions, uniqueness_descriptions], axis=0)
        scores = concat([scores, uniqueness_scores], axis=0)

    for domain, jobs in [
        ("structure", structure_jobs),
        ("frequency", frequency_jobs),
        ("participation_prob_jobs", participation_prob_jobs),
        ("participation_rate_jobs", participation_rate_jobs),
        ("transitions", transition_jobs),
        ("scheduling", time_jobs),
    ]:
        for feature, size, description, distance in jobs:
            # unpack tuples
            feature_name, feature = feature
            description_name, describe = description
            distance_name, distance = distance

            # build observed features
            observed_features = feature(observed)

            # need to create a default feature for missing sampled features
            default = extract_default(observed_features)

            # create an observed feature count and description
            feature_weight = size(observed_features)
            feature_weight.name = "feature count"

            description = describe(observed_features)
            feature_descriptions = DataFrame(
                {"feature count": feature_weight, "observed": description}
            )

            # sort by count and description, drop description and add distance description
            feature_descriptions = feature_descriptions.sort_values(
                ascending=False, by=["feature count", "observed"]
            )

            feature_scores = feature_descriptions.copy()

            # iterate through samples
            for model, y in sampled.items():
                synth_features = feature(y)
                feature_descriptions = concat(
                    [
                        feature_descriptions,
                        describe_feature(model, synth_features, describe),
                    ],
                    axis=1,
                )

                # report sampled distances
                feature_scores = concat(
                    [
                        feature_scores,
                        score_features(
                            model,
                            observed_features,
                            synth_features,
                            distance,
                            default,
                        ),
                    ],
                    axis=1,
                )

            # add domain and feature name to index
            feature_descriptions["description"] = description_name
            feature_scores["distance"] = distance_name
            feature_descriptions.index = MultiIndex.from_tuples(
                [(domain, feature_name, f) for f in feature_descriptions.index],
                name=["domain", "feature", "segment"],
            )
            feature_scores.index = MultiIndex.from_tuples(
                [(domain, feature_name, f) for f in feature_scores.index],
                name=["domain", "feature", "segment"],
            )
            descriptions = concat([descriptions, feature_descriptions], axis=0)
            scores = concat([scores, feature_scores], axis=0)

    # remove nans
    descriptions = descriptions.fillna(0)
    scores = scores.fillna(0)

    # features
    features_descriptions = (
        descriptions.drop("description", axis=1)
        .groupby(["domain", "feature"])
        .apply(weighted_av)
    )
    features_descriptions["description"] = (
        descriptions["description"].groupby(["domain", "feature"]).first()
    )

    features_scores = (
        scores.drop("distance", axis=1)
        .groupby(["domain", "feature"])
        .apply(weighted_av)
    )
    features_scores["distance"] = (
        scores["distance"].groupby(["domain", "feature"]).first()
    )

    # rank
    feature_ranks = features_scores.drop(["observed", "distance"], axis=1).rank(
        axis=1, method="min"
    )
    col_ranks = feature_ranks.sum(axis=0)
    ranked = [i for _, i in sorted(zip(col_ranks, col_ranks.index))]
    feature_ranks = feature_ranks[ranked]

    # themes
    domain_descriptions = (
        features_descriptions.drop("description", axis=1)
        .groupby("domain")
        .mean()
    )
    domain_scores = (
        features_scores.drop("distance", axis=1).groupby("domain").mean()
    )

    # rank
    domain_ranks = domain_scores.drop("observed", axis=1).rank(
        axis=1, method="min"
    )
    col_ranks = domain_ranks.sum(axis=0)
    ranked = [i for _, i in sorted(zip(col_ranks, col_ranks.index))]
    domain_ranks = domain_ranks[ranked]

    if log_dir is not None:
        descriptions.to_csv(Path(log_dir, "descriptions.csv"))
        features_descriptions.to_csv(Path(log_dir, "feature_descriptions.csv"))
        domain_descriptions.to_csv(Path(log_dir, "domain_descriptions.csv"))
        scores.to_csv(Path(log_dir, "scores.csv"))
        features_scores.to_csv(Path(log_dir, "feature_scores.csv"))
        domain_scores.to_csv(Path(log_dir, "domain_scores.csv"))

    if head is not None:
        descriptions_short = descriptions.groupby(["domain", "feature"]).head(
            head
        )
        descriptions_short.to_csv(Path(log_dir, "descriptions_short.csv"))
        scores_short = scores.groupby(["domain", "feature"]).head(head)
        scores_short.to_csv(Path(log_dir, "scores_short.csv"))
    else:
        scores_short = scores

    if verbose:
        print("\nDescriptions:")
        print(
            descriptions_short.to_markdown(
                tablefmt="fancy_grid", floatfmt=".3f"
            )
        )
        print("\nScores:")
        print(scores_short.to_markdown(tablefmt="fancy_grid", floatfmt=".3f"))
    print("\nFeature Descriptions:")
    print(
        features_descriptions.to_markdown(tablefmt="fancy_grid", floatfmt=".3f")
    )
    print("\nFeature Scores:")
    print(features_scores.to_markdown(tablefmt="fancy_grid", floatfmt=".3f"))
    print("\nFeature Ranks:")
    print(feature_ranks.to_markdown(tablefmt="fancy_grid"))
    print("\nDomain Descriptions:")
    print(
        domain_descriptions.to_markdown(tablefmt="fancy_grid", floatfmt=".3f")
    )
    print("\nDomain Scores:")
    print(domain_scores.to_markdown(tablefmt="fancy_grid", floatfmt=".3f"))
    print("\nDomain Ranks:")
    print(domain_ranks.to_markdown(tablefmt="fancy_grid"))


def describe_feature(
    model: str,
    features: dict[str, tuple[np.array, np.array]],
    describe: Callable,
):
    feature_description = describe(features)
    feature_description.name = model
    return feature_description


def score_features(
    model: str,
    a: dict[str, tuple[np.array, np.array]],
    b: dict[str, tuple[np.array, np.array]],
    distance: Callable,
    default: tuple[np.array, np.array],
):
    index = set(a.keys()) | set(b.keys())
    metrics = Series(
        {k: distance(a.get(k, default), b.get(k, default)) for k in index},
        name=model,
    )
    metrics = metrics.fillna(0)
    metrics = metrics[np.isfinite(metrics)]
    return metrics


def extract_default(features: dict[str, tuple[np.array, np.array]]):
    # we use a single feature of zeros as required
    default_sample = next(iter(features.values()))
    default_shape = list(default_sample[0].shape)
    default_shape[0] = 1
    default_support = np.zeros(default_shape)
    return (default_support, np.array([1]))


def weighted_av(report: DataFrame, weight_col: str = "feature count") -> Series:
    weights = report[weight_col]
    total = sum(weights)
    cols = list(report.columns)
    cols.remove(weight_col)
    scores = DataFrame()
    for c in cols:
        scores[c] = report[c] * weights / total
    return scores.sum()


# def report_diff(
#     observed: DataFrame,
#     ys: dict[str, DataFrame],
#     feature: Callable,
#     head: Optional[int] = None,
# ) -> DataFrame:
#     x_report = feature(observed)
#     x_report.name = "observed"
#     report = DataFrame(x_report)
#     for name, y in ys.items():
#         y_report = feature(y)
#         y_report.name = name
#         report = concat([report, y_report], axis=1)
#         report = report.fillna(0)
#         report[f"{name} delta"] = report[name] - report.observed
#     if head is not None:
#         report = report.head(head)
#     return report


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
