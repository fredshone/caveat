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
from caveat.distance import abs_av_diff, emd, mae, mape
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
        ("duration", structural.duration_consistency),
        (feature_weight),
        ("duration", average),
        ("MAE", abs_av_diff),
    ),
    (
        ("home based", structural.start_and_end_acts),
        (feature_weight),
        ("prob.", average),
        ("MAE", abs_av_diff),
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
        ("MAPE", mape),
    ),
    (
        ("joint participation", participation.joint_participation_prob),
        (feature_weight),
        ("prob.", average),
        ("MAPE", mape),
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
        ("start times", times.start_times_by_act_plan_enum),
        (feature_weight),
        ("average", average),
        ("EMD", emd),
    ),
    (
        ("end times", times.end_times_by_act_plan_enum),
        (feature_weight),
        ("average", average),
        ("EMD", emd),
    ),
    (
        ("durations", times.durations_by_act_plan_enum),
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
    head: int = 10,
    verbose: bool = True,
    report_stats: bool = False,
):
    descriptions = DataFrame()
    distances = DataFrame()

    # Evaluate Creativity
    observed_hash = creativity.hash_population(observed)
    observed_diversity = creativity.diversity(observed, observed_hash)
    creativity_descriptions = DataFrame(
        {
            "feature count": [observed.pid.nunique()] * 2,
            "observed": [observed_diversity, 1],
        }
    )
    creativity_distance = DataFrame(
        {
            "feature count": [observed.pid.nunique()] * 2,
            "observed": [observed_diversity, 0],
        }
    )

    creativity_descs = []
    creativity_dists = []
    for model, y in sampled.items():
        y_hash = creativity.hash_population(y)
        y_diversity = creativity.diversity(y, y_hash)
        creativity_descs.append(
            Series(
                [y_diversity, creativity.novelty(observed_hash, y_hash)],
                name=model,
            )
        )
        creativity_dists.append(
            Series(
                [
                    abs(y_diversity - observed_diversity),
                    creativity.conservatism(observed_hash, y_hash),
                ],
                name=model,
            )
        )
    creativity_descs.append(
        Series(["prob. unique", "prob. novel"], name="description")
    )
    creativity_dists.append(
        Series(["abs error", "prob. conservative"], name="distance")
    )

    descriptions = concat(
        [creativity_descriptions, concat(creativity_descs, axis=1)], axis=1
    )
    distances = concat(
        [creativity_distance, concat(creativity_dists, axis=1)], axis=1
    )
    descriptions.index = MultiIndex.from_tuples(
        [("creativity", "diversity", "all"), ("creativity", "novelty", "all")],
        names=["domain", "feature", "segment"],
    )
    distances.index = MultiIndex.from_tuples(
        [
            ("creativity", "diversity", "all"),
            ("creativity", "conservatism", "all"),
        ],
        names=["domain", "feature", "segment"],
    )

    # Evaluate Correctness
    for domain, jobs in [
        ("structure", structure_jobs),
        ("frequency", frequency_jobs),
        ("participation_probs", participation_prob_jobs),
        ("participation_rates", participation_rate_jobs),
        ("transitions", transition_jobs),
        ("timing", time_jobs),
    ]:
        for feature, size, description_job, distance_job in jobs:
            # unpack tuples
            feature_name, feature = feature
            print(f"Calculating {feature_name}...")
            description_name, describe = description_job
            distance_name, distance_metric = distance_job

            # build observed features
            observed_features = feature(observed)

            # need to create a default feature for missing sampled features
            default = extract_default(observed_features)

            # create an observed feature count and description
            feature_weight = size(observed_features)
            feature_weight.name = "feature count"

            description_job = describe(observed_features)
            feature_descriptions = DataFrame(
                {"feature count": feature_weight, "observed": description_job}
            )

            # sort by count and description, drop description and add distance description
            feature_descriptions = feature_descriptions.sort_values(
                ascending=False, by=["feature count", "observed"]
            )

            feature_distances = feature_descriptions.copy()

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
                feature_distances = concat(
                    [
                        feature_distances,
                        score_features(
                            model,
                            observed_features,
                            synth_features,
                            distance_metric,
                            default,
                        ),
                    ],
                    axis=1,
                )

            # add domain and feature name to index
            feature_descriptions["description"] = description_name
            feature_distances["distance"] = distance_name
            feature_descriptions.index = MultiIndex.from_tuples(
                [(domain, feature_name, f) for f in feature_descriptions.index],
                name=["domain", "feature", "segment"],
            )
            feature_distances.index = MultiIndex.from_tuples(
                [(domain, feature_name, f) for f in feature_distances.index],
                name=["domain", "feature", "segment"],
            )
            descriptions = concat([descriptions, feature_descriptions], axis=0)
            distances = concat([distances, feature_distances], axis=0)

    # remove nans
    descriptions = descriptions.fillna(0)
    distances = distances.fillna(0)

    # features
    features_descriptions = (
        descriptions.drop("description", axis=1)
        .groupby(["domain", "feature"])
        .apply(weighted_av)
    )
    features_descriptions["description"] = (
        descriptions["description"].groupby(["domain", "feature"]).first()
    )

    features_distances = (
        distances.drop("distance", axis=1)
        .groupby(["domain", "feature"])
        .apply(weighted_av)
    )
    features_distances["distance"] = (
        distances["distance"].groupby(["domain", "feature"]).first()
    )

    # rank
    feature_ranks = features_distances.drop(
        ["observed", "distance"], axis=1
    ).rank(axis=1, method="min")
    col_ranks = feature_ranks.sum(axis=0)
    ranked = [i for _, i in sorted(zip(col_ranks, col_ranks.index))]
    feature_ranks = feature_ranks[ranked]

    # themes
    domain_descriptions = (
        features_descriptions.drop("description", axis=1)
        .groupby("domain")
        .mean()
    )
    domain_distances = (
        features_distances.drop("distance", axis=1).groupby("domain").mean()
    )

    # rank
    domain_ranks = domain_distances.drop("observed", axis=1).rank(
        axis=1, method="min"
    )
    col_ranks = domain_ranks.sum(axis=0)
    ranked = [i for _, i in sorted(zip(col_ranks, col_ranks.index))]
    domain_ranks = domain_ranks[ranked]

    if report_stats:
        select = list(sampled.keys())
        descriptions["mean"] = descriptions[select].mean(axis=1)
        descriptions["std"] = descriptions[select].std(axis=1)
        features_descriptions["mean"] = features_descriptions[select].mean(
            axis=1
        )
        features_descriptions["std"] = features_descriptions[select].std(axis=1)
        domain_descriptions["mean"] = domain_descriptions[select].mean(axis=1)
        domain_descriptions["std"] = domain_descriptions[select].std(axis=1)
        distances["mean"] = distances[select].mean(axis=1)
        distances["std"] = distances[select].std(axis=1)
        features_distances["mean"] = features_distances[select].mean(axis=1)
        features_distances["std"] = features_distances[select].std(axis=1)
        domain_distances["mean"] = domain_distances[select].mean(axis=1)
        domain_distances["std"] = domain_distances[select].std(axis=1)

    if log_dir is not None:
        descriptions.to_csv(Path(log_dir, "descriptions.csv"))
        features_descriptions.to_csv(Path(log_dir, "feature_descriptions.csv"))
        domain_descriptions.to_csv(Path(log_dir, "domain_descriptions.csv"))
        distances.to_csv(Path(log_dir, "evaluation.csv"))
        features_distances.to_csv(Path(log_dir, "feature_evaluation.csv"))
        domain_distances.to_csv(Path(log_dir, "domain_evaluation.csv"))

    if head is not None:
        descriptions_short = descriptions.groupby(["domain", "feature"]).head(
            head
        )
        distances_short = distances.groupby(["domain", "feature"]).head(head)
        if log_dir is not None:
            descriptions_short.to_csv(Path(log_dir, "descriptions_short.csv"))
            distances_short.to_csv(Path(log_dir, "evaluation_short.csv"))
    else:
        descriptions_short = descriptions
        distances_short = distances

    if verbose:
        print("\nDescriptions:")
        print(
            descriptions_short.to_markdown(
                tablefmt="fancy_grid", floatfmt=".3f"
            )
        )
        print("\nEvalutions (Distance):")
        print(
            distances_short.to_markdown(tablefmt="fancy_grid", floatfmt=".3f")
        )
    print("\nFeature Descriptions:")
    print(
        features_descriptions.to_markdown(tablefmt="fancy_grid", floatfmt=".3f")
    )
    print("\nFeature Evaluations (Distance):")
    print(features_distances.to_markdown(tablefmt="fancy_grid", floatfmt=".3f"))
    print("\nFeature Evaluations (Ranked):")
    print(feature_ranks.to_markdown(tablefmt="fancy_grid"))
    print("\nDomain Descriptions:")
    print(
        domain_descriptions.to_markdown(tablefmt="fancy_grid", floatfmt=".3f")
    )
    print("\nDomain Evaluations (Distance):")
    print(domain_distances.to_markdown(tablefmt="fancy_grid", floatfmt=".3f"))
    print("\nDomain Evaluations (Ranked):")
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
