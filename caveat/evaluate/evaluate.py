from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from pandas import DataFrame, MultiIndex, Series, concat

from caveat.evaluate.describe.features import (
    average,
    average2d,
    average_density,
    feature_value,
    feature_weight,
)
from caveat.evaluate.distance import emd
from caveat.evaluate.features import (
    creativity,
    frequency,
    participation,
    structural,
    times,
    transitions,
)

sample_quality_jobs = [
    (
        ("duration", structural.duration_consistency),
        (feature_weight),
        ("duration", average),
        ("EMD", emd),
    ),
    (
        ("home based", structural.start_and_end_acts),
        (feature_weight),
        ("prob.", average),
        ("EMD", emd),
    ),
]
count_jobs = [
    (
        ("total schedules", frequency.count_schedules),
        (feature_value),
        ("count", feature_value),
        ("EMD", emd),
    )
]
aggregate_jobs = [
    (
        ("agg. frequency", frequency.activity_frequencies),
        (feature_weight),
        ("average freq.", average_density),
        ("EMD", emd),
    )
]
# participation_prob_jobs = [

#     (
#         ("participation", participation.participation_prob_by_act),
#         (feature_weight),
#         ("prob.", average),
#         ("MAPE", mape),
#     ),
#     (
#         ("joint participation", participation.joint_participation_prob),
#         (feature_weight),
#         ("prob.", average),
#         ("MAPE", mape),
#     ),
# ]
participation_rate_jobs = [
    (
        ("lengths", structural.sequence_lengths),
        (feature_weight),
        ("length.", average),
        ("EMD", emd),
    ),
    (
        ("participation rate", participation.participation_rates_by_act_enum),
        (feature_weight),
        ("av. rate", average),
        ("EMD", emd),
    ),
    (
        ("pair participation rate", participation.joint_participation_rate),
        (feature_weight),
        ("av rate.", average),
        ("EMD", emd),
    ),
]
transition_jobs = [
    (
        ("2-gram", transitions.transitions_by_act),
        (feature_weight),
        ("av. rate", average),
        ("EMD", emd),
    ),
    (
        ("3-gram", transitions.transition_3s_by_act),
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


def evaluate_subsampled(
    synthetic_schedules: dict[str, DataFrame],
    synthetic_attributes: dict[str, DataFrame],
    target_schedules: DataFrame,
    target_attributes: DataFrame,
    split_on: List[str],
    report_stats: bool = True,
    verbose: bool = False,
):
    descriptions = []
    distances = []
    for split in split_on:
        target_cats = target_attributes[split].unique()
        for cat in target_cats:
            target_pids = target_attributes[target_attributes[split] == cat].pid
            sub_target = target_schedules[
                target_schedules.pid.isin(target_pids)
            ]
            sub_schedules = {}
            for model, attributes in synthetic_attributes.items():
                sample_pids = attributes[attributes[split] == cat].pid
                if verbose:
                    print(
                        f">>> Subsampled {model} {split}={cat} with {len(sample_pids)}"
                    )
                sample_schedules = synthetic_schedules[model]
                sub_schedules[model] = sample_schedules[
                    sample_schedules.pid.isin(sample_pids)
                ]

            sub_reports = process(
                synthetic_schedules=sub_schedules, target_schedules=sub_target
            )
            for r in sub_reports:  # add sub pop to index
                names = list(r.index.names) + ["sub_pop"]
                r.index = MultiIndex.from_tuples(
                    [(*i, f"{split}={cat}") for i in r.index], names=names
                )
            descriptions.append(sub_reports[0])
            distances.append(sub_reports[1])

    descriptions = concat(descriptions, axis=0)
    distances = concat(distances, axis=0)

    frames = describe_splits(descriptions, distances)

    if report_stats:
        columns = list(synthetic_schedules.keys())
        for frame in frames.values():
            add_stats(data=frame, columns=columns)

    return frames


def evaluate(
    synthetic_schedules: dict[str, DataFrame],
    target_schedules: DataFrame,
    report_stats: bool = True,
):
    descriptions, distances = process(synthetic_schedules, target_schedules)
    frames = describe(descriptions, distances)

    if report_stats:
        columns = list(synthetic_schedules.keys())
        for frame in frames.values():
            add_stats(data=frame, columns=columns)

    return frames


def process(
    synthetic_schedules: dict[str, DataFrame], target_schedules: DataFrame
) -> Tuple[DataFrame, DataFrame]:
    # evaluate creativity
    descriptions, distances = eval_models_creativity(
        synthetic_schedules=synthetic_schedules,
        target_schedules=target_schedules,
    )

    for domain, jobs in [
        ("count", count_jobs),
        ("sample quality", sample_quality_jobs),
        ("aggregate", aggregate_jobs),
        # ("participation_probs", participation_prob_jobs),
        ("participations", participation_rate_jobs),
        ("transitions", transition_jobs),
        ("timing", time_jobs),
    ]:
        for feature, size, description_job, distance_job in jobs:
            feature_descriptions, feature_distances = (
                eval_models_density_estimation(
                    synthetic_schedules,
                    target_schedules,
                    domain,
                    feature,
                    size,
                    description_job,
                    distance_job,
                )
            )
            descriptions = concat([descriptions, feature_descriptions], axis=0)
            distances = concat([distances, feature_distances], axis=0)

    # remove nans
    descriptions = descriptions.fillna(0.0)
    distances = distances.fillna(0.0)
    return descriptions, distances


def describe(
    descriptions: DataFrame, distances: DataFrame
) -> dict[str, DataFrame]:
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
        .apply(distance_weighted_av)
    )
    features_distances["distance"] = (
        distances["distance"].groupby(["domain", "feature"]).first()
    )

    # domains
    domain_descriptions = (
        features_descriptions.drop("description", axis=1)
        .groupby("domain")
        .mean()
    )
    domain_distances = (
        features_distances.drop("distance", axis=1).groupby("domain").mean()
    )

    frames = {
        "descriptions": descriptions,
        "feature_descriptions": features_descriptions,
        "domain_descriptions": domain_descriptions,
        "distances": distances,
        "feature_distances": features_distances,
        "domain_distances": domain_distances,
    }
    return frames


def describe_splits(
    descriptions: DataFrame, distances: DataFrame
) -> dict[str, DataFrame]:
    # features
    features_descriptions = (
        descriptions.drop("description", axis=1)
        .groupby(["domain", "feature", "sub_pop"])
        .apply(weighted_av)
    )
    features_descriptions["description"] = (
        descriptions["description"]
        .groupby(["domain", "feature", "sub_pop"])
        .first()
    )

    features_distances = (
        distances.drop("distance", axis=1)
        .groupby(["domain", "feature", "sub_pop"])
        .apply(distance_weighted_av)
    )
    features_distances["distance"] = (
        distances["distance"].groupby(["domain", "feature", "sub_pop"]).first()
    )

    # themes
    domain_descriptions = (
        features_descriptions.drop("description", axis=1)
        .groupby("domain")
        .mean()
    )
    domain_distances = (
        features_distances.drop("distance", axis=1).groupby("domain").mean()
    )

    frames = {
        "descriptions": descriptions,
        "feature_descriptions": features_descriptions,
        "domain_descriptions": domain_descriptions,
        "distances": distances,
        "feature_distances": features_distances,
        "domain_distances": domain_distances,
    }
    return frames


def eval_models_creativity(
    synthetic_schedules: dict[str, DataFrame], target_schedules: DataFrame
) -> Tuple[DataFrame, DataFrame]:
    # Evaluate Creativity
    observed_hash = creativity.hash_population(target_schedules)
    observed_diversity = creativity.diversity(target_schedules, observed_hash)
    feature_count = target_schedules.pid.nunique()
    creativity_descriptions = DataFrame(
        {
            "observed__weight": [feature_count] * 2,
            "observed": [observed_diversity, 1],
        }
    )
    creativity_distance = DataFrame(
        {
            "observed__weight": [feature_count] * 2,
            "observed": [1 - observed_diversity, 0],
        }
    )

    creativity_descs = []
    creativity_dists = []
    for model, y in synthetic_schedules.items():
        y_hash = creativity.hash_population(y)
        y_diversity = creativity.diversity(y, y_hash)
        y_count = y.pid.nunique()
        creativity_descs.append(
            Series(
                [y_diversity, creativity.novelty(observed_hash, y_hash)],
                name=model,
            )
        )
        creativity_descs.append(  # add feature count
            Series([y_count, y_count], name=f"{model}__weight")
        )
        creativity_dists.append(
            Series(
                [
                    1 - y_diversity,
                    creativity.conservatism(observed_hash, y_hash),
                ],
                name=model,
            )
        )
        creativity_dists.append(  # add feature count
            Series([y_count, y_count], name=f"{model}__weight")
        )

    creativity_descs.append(
        Series(["prob. unique", "prob. novel"], name="description")
    )
    creativity_dists.append(
        Series(["prob. not unique", "prob. conservative"], name="distance")
    )
    # combine
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
            ("creativity", "homogeneity", "all"),
            ("creativity", "conservatism", "all"),
        ],
        names=["domain", "feature", "segment"],
    )
    return descriptions, distances


def eval_models_density_estimation(
    synthetic_schedules: dict[str, DataFrame],
    target_schedules: DataFrame,
    domain: str,
    feature: Tuple[str, Callable],
    size: Callable,
    description_job: Tuple[str, Callable],
    distance_job: Tuple[str, Callable],
) -> Tuple[DataFrame, DataFrame]:
    # unpack tuples
    feature_name, feature = feature
    description_name, describe = description_job
    distance_name, distance_metric = distance_job

    # build observed features
    observed_features = feature(target_schedules)

    # need to create a default feature for missing sampled features
    default = extract_default(observed_features)

    # create an observed feature count and description
    feature_weight = size(observed_features)
    feature_weight.name = "observed__weight"

    description_job = describe(observed_features)
    feature_descriptions = DataFrame(
        {"observed__weight": feature_weight, "observed": description_job}
    )

    # sort by count and description, drop description and add distance description
    feature_descriptions = feature_descriptions.sort_values(
        ascending=False, by=["observed__weight", "observed"]
    )

    feature_distances = feature_descriptions.copy()

    # iterate through samples
    for model, y in synthetic_schedules.items():
        synth_features = feature(y)
        synth_weight = size(synth_features)
        synth_weight.name = f"{model}__weight"

        feature_descriptions = concat(
            [
                synth_weight,
                feature_descriptions,
                describe_feature(model, synth_features, describe),
            ],
            axis=1,
        )

        # report sampled distances
        feature_distances = concat(
            [
                synth_weight,
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

    return feature_descriptions, feature_distances


def rank(data: DataFrame) -> DataFrame:
    # feature rank
    rank = data.drop(["observed", "distance"], axis=1, errors="ignore").rank(
        axis=1, method="min"
    )
    col_ranks = rank.sum(axis=0)
    ranked = [i for _, i in sorted(zip(col_ranks, col_ranks.index))]
    return rank[ranked]


def report(
    frames: dict[str, DataFrame],
    log_dir: Optional[Path] = None,
    head: Optional[int] = None,
    verbose: bool = True,
    suffix: str = "",
    ranking: bool = False,
):
    if head is not None:
        frames["descriptions_short"] = (
            frames["descriptions"].groupby(["domain", "feature"]).head(head)
        )
        frames["distances_short"] = (
            frames["distances"].groupby(["domain", "feature"]).head(head)
        )
    else:
        # default to full
        frames["descriptions_short"] = frames["descriptions"]
        frames["distances_short"] = frames["distances"]

    if log_dir is not None:
        for name, frame in frames.items():
            frame.to_csv(Path(log_dir, f"{name}{suffix}.csv"))

    if verbose:
        print("\nDescriptions:")
        print_markdown(frames["descriptions_short"])
        print("\nEvalutions (Distance):")
        print_markdown(frames["distances_short"])

    print("\nFeature Descriptions:")
    print_markdown(frames["feature_descriptions"])
    print("\nFeature Evaluations (Distance):")
    print_markdown(frames["feature_distances"])
    if ranking:
        print("\nFeature Evaluations (Ranked):")
        print_markdown(rank(frames["feature_distances"]))

    print("\nDomain Descriptions:")
    print_markdown(frames["domain_descriptions"])
    print("\nDomain Evaluations (Distance):")
    print_markdown(frames["domain_distances"])
    if ranking:
        print("\nDomain Evaluations (Ranked):")
        print_markdown(rank(frames["domain_distances"]))


def report_splits(
    frames: dict[str, DataFrame],
    log_dir: Optional[Path] = None,
    head: Optional[int] = None,
    verbose: bool = True,
    suffix: str = "",
    ranking: bool = False,
):
    if head is not None:
        frames["descriptions_short"] = (
            frames["descriptions"]
            .groupby(["domain", "feature", "sub_pop"])
            .head(head)
        )
        frames["distances_short"] = (
            frames["distances"]
            .groupby(["domain", "feature", "sub_pop"])
            .head(head)
        )
    else:
        # default to full
        frames["descriptions_short"] = frames["descriptions"]
        frames["distances_short"] = frames["distances"]

    if log_dir is not None:
        for name, frame in frames.items():
            frame.to_csv(Path(log_dir, f"{name}{suffix}.csv"))

    if verbose:
        print("\nDescriptions:")
        print_markdown(frames["descriptions_short"])
        print("\nEvalutions (Distance):")
        print_markdown(frames["distances_short"])

    print("\nFeature Descriptions:")
    print_markdown(frames["feature_descriptions"])
    print("\nFeature Evaluations (Distance):")
    print_markdown(frames["feature_distances"])
    if ranking:
        print("\nFeature Evaluations (Ranked):")
        print_markdown(rank(frames["feature_distances"]))

    print("\nDomain Descriptions:")
    print_markdown(frames["domain_descriptions"])
    print("\nDomain Evaluations (Distance):")
    print_markdown(frames["domain_distances"])
    if ranking:
        print("\nDomain Evaluations (Ranked):")
        print_markdown(rank(frames["domain_distances"]))


def add_stats(data: DataFrame, columns: dict[str, DataFrame]):
    data["mean"] = data[columns].mean(axis=1)
    data["std"] = data[columns].std(axis=1)


def print_markdown(data: DataFrame):
    print(data.to_markdown(tablefmt="fancy_grid", floatfmt=".3f"))


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
        {
            k: distance(
                defaulting_get(a, k, default), defaulting_get(b, k, default)
            )
            for k in index
        },
        name=model,
    )
    metrics = metrics.fillna(0)
    metrics = metrics[np.isfinite(metrics)]
    return metrics


def defaulting_get(
    features: dict[str, tuple[np.array, np.array]],
    key: str,
    default: tuple[np.array, np.array],
):
    feature = features.get(key)
    if feature is None:
        return default
    support, weights = feature
    if len(support) == 0:
        return default
    return feature


def extract_default(features: dict[str, tuple[np.array, np.array]]):
    # we use a single feature of zeros as required
    # look for a zize
    default_shape = extract_default_shape(features)
    default_support = np.zeros(default_shape)
    return (default_support, np.array([1]))


def extract_default_shape(
    features: dict[str, tuple[np.array, np.array]]
) -> np.array:
    for k, _ in iter(features.values()):
        if len(k) > 0:
            default_shape = list(k.shape)
            default_shape[0] = 1
            return default_shape
    raise ValueError("No features found in the given dictionary.")


def weighted_av(report: DataFrame, weight_col: str = "__weight") -> Series:
    """Weighted avergae of dataframe using weights in the weight column."""
    cols = list(report.columns)
    cols = [c for c in cols if not c.endswith(weight_col)]
    scores = DataFrame()
    for c in cols:
        weights = report[f"{c}{weight_col}"]
        total = weights.sum()
        scores[c] = report[c] * weights / total
    return scores.sum()


def distance_weighted_av(
    report: DataFrame,
    base_col: str = "observed__weight",
    weight_col: str = "__weight",
) -> Series:
    """Weighted avergae of dataframe using weights in the weight column and a base column.
    This deals with cases where models have different features.
    """
    cols = list(report.columns)
    cols = [c for c in cols if not c.endswith(weight_col)]
    base_weights = report[base_col]
    scores = DataFrame()
    for c in cols:
        weights = report[f"{c}{weight_col}"]
        weights = (weights + base_weights) / 2
        total = weights.sum()
        scores[c] = report[c] * weights / total
    return scores.sum()
