from numpy import array, ndarray
from pandas import DataFrame

from caveat.features.utils import weighted_features


def start_times_by_act(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    return weighted_features(
        population.groupby("act", observed=False).start.apply(list).to_dict()
    )


def end_times_by_act(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    return weighted_features(
        population.groupby("act", observed=False).end.apply(list).to_dict()
    )


def durations_by_act(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    return weighted_features(
        population.groupby("act", observed=False).duration.apply(list).to_dict()
    )


def zip_columns(group) -> ndarray:
    return array([(s, d) for s, d in zip(group.start, group.duration)])


def start_durations_by_act(population: DataFrame) -> dict[str, ndarray]:
    sds = population.groupby("act", observed=False).apply(zip_columns).to_dict()
    return sds


def start_and_duration_by_act_bins(
    population: DataFrame, bin_size: int = 15
) -> dict[str, tuple[ndarray, ndarray]]:
    return weighted_features(
        start_durations_by_act(population), bin_size=bin_size
    )


def start_times_by_act_plan_seq(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    actseq = population.groupby("pid", as_index=False).cumcount().astype(
        str
    ) + population.act.astype(str)
    return weighted_features(
        population.groupby(actseq).start.apply(list).to_dict()
    )


def start_times_by_act_plan_enum(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    actseq = population.act.astype(str) + population.groupby(
        ["pid", "act"], as_index=False, observed=False
    ).cumcount().astype(str)
    return weighted_features(
        population.groupby(actseq).start.apply(list).to_dict()
    )


def end_times_by_act_plan_seq(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    actseq = population.groupby("pid", as_index=False).cumcount().astype(
        str
    ) + population.act.astype(str)
    return weighted_features(
        population.groupby(actseq).end.apply(list).to_dict()
    )


def end_times_by_act_plan_enum(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    actseq = population.act.astype(str) + population.groupby(
        ["pid", "act"], as_index=False, observed=False
    ).cumcount().astype(str)
    return weighted_features(
        population.groupby(actseq).end.apply(list).to_dict()
    )


def durations_by_act_plan_seq(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    actseq = population.groupby("pid", as_index=False).cumcount().astype(
        str
    ) + population.act.astype(str)
    return weighted_features(
        population.groupby(actseq).duration.apply(list).to_dict()
    )


def durations_by_act_plan_enum(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    actseq = population.act.astype(str) + population.groupby(
        ["pid", "act"], as_index=False, observed=False
    ).cumcount().astype(str)
    return weighted_features(
        population.groupby(actseq).duration.apply(list).to_dict()
    )
