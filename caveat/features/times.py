from typing import Optional

from matplotlib import pyplot as plt
from matplotlib.figure import Axes, Figure
from pandas import DataFrame, MultiIndex, Series


def start_times(population: DataFrame) -> list:
    return {"all start times": population.start.tolist()}


def end_times(population: DataFrame) -> list:
    return {"all end times": population.end.tolist()}


def durations(population: DataFrame) -> list:
    return {"all durations": population.duration.tolist()}


def start_times_by_act(population: DataFrame) -> dict[str, list]:
    return population.groupby("act", observed=False).start.apply(list).to_dict()


def end_times_by_act(population: DataFrame) -> dict[str, list]:
    return population.groupby("act", observed=False).end.apply(list).to_dict()


def durations_by_act(population: DataFrame) -> dict[str, list]:
    return (
        population.groupby("act", observed=False).duration.apply(list).to_dict()
    )


def zip_columns(group) -> list[list]:
    return [[s, d] for s, d in zip(group.start, group.duration)]


def start_durations_by_act(population: DataFrame) -> dict[str, list]:
    return (
        population.groupby("act", observed=False).apply(zip_columns).to_dict()
    )


def start_times_by_act_plan_seq(population: DataFrame) -> dict[str, list]:
    actseq = population.groupby("pid", as_index=False).cumcount().astype(
        str
    ) + population.act.astype(str)
    return population.groupby(actseq).start.apply(list).to_dict()


def start_times_by_act_plan_enum(population: DataFrame) -> dict[str, list]:
    actseq = population.act.astype(str) + population.groupby(
        ["pid", "act"], as_index=False, observed=False
    ).cumcount().astype(str)
    return population.groupby(actseq).start.apply(list).to_dict()


def end_times_by_act_plan_seq(population: DataFrame) -> dict[str, list]:
    actseq = population.groupby("pid", as_index=False).cumcount().astype(
        str
    ) + population.act.astype(str)
    return population.groupby(actseq).end.apply(list).to_dict()


def end_times_by_act_plan_enum(population: DataFrame) -> dict[str, list]:
    actseq = population.act.astype(str) + population.groupby(
        ["pid", "act"], as_index=False, observed=False
    ).cumcount().astype(str)
    return population.groupby(actseq).end.apply(list).to_dict()


def durations_by_act_plan_seq(population: DataFrame) -> dict[str, list]:
    actseq = population.groupby("pid", as_index=False).cumcount().astype(
        str
    ) + population.act.astype(str)
    return population.groupby(actseq).duration.apply(list).to_dict()


def durations_by_act_plan_enum(population: DataFrame) -> dict[str, list]:
    actseq = population.act.astype(str) + population.groupby(
        ["pid", "act"], as_index=False, observed=False
    ).cumcount().astype(str)
    return population.groupby(actseq).duration.apply(list).to_dict()


def average_start_times(population: DataFrame) -> Series:
    """
    Calculates the average start time for each activity type in the given population.

    Args:
        population (DataFrame): A pandas DataFrame containing the population data.

    Returns:
        Series: A pandas Series containing the average start time for each activity.
    """
    metrics = population.groupby("act", observed=False).start.mean()
    order = (
        population.groupby("act", observed=False)
        .start.count()
        .sort_values(ascending=False)
        .index
    )
    metrics = metrics[order]
    metrics.index = MultiIndex.from_tuples(
        [("average start time", act) for act in metrics.index]
    )
    return metrics


def average_start_times_plan_seq(population: DataFrame) -> Series:
    """
    Calculates the average start time for each activity indexed by plan sequence enumeration in the given population.

    Args:
        population (DataFrame): A pandas DataFrame containing the population data.

    Returns:
        Series: A pandas Series containing the average start time for each activity sequence.
    """
    actseq = population.act.astype(str) + population.groupby(
        "pid", as_index=False
    ).cumcount().astype(str)
    metrics = population.groupby(actseq).start.mean()
    order = (
        population.groupby(actseq, observed=False)
        .start.count()
        .sort_values(ascending=False)
        .index
    )
    metrics = metrics[order]
    metrics.index = MultiIndex.from_tuples(
        [("average start time plan seq", act) for act in metrics.index]
    )
    return metrics


def average_start_times_act_seq(population: DataFrame) -> Series:
    """
    Calculates the average start time for each activity indexed by enumeration in the population DataFrame.

    Args:
        population (DataFrame): A pandas DataFrame containing the population data.

    Returns:
        Series: A pandas Series containing the average start time for each activity enumeration.
    """
    actseq = population.act.astype(str) + population.groupby(
        ["pid", "act"], as_index=False, observed=False
    ).cumcount().astype(str)
    metrics = population.groupby(actseq).start.mean()
    order = (
        population.groupby(actseq, observed=False)
        .start.count()
        .sort_values(ascending=False)
        .index
    )
    metrics = metrics[order]
    metrics.index = MultiIndex.from_tuples(
        [("average start time act seq", act) for act in metrics.index]
    )
    return metrics


def average_end_times(population: DataFrame) -> Series:
    """
    Calculate the average end time for each activity in the given population.

    Args:
        population (DataFrame): A DataFrame containing the population data.

    Returns:
        Series: A Series containing the average end time for each activity, indexed by activity name.
    """
    metrics = population.groupby("act", observed=False).end.mean()
    order = (
        population.groupby("act", observed=False)
        .end.count()
        .sort_values(ascending=False)
        .index
    )
    metrics = metrics[order]
    metrics.index = MultiIndex.from_tuples(
        [("average end time", act) for act in metrics.index]
    )
    return metrics


def average_end_times_plan_seq(population: DataFrame) -> Series:
    """
    Calculates the average end time for each activity indexed by plan sequence in the given population.

    Args:
        population (DataFrame): A pandas DataFrame containing the population data.

    Returns:
        Series: A pandas Series containing average end times indexed by plan enumerated activity type.
    """
    actseq = population.act.astype(str) + population.groupby(
        "pid", as_index=False
    ).cumcount().astype(str)
    metrics = population.groupby(actseq).end.mean()
    order = (
        population.groupby(actseq, observed=False)
        .end.count()
        .sort_values(ascending=False)
        .index
    )
    metrics = metrics[order]
    metrics.index = MultiIndex.from_tuples(
        [("average end time plan seq", act) for act in metrics.index]
    )
    return metrics


def average_end_times_act_seq(population: DataFrame) -> Series:
    """
    Calculates the average end time for each activity indexed by enumeration.

    Args:
        population (DataFrame): A DataFrame containing the population data.

    Returns:
        Series: A Series containing average end times, indexed by activity sequence enumeration.
    """
    actseq = population.act.astype(str) + population.groupby(
        ["pid", "act"], as_index=False, observed=False
    ).cumcount().astype(str)
    metrics = population.groupby(actseq).end.mean()
    order = (
        population.groupby(actseq, observed=False)
        .end.count()
        .sort_values(ascending=False)
        .index
    )
    metrics = metrics[order]
    metrics.index = MultiIndex.from_tuples(
        [("average end time act seq", act) for act in metrics.index]
    )
    return metrics


def times_distributions_plot(
    observed: DataFrame, ys: Optional[dict[DataFrame]], **kwargs
) -> Figure:
    fig, axs = plt.subplots(
        3,
        observed.act.nunique(),
        figsize=kwargs.pop("figsize", (12, 5)),
        sharex=True,
        sharey=False,
        tight_layout=True,
    )
    acts = list(observed.act.value_counts(ascending=False).index)
    _times_plot("observed", observed, acts, axs=axs)
    if ys is None:
        return fig
    for name, y in ys.items():
        _times_plot(name, y, acts, axs=axs)
    return fig


def _times_plot(
    name: str,
    population: DataFrame,
    acts: list[str],
    axs: Axes,
    xmin: int = 0,
    xmax: int = 1440,
    step: int = 30,
    **kwargs,
) -> (Figure, Axes):
    starts = population.groupby("act", observed=False).start
    ends = population.groupby("act", observed=False).end
    durations = population.groupby("act", observed=False).duration
    bins = list(range(xmin, xmax, step))

    for i, act in enumerate(acts):
        if act not in population.act.unique():
            continue
        axs[0][i].set_title(act.title(), fontstyle="italic")
        axs[0][i].hist(
            starts.get_group(act),
            bins=bins,
            density=True,
            histtype="step",
            label=name,
            **kwargs,
        )
        axs[0][i].set_xlim(xmin, xmax)
        axs[0][i].set_yticklabels([])
        axs[0][i].set(ylabel=None)
        axs[0][0].legend(fontsize="x-small")

        axs[1][i].hist(
            ends.get_group(act),
            bins=bins,
            density=True,
            histtype="step",
            label=name,
            **kwargs,
        )
        axs[1][i].set_xlim(xmin, xmax)
        axs[1][i].set_yticklabels([])
        axs[1][i].set(ylabel=None)

        axs[2][i].hist(
            durations.get_group(act),
            bins=bins,
            density=True,
            histtype="step",
            label=name,
            **kwargs,
        )
        axs[2][i].set_xlim(xmin, xmax)
        axs[2][i].set_yticklabels([])
        axs[2][i].set(ylabel=None)

    axs[0][0].set(ylabel="Start times")
    axs[1][0].set(ylabel="End times")
    axs[2][0].set(ylabel="Durations")


def joint_time_distributions_plot(
    observed: DataFrame, ys: Optional[dict[DataFrame]], **kwargs
) -> Figure:
    if ys is None:
        ys = dict()
    acts = list(observed.act.value_counts(ascending=False).index)

    fig = plt.figure(
        constrained_layout=True, figsize=kwargs.pop("figsize", (15, 4))
    )

    subfigs = fig.subfigures(nrows=len(ys) + 1, ncols=1)
    # deal with observed first
    if not ys:
        subfig = subfigs
    else:
        subfig = subfigs[0]
    subfig.suptitle("Observed", fontstyle="italic")
    axs = subfig.subplots(nrows=1, ncols=len(acts), sharex=True, sharey=True)
    _joint_time_plot("obseved", observed, axs, acts)

    # act column titles
    for ax, act in zip(axs, acts):
        ax.set_title(act.title(), fontsize="large")

    # now deal with ys
    for i, (name, y) in enumerate(ys.items()):
        subfig = subfigs[i + 1]
        subfig.suptitle(name.title(), fontstyle="italic")
        axs = subfig.subplots(
            nrows=1, ncols=len(acts), sharex=True, sharey=True
        )
        _joint_time_plot(name, y, axs, acts)

    # xlabel on bottom row
    for ax in axs:
        ax.set(xlabel="Start times\n(minutes)")

    return fig


def _joint_time_plot(
    name: str,
    population: DataFrame,
    axs: Axes,
    acts: list[str],
    xmin: int = 0,
    xmax: int = 1441,
    ymin: int = 0,
    ymax: int = 800,
    xstep: int = 30,
    ystep: int = 30,
    **kwargs,
):
    starts = population.groupby("act", observed=False).start
    durations = population.groupby("act", observed=False).duration

    start_bins = list(range(xmin, xmax, xstep))
    duration_bins = list(range(ymin, ymax, ystep))

    for i, act in enumerate(acts):
        ylabel = "Durations (minutes)"
        axs[0].set_ylabel(ylabel)
        if act not in population.act.unique():
            continue
        act_starts = starts.get_group(act)
        act_durations = durations.get_group(act)
        axs[i].hist2d(
            x=act_starts,
            y=act_durations,
            bins=(start_bins, duration_bins),
            cmap="Reds",
        )
