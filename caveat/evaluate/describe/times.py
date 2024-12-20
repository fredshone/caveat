from typing import Optional, Tuple

from matplotlib import colormaps, patches
from matplotlib import pyplot as plt
from matplotlib.figure import Axes, Figure
from pandas import DataFrame


def times_distributions_plot(
    observed: DataFrame, ys: Optional[dict[str, DataFrame]], **kwargs
) -> Figure:
    ratios = [1 for _ in range(4)]
    ratios[0] = 0.2
    fig, axs = plt.subplots(
        4,
        observed.act.nunique(),
        figsize=kwargs.pop("figsize", (12, 5)),
        sharex=True,
        sharey=False,
        # tight_layout=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": ratios},
    )
    acts = list(observed.act.value_counts(ascending=False).index)
    name = kwargs.pop("observed_title", "Observed")
    _times_plot(name, observed, acts, axs=axs)
    if ys is None:
        return fig
    for name, y in ys.items():
        _times_plot(name, y, acts, axs=axs)
    for ax in axs[0]:
        ax.tick_params(axis="x", which="both", length=0.0)
    handles, labels = axs[1][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        fontsize=9,
        ncol=len(ys) + 1,
        frameon=False,
    )
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
) -> Tuple[Figure, Axes]:
    starts = population.groupby("act", observed=False).start
    ends = population.groupby("act", observed=False).end
    durations = population.groupby("act", observed=False).duration
    bins = list(range(xmin, xmax, step))

    for i, act in enumerate(acts):
        if act not in population.act.unique():
            continue
        axs[0][i].spines["top"].set_visible(False)
        axs[0][i].spines["right"].set_visible(False)
        axs[0][i].spines["bottom"].set_visible(False)
        axs[0][i].spines["left"].set_visible(False)
        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])

        axs[1][i].set_title(act.title(), fontsize="small")
        start_group = starts.get_group(act)
        if len(start_group) < 5:
            continue
        axs[1][i].hist(
            starts.get_group(act),
            bins=bins,
            density=True,
            histtype="step",
            label=name,
            linewidth=1.4,
            **kwargs,
        )
        axs[1][i].set_xlim(xmin, xmax)
        axs[1][i].set_yticklabels([])
        axs[1][i].set(ylabel=None)
        axs[1][i].set_xticks([])
        axs[1][i].set_yticks([])

        axs[2][i].hist(
            ends.get_group(act),
            bins=bins,
            density=True,
            histtype="step",
            label=name,
            linewidth=1.4,
            **kwargs,
        )
        axs[2][i].set_xlim(xmin, xmax)
        axs[2][i].set_yticklabels([])
        axs[2][i].set(ylabel=None)
        axs[2][i].set_xticks([])
        axs[2][i].set_yticks([])

        axs[3][i].hist(
            durations.get_group(act),
            bins=bins,
            density=True,
            histtype="step",
            label=name,
            linewidth=1.4,
            **kwargs,
        )
        axs[3][i].set_xlim(xmin, xmax)
        axs[3][i].set_yticklabels([])
        axs[3][i].set(ylabel=None)
        axs[3][i].set_xticks(
            [0, 240, 480, 720, 960, 1200, 1440],
            labels=[
                "00:00",
                "04:00",
                "08:00",
                "12:00",
                "16:00",
                "20:00",
                "24:00",
            ],
            rotation=90,
            fontsize=8,
        )
        axs[3][i].set_yticks([])
        axs[3][i].set_xlabel("Time/Duration", fontsize=8)

    axs[1][0].set_ylabel("Start Time\nDensities", fontsize=10)
    axs[2][0].set_ylabel("End Time\nDensities", fontsize=10)
    axs[3][0].set_ylabel("Duration\nDensities", fontsize=10)


def joint_time_distributions_plot(
    observed: DataFrame, ys: Optional[dict[DataFrame]], **kwargs
) -> Figure:
    if ys is None:
        ys = dict()
    acts = list(observed.act.value_counts(ascending=False).index)
    n_acts = len(acts)
    rows = len(ys) + 2
    ratios = [1 for _ in range(rows)]
    ratios[0] = 0.2

    cmaps = kwargs.pop("cmaps", {})
    legend = []
    legend_colours = []

    fig, axs = plt.subplots(
        rows,
        observed.act.nunique(),
        figsize=kwargs.pop("figsize", (12, 5)),
        sharex=False,
        sharey=False,
        constrained_layout=True,
        gridspec_kw={"height_ratios": ratios},
    )

    # deal with observed first
    name = kwargs.pop("observed_title", "Observed")

    legend.append(name)
    cmap = cmaps.get(0, "Blues")
    lcolours = colormaps[cmap]([0, 0.5, 1])
    legend_colours.append(lcolours[int(len(lcolours) / 2)])

    _joint_time_plot(observed, axs[1], acts, cmap=cmap)

    # now deal with ys
    for i, (name, y) in enumerate(ys.items()):

        legend.append(name)
        cmap = cmaps.get(i + 1, "Reds")
        lcolours = colormaps[cmap]([0, 0.5, 1])
        legend_colours.append(lcolours[int(len(lcolours) / 2)])

        _joint_time_plot(y, axs[i + 2], acts, cmap=cmap)

    # xlabel on bottom row
    for ax in axs[-1]:
        ax.set_xlabel("Start times", fontsize=8)
        ax.set_xticks(
            [240, 480, 720, 960, 1200, 1440],
            labels=["04:00", "08:00", "12:00", "16:00", "20:00", "24:00"],
            rotation=90,
            fontsize=8,
        )

    # acts
    for ax, act in zip(axs[1], acts):
        ax.set_title(act.title(), fontsize=9)

    # deal with legend
    for i in range(n_acts):
        axs[0][i].spines["top"].set_visible(False)
        axs[0][i].spines["right"].set_visible(False)
        axs[0][i].spines["bottom"].set_visible(False)
        axs[0][i].spines["left"].set_visible(False)
        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
    for ax in axs[0]:
        ax.tick_params(axis="x", which="both", length=0.0)

    handles = [
        patches.Patch(color=c, label=l) for c, l in zip(legend_colours, legend)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        fontsize=9,
        ncol=len(ys) + 1,
        frameon=False,
    )

    return fig


def _joint_time_plot(
    population: DataFrame,
    axs: Axes,
    acts: list[str],
    cmap: str,
    xmin: int = 240,
    xmax: int = 1441,
    ymin: int = 0,
    ymax: int = 960,
    xstep: int = 30,
    ystep: int = 30,
):
    starts = population.groupby("act", observed=False).start
    durations = population.groupby("act", observed=False).duration

    start_bins = list(range(xmin, xmax, xstep))
    duration_bins = list(range(ymin, ymax, ystep))

    for i, act in enumerate(acts):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        if act not in population.act.unique():
            continue
        act_starts = starts.get_group(act)
        act_durations = durations.get_group(act)
        axs[i].hist2d(
            x=act_starts,
            y=act_durations,
            bins=(start_bins, duration_bins),
            cmap=cmap,
        )
    ylabel = "Durations"
    axs[0].set_ylabel(ylabel, fontsize=8)
    axs[0].set_yticks(
        [0, 120, 240, 360, 480, 600, 720, 840, 960],
        labels=[
            "00:00",
            "02:00",
            "04:00",
            "06:00",
            "08:00",
            "10:00",
            "12:00",
            "14:00",
            "16:00",
        ],
        fontsize=8,
    )
