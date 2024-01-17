from datetime import datetime, timedelta
from typing import Optional

from matplotlib import pyplot as plt
from pandas import DataFrame

from caveat.features.frequency import activity_bins


def frequency_plots(
    observed: DataFrame, ys: Optional[dict[DataFrame]], **kwargs
):
    if ys is None:
        ys = dict()
    acts = list(observed.act.value_counts(ascending=False).index)
    class_map = {n: i for i, n in enumerate(acts)}

    fig, axs = plt.subplots(
        sharex=True,
        sharey=True,
        nrows=1,
        ncols=len(ys) + 1,
        constrained_layout=True,
        figsize=kwargs.pop("figsize", (15, 4)),
    )

    if not ys:
        ax = axs
    else:
        ax = axs[0]

    plot_agg_acts("observed", observed, class_map, ax=ax, legend=True, **kwargs)

    # now deal with ys
    for i, (name, y) in enumerate(ys.items()):
        ax = axs[i + 1]
        plot_agg_acts(name, y, class_map, ax=ax, legend=False, **kwargs)

    return fig


def plot_agg_acts(
    name: str,
    population: DataFrame,
    class_map: dict,
    duration: int = 1440,
    step: int = 10,
    ax=None,
    legend=True,
    **kwargs,
):
    bins = activity_bins(
        population, duration=duration, step=step, class_map=class_map
    )
    columns = list(class_map.keys())
    totals = bins.sum(0)
    sorted_cols = [x for _, x in sorted(zip(totals, columns))]
    df = DataFrame(bins, columns=columns)[sorted_cols]
    df.index = [
        datetime(2021, 11, 1, 0) + timedelta(minutes=i * step)
        for i in range(len(df.index))
    ]
    fig = df.plot(
        kind="bar", stacked=True, width=1, ax=ax, legend=legend, **kwargs
    )
    if legend:
        ax.legend(loc="right")
    ax = fig.axes
    labels = [" " for _ in range(len(df.index))]
    labels[:: int(120 / step)] = [x.strftime("%H:%M") for x in df.index][
        :: int(120 / step)
    ]
    ax.set_xticklabels(labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xlabel("Time of day")
    ax.set_ylabel("Activity frequency")
    ax.set_title(name.title())
    return ax
