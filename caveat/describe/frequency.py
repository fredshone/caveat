from datetime import datetime, timedelta

from pandas import DataFrame

from caveat.features.frequency import activity_bins


def plot_agg_acts(
    population: DataFrame, class_map: dict, duration: int = 1440, step: int = 10
):
    bins = activity_bins(
        population, duration=duration, step_size=step, class_map=class_map
    )
    columns = list(class_map.keys())
    totals = bins.sum(0)
    sorted_cols = [x for _, x in sorted(zip(totals, columns))]
    df = DataFrame(bins, columns=columns)[sorted_cols]
    df.index = [
        datetime(2021, 11, 1, 0) + timedelta(minutes=i * step)
        for i in range(len(df.index))
    ]
    fig = df.plot(kind="bar", stacked=True, width=1)
    ax = fig.axes
    labels = [" " for _ in range(len(df.index))]
    labels[:: int(60 / step)] = [x.strftime("%H:%M") for x in df.index][
        :: int(60 / step)
    ]
    ax.set_xticklabels(labels)
