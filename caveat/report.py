import matplotlib.pyplot as plt
from pandas import DataFrame, Series, concat, cut


# ================ Sequence Metrics ================== #
def collect_sequence(acts: Series) -> str:
    return "-".join(acts)


def sequence_frequencies(x: DataFrame, y: DataFrame) -> DataFrame:
    report = DataFrame()
    report["observed"] = (
        x.groupby("pid")
        .act.apply(collect_sequence)
        .value_counts(normalize=True)
    )
    report["synth"] = (
        y.groupby("pid")
        .act.apply(collect_sequence)
        .value_counts(normalize=True)
    )
    report = report.fillna(0)

    report["metric"] = "activity sequence rates"
    report = report.set_index(["metric", "act"])

    report["delta"] = report.synth - report.observed
    report["perc"] = report.delta / report.observed

    # print((report.delta ** 2).mean())
    return report.sort_values("observed", ascending=False).head(5)


def transitions(x: DataFrame) -> Series:
    transitions = {}
    for acts in x.groupby("pid").act.apply(list):
        for i in range(len(acts) - 1):
            t = "-".join(acts[i : i + 2])
            transitions[t] = transitions.get(t, 0) + 1
    transitions = Series(transitions).sort_values(ascending=False)
    transitions /= transitions.sum()
    return transitions


# ================ Scheduling Metrics ================== #
def av_activity_durations(x: DataFrame, y: DataFrame) -> DataFrame:
    report = DataFrame()
    report["observed"] = x.groupby("act", observed=False).duration.mean()
    report["synth"] = y.groupby("act", observed=False).duration.mean()

    report = report.reset_index()
    report["metric"] = "av. activity durations"
    report = report.set_index(["metric", "act"])

    report["delta"] = report.synth - report.observed
    report["perc"] = report.delta / report.observed
    return report


def average_activity_starts(plans: DataFrame) -> Series:
    plans = plans[plans.start != 0]  # ignore first act
    return plans.groupby("act", observed=False).start.mean()


def average_starts(plans: DataFrame) -> int:
    plans = plans[plans.start != 0]
    return plans.start.mean()


def av_activity_start_times(x: DataFrame, y: DataFrame) -> DataFrame:
    report = DataFrame()
    report["observed"] = average_activity_starts(x)
    report["synth"] = average_activity_starts(y)
    totals = DataFrame(
        [
            {
                "act": "all",
                "observed": average_starts(x),
                "synth": average_starts(y),
            }
        ]
    ).set_index("act")
    report = concat([totals, report])

    report = report.reset_index()
    report["metric"] = "av. activity starts"
    report = report.set_index(["metric", "act"])

    report["delta"] = report.synth - report.observed
    report["perc"] = report.delta / x.end.max()  # norm by day duration
    return report


def average_activity_ends(plans: DataFrame) -> Series:
    end = plans.end.max()
    plans = plans[plans.end != end]  # ignore last act
    return plans.groupby("act", observed=False).end.mean()


def average_ends(plans: DataFrame) -> int:
    end = plans.end.max()
    plans = plans[plans.end != end]  # ignore last act
    return plans.end.mean()


def av_activity_end_times(x: DataFrame, y: DataFrame) -> DataFrame:
    report = DataFrame()
    report["observed"] = average_activity_ends(x)
    report["synth"] = average_activity_ends(y)

    totals = DataFrame(
        [{"act": "all", "observed": average_ends(x), "synth": average_ends(y)}]
    ).set_index("act")
    report = concat([totals, report])

    report = report.reset_index()
    report["metric"] = "av. activity ends"
    report = report.set_index(["metric", "act"])

    report["delta"] = report.synth - report.observed
    report["perc"] = report.delta / x.end.max()  # norm by day duration
    return report


def bin_times(x: DataFrame, mini: int = 0, maxi: int = 1440, step: int = 15):
    bins = list(range(mini - step, maxi, step))
    labels = bins[1:]
    return cut(x["start"], bins=bins, labels=labels).value_counts()


def time_distributions(population: DataFrame) -> tuple[dict, dict, dict]:
    acts = population.act.unique()
    acts.sort()
    starts = {k: [] for k in acts}
    ends = {k: [] for k in acts}
    durations = {k: [] for k in acts}
    for act, acts in population.groupby("act", observed=False):
        starts[act] = list(acts.start)
        ends[act] = list(acts.end)
        durations[act] = list(acts.duration)
    return starts, ends, durations


def plot_time_distributions(x, axs=None, label: str = ""):
    step = 30
    mini = 0
    maxi = 1441

    starts, ends, durations = time_distributions(x)

    bins = list(range(mini, maxi, step))
    if axs is None:
        fig, axs = plt.subplots(
            3,
            len(starts),
            figsize=(12, 5),
            sharex=True,
            sharey=False,
            tight_layout=True,
        )
    else:
        fig, axs = axs

    for i, act in enumerate(starts.keys()):
        axs[0][i].set_title(act.title(), fontstyle="italic")
        axs[0][i].hist(
            starts[act], bins=bins, density=True, histtype="step", label=label
        )
        axs[0][i].set_xlim(mini, maxi)
        axs[0][i].set_yticklabels([])
        axs[0][i].set(ylabel=None)
        axs[0][0].set(ylabel="Start times")
        axs[0][-1].legend()

        axs[1][i].hist(
            ends[act], bins=bins, density=True, histtype="step", label=label
        )
        axs[1][i].set_xlim(mini, maxi)
        axs[1][i].set_yticklabels([])
        axs[1][i].set(ylabel=None)
        axs[1][0].set(ylabel="End times")

        axs[2][i].hist(
            durations[act],
            bins=bins,
            density=True,
            histtype="step",
            label=label,
        )
        # pd.Series(durations[act]).plot.kde(ind=ind, bw_method=bw_method, ax=axs[2][i])
        axs[2][i].set_xlim(mini, maxi)
        axs[2][i].set_yticklabels([])
        axs[2][i].set(ylabel=None)
        axs[2][0].set(ylabel="Durations")
        axs[2][i].set(xlabel="Time of day (minutes)")

    return fig, axs


def plot_joint_time_distributions(x):
    step = 30
    mini = 0
    maxi = 1441

    starts, _, durations = time_distributions(x)

    bins = list(range(mini, maxi, step))
    duration_bins = list(range(mini, 800, step))

    fig, axs = plt.subplots(
        1,
        len(starts),
        figsize=(15, 4),
        sharex=True,
        sharey=True,
        tight_layout=True,
    )
    for i, (act, act_starts) in enumerate(starts.items()):
        act_durations = durations[act]

        axs[i].set_title(act.title(), fontstyle="italic")
        axs[0].set(ylabel="Durations (minutes)")
        axs[i].set(xlabel="Start times (minutes)")
        # axs[i].set_xlim(0, 1440)
        # axs[i].set_ylim(0, 1440)
        axs[i].hist2d(act_starts, act_durations, bins=(bins, duration_bins))
    return fig, axs
