from typing import Optional, Tuple

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap as CMap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from pandas import DataFrame

from caveat.evaluate.features.transitions import sequence_probs


def _probs_plot(
    population: DataFrame, acts: list[str], cmap: Optional[CMap], ax=Axes
) -> Tuple[Figure, Axes]:
    probs = sequence_probs(population)
    accumulated = probs[::-1].cumsum()[::-1]

    ys = []
    widths = []
    heights = []
    lefts = []
    cols = []
    for idx, p, ap in zip(probs.index, probs, accumulated):
        seq = idx[1].split(">")
        width = 1 / len(seq)
        for i, act in enumerate(seq):
            ys.append(ap - p)
            widths.append(width)
            heights.append(p)
            lefts.append(i * width)
            cols.append(cmap[act])

    ax.barh(
        y=ys, width=widths, height=heights, left=lefts, color=cols, align="edge"
    )
    ax.hlines(ys, xmin=0, xmax=1, color="white", linewidth=0.1)
    ax.axis("off")


def sequence_prob_plot(
    observed: DataFrame, ys: Optional[dict[DataFrame]], **kwargs
) -> Figure:
    acts = list(observed.act.value_counts(ascending=False).index)
    cmap = kwargs.pop("cmap", None)
    if cmap is None:
        cmap = plt.cm.Set3
        colors = cmap.colors
        factor = (len(acts) // len(colors)) + 1
        cmap = dict(zip(acts, colors * factor))

    n_plots = len(ys) + 2
    ratios = [1 for _ in range(n_plots)]
    ratios[-1] = 0.5

    fig, axs = plt.subplots(
        1,
        n_plots,
        figsize=kwargs.pop("figsize", (12, 5)),
        sharex=True,
        sharey=True,
        tight_layout=True,
        gridspec_kw={"width_ratios": ratios},
    )
    acts = list(observed.act.value_counts(ascending=False).index)
    _probs_plot(observed, acts, ax=axs[0], cmap=cmap)
    observed_title = kwargs.pop("observed_title", "Observed")
    axs[0].set_title(observed_title)
    if ys is None:
        return fig
    for i, (name, y) in enumerate(ys.items()):
        _probs_plot(y, acts, ax=axs[i + 1], cmap=cmap)
        axs[i + 1].set_title(name.title())

    elements = [Patch(facecolor=cmap[act], label=act.title()) for act in acts]
    axs[-1].axis("off")
    axs[-1].legend(handles=elements, loc="center left", frameon=False)

    return fig


#
