import numpy as np

from analysis.colormap import color_algorithms
from analysis.LCA import lca
from analysis.lineplot import lineplot
from analysis.SSA import ssa


def plot_lca_ssa(
    axs,
    result,
    _config,
    step_sizes,
    metric,
    percent_metric,
    algorithm,
    config,
    cutoff,
    n_rows,
    n_cols,
):
    mean, standard_error, opt_step_size = lca(
        result, _config, step_sizes, metric, percent=percent_metric
    )
    lineplot(
        axs[0, 0],
        np.arange(len(mean)),
        mean,
        standard_error,
        opt_step_size,
        n_std=2.5,
        color=color_algorithms.get(algorithm),
        show_legend=True,
        xscale={"value": "linear", "base": 10},
        ylim={
            "bottom": config.get("lca_ylim_bottom"),
            "top": config.get("lca_ylim_top"),
        },
        xlim={
            "bottom": config.get("lca_xlim_bottom"),
            "top": config.get("lca_xlim_top"),
        },
    )

    means, standard_errors = ssa(
        result, _config, step_sizes, metric, percent=percent_metric, cutoff=cutoff
    )
    lineplot(
        axs[0, 1],
        step_sizes,
        means,
        standard_errors,
        algorithm,
        n_std=2.5,
        color=color_algorithms.get(algorithm),
        marker="o",
        show_legend=True,
        xscale={"value": "log", "base": 2},
        ylim={
            "bottom": config.get("lca_ylim_bottom"),
            "top": config.get("lca_ylim_top"),
        },
        xlim={
            "bottom": config.get("ssa_xlim_bottom"),
            "top": config.get("ssa_xlim_top"),
        },
    )

    for row in range(n_rows):
        for col in range(n_cols):
            axs[row, col].spines["right"].set_visible(False)
            axs[row, col].spines["top"].set_visible(False)
