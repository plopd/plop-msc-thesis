import copy

import numpy as np
from tqdm.auto import tqdm

from analysis.colormap import colors
from analysis.results import get_data_by


def line_plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a graph with a lineplot

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    data1 : array
       The x data

    data2 : array
       The y data

    param_dict : dict
       Dictionary of kwargs to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    out = ax.plot(data1, data2, **param_dict)
    return out


def get_LCA(ax, result, search_dct, metric, **param_dict):
    n_runs = param_dict.get("n_runs")
    stepsizes = result.get_param_val("alpha", search_dct, n_runs)
    optim_stepsize_data = None
    optim_stepsize = None
    current_best = np.inf
    for stepsize in tqdm(stepsizes):
        idx_exp_search_dict = copy.deepcopy(search_dct)
        idx_exp_search_dict["alpha"] = stepsize
        idx_data = result.find_experiment_by(idx_exp_search_dict, n_runs)
        data = result.load(idx_data)
        if data is None:
            print(f"LCA: No data found for stepsize: {stepsize}")
            continue
        mean, se = get_data_by(data, name=metric)
        if mean < current_best:
            current_best = mean
            optim_stepsize = stepsize
            optim_stepsize_data = data
    mean_err = optim_stepsize_data.mean(axis=0)
    s = optim_stepsize_data.std(axis=0) / np.sqrt(n_runs)

    steps = np.arange(len(mean_err))
    line_plotter(
        ax,
        steps,
        mean_err,
        param_dict={
            "label": f"{search_dct['algorithm']}"
            + "2^{}".format(int(np.log2(optim_stepsize))),
            "c": colors[search_dct["algorithm"]],
        },
    )

    se_upper = mean_err + 2.5 * s
    se_lower = mean_err - 2.5 * s
    ax.fill_between(
        np.arange(optim_stepsize_data.shape[1]),
        mean_err,
        se_upper,
        color=colors[search_dct["algorithm"]],
        alpha=0.15,
    )
    ax.fill_between(
        np.arange(optim_stepsize_data.shape[1]),
        mean_err,
        se_lower,
        color=colors[search_dct["algorithm"]],
        alpha=0.15,
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Walks/Episodes", labelpad=25)
    ax.set_ylabel(f"RMSVE over {n_runs} runs", labelpad=25)
    ax.legend(loc="upper right")

    return ax
