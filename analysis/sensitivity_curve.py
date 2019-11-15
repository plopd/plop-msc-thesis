import copy

import numpy as np
from tqdm import tqdm

from analysis.colormap import colors
from analysis.results import get_data_by


def get_SSA(ax, result, search_dct, cutoff, name):
    stepsizes = result.get_param_val("alpha", search_dct)
    stepsizes = sorted(stepsizes)
    m = []
    s = []
    xs = []
    for stepsize in tqdm(stepsizes):
        idx_exp_search_dict = copy.deepcopy(search_dct)
        idx_exp_search_dict["alpha"] = stepsize
        idx_data = result.find_experiment_by(idx_exp_search_dict)
        data = result.load(idx_data)
        if data is None:
            continue
        mean, se = get_data_by(data, name=name)
        if mean < cutoff:
            m.append(mean)
            s.append(se)
        else:
            m.append(cutoff)
            s.append(0.0)
        xs.append(stepsize)
    m = np.array(m)
    s = np.array(s)
    ax.plot(
        xs,
        m,
        label=f"{search_dct['algorithm']}",
        c=colors[search_dct["algorithm"]],
        marker="o",
    )
    se_upper = m + 2.5 * s
    se_lower = m - 2.5 * s
    ax.fill_between(xs, m, se_upper, color=colors[search_dct["algorithm"]], alpha=0.15)
    ax.fill_between(xs, m, se_lower, color=colors[search_dct["algorithm"]], alpha=0.15)
    ax.set_xscale("log")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Stepsize", labelpad=25)

    return ax
