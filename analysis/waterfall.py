import copy

import numpy as np
from tqdm import tqdm

from analysis.colormap import colors
from analysis.colormap import locs
from analysis.results import get_data_by


def get_WF(ax, result, stepsize_search_dct, cutoff, name, methods):
    stepsizes = result.get_param_val("alpha", stepsize_search_dct)
    stepsizes = sorted(stepsizes)
    m = []
    n_negatives = 0
    xs = []
    for stepsize in tqdm(stepsizes):
        idx_exp_search_dict = copy.deepcopy(stepsize_search_dct)
        idx_exp_search_dict["alpha"] = stepsize
        idx_data = result.find_experiment_by(idx_exp_search_dict)
        data = result.load(idx_data)
        if data is None:
            continue
        mean, _ = get_data_by(data, name=name)
        if mean <= cutoff:
            m.append(mean)
        else:
            m.append(cutoff)
            n_negatives += 1
        xs.append(
            np.random.uniform(
                locs[stepsize_search_dct["algorithm"]] - 0.15,
                locs[stepsize_search_dct["algorithm"]] + 0.15,
            )
        )
    m = np.array(m)
    ax.scatter(
        xs,
        m,
        facecolors="none",
        edgecolors=colors[stepsize_search_dct["algorithm"]],
        s=100,
    )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.text(
        locs[stepsize_search_dct["algorithm"]],
        cutoff,
        "{:2d}%".format(int(100 * n_negatives / len(stepsizes))),
        color=colors[stepsize_search_dct["algorithm"]],
    )
    ax.set_xticks(ticks=[locs[m] for m in methods])
    ax.set_xticklabels(labels=[m.upper() for m in methods])
    ax.set_xlabel("Methods", labelpad=25)

    return ax
