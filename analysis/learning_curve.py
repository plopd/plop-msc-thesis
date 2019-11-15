import copy

import numpy as np
from tqdm.auto import tqdm

from analysis.colormap import colors
from analysis.results import get_data_by
from results.exact_lstd_chain import compute_solution  # noqa f401


def get_LCA(ax, result, search_dct, metric):
    stepsizes = result.get_param_val("alpha", search_dct)
    optim_stepsize_data = None
    optim_stepsize = None
    current_best = np.inf
    for stepsize in tqdm(stepsizes):
        idx_exp_search_dict = copy.deepcopy(search_dct)
        idx_exp_search_dict["alpha"] = stepsize
        idx_data = result.find_experiment_by(idx_exp_search_dict)
        data = result.load(idx_data)
        if data is None:
            print(f"LCA: No data found for stepsize: {stepsize}")
            continue
        mean, se = get_data_by(data, name=metric)
        if mean < current_best:
            current_best = mean
            optim_stepsize = stepsize
            optim_stepsize_data = data
    m = optim_stepsize_data.mean(axis=0)
    s = optim_stepsize_data.std(axis=0) / np.sqrt(result.runs)

    ax.plot(
        m,
        label=f"{search_dct['algorithm']} "
        + "2^{}".format(int(np.log2(optim_stepsize))),
        c=colors[search_dct["algorithm"]],
    )
    se_upper = m + 2.5 * s
    se_lower = m - 2.5 * s
    ax.fill_between(
        np.arange(optim_stepsize_data.shape[1]),
        m,
        se_upper,
        color=colors[search_dct["algorithm"]],
        alpha=0.15,
    )
    ax.fill_between(
        np.arange(optim_stepsize_data.shape[1]),
        m,
        se_lower,
        color=colors[search_dct["algorithm"]],
        alpha=0.15,
    )
    # _, msve_lstd, _, _ = compute_solution(
    #     stepsize_search_dct['N'],
    #     stepsize_search_dct['algorithm'],
    #     'tabular',
    #     stepsize_search_dct['interest'],
    #     result.runs)
    # ax.axhline(msve_lstd[0,0], 0, len(m), color=colors[stepsize_search_dct["algorithm"]],
    #            label=f"{stepsize_search_dct['algorithm']}-LS", linestyle='-.')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Walks/Episodes", labelpad=25)
    ax.set_ylabel(f"RMSVE over {result.runs} runs", labelpad=25)
    ax.legend(loc="upper right")

    return ax
