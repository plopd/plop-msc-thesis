import copy

import numpy as np

from analysis.colormap import colors
from analysis.results import get_data_by
from results.exact_lstd_chain import compute_solution  # noqa f401


# Gather methods by experiment:
#     data_learning_curve_methods = [{mean, se, params_method**}]
#     For m in methods:
#         Gather stepsizes by method m
#         best_stepsize_data =
#         For s in stepsizes:
#             Gather data by m and s
#             Compare to best
#     plot_learning_curve(data_learning_curve_methods)


def get_LCA(ax, result, N, features, interest, n_runs_exact):
    methods = result.get_param_val("algorithm", {"env": "chain"})
    data_learning_curve_methods = []
    for idx_m, method in enumerate(methods):
        stepsize_search_dct = {
            "algorithm": method,
            "env": "chain",
            "features": features,
            "interest": interest,
        }
        stepsizes = result.get_param_val("alpha", stepsize_search_dct)
        best_stepsize_data = None
        best_stepsize = None
        best_val = np.inf
        for stepsize in stepsizes:
            idx_exp_search_dict = copy.deepcopy(stepsize_search_dct)
            idx_exp_search_dict["alpha"] = stepsize
            idx_data = result.find_experiment_by(idx_exp_search_dict)
            data = result._load(idx_data)
            mean, se = get_data_by(data, name="end")
            if mean < best_val:
                best_val = mean
                best_stepsize = stepsize
                best_stepsize_data = data
        m = best_stepsize_data.mean(axis=0)
        s = best_stepsize_data.std(axis=0) / np.sqrt(result.runs)
        data_learning_curve_methods.append(
            {"mean": m, "se": s, "algorithm": method, "alpha": best_stepsize}
        )
        ax.plot(
            data_learning_curve_methods[idx_m]["mean"],
            label=f"{data_learning_curve_methods[idx_m]['algorithm']} "
            + "2^{}".format(int(np.log2(data_learning_curve_methods[idx_m]["alpha"]))),
            c=colors[method],
        )
        se_upper = m + 2.5 * s
        se_lower = m - 2.5 * s
        ax.fill_between(
            np.arange(best_stepsize_data.shape[1]),
            m,
            se_upper,
            color=colors[method],
            alpha=0.15,
        )
        ax.fill_between(
            np.arange(best_stepsize_data.shape[1]),
            m,
            se_lower,
            color=colors[method],
            alpha=0.15,
        )
        # _, msve_lstd, _, _ = compute_solution(N, method, features, interest, n_runs_exact)
        # ax.axhline(msve_lstd[0,0], 0, len(data_learning_curve_methods[idx_m]['mean']), color=colors[method],
        #            label=f"{data_learning_curve_methods[idx_m]['algorithm']}-LS", linestyle='-.')
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    ax.set_xlabel("Walks/Episodes", labelpad=25)
    ax.set_ylabel(f"RMSVE over {result.runs} runs", labelpad=25)
    ax.legend(loc="upper right")
