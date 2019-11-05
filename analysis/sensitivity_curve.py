import copy

import numpy as np

from analysis.colormap import colors
from analysis.results import get_data_by


def get_SSA(ax, result, FEATURES, INTEREST, cutoff):

    methods = result.get_param_val("algorithm", {"env": "chain"})
    data_learning_curve_methods = []
    for idx_m, method in enumerate(methods):
        stepsize_search_dct = {
            "algorithm": method,
            "env": "chain",
            "features": FEATURES,
            "interest": INTEREST,
        }
        stepsizes = result.get_param_val("alpha", stepsize_search_dct)
        stepsizes = sorted(stepsizes)
        m = []
        s = []
        xs = []
        for stepsize in stepsizes:
            idx_exp_search_dict = copy.deepcopy(stepsize_search_dct)
            idx_exp_search_dict["alpha"] = stepsize
            idx_data = result.find_experiment_by(idx_exp_search_dict)
            data = result._load(idx_data)
            mean, se = get_data_by(data, name="end")
            if mean < cutoff:
                m.append(mean)
                s.append(se)
            else:
                m.append(cutoff)
                s.append(0.0)
            xs.append(stepsize)
        m = np.array(m)
        s = np.array(s)
        data_learning_curve_methods.append({"mean": m, "se": s, "algorithm": method})
        ax.plot(
            xs,
            data_learning_curve_methods[idx_m]["mean"],
            label=f"{data_learning_curve_methods[idx_m]['algorithm']}",
            c=colors[method],
            marker="o",
        )
        se_upper = m + 2.5 * s
        se_lower = m - 2.5 * s
        ax.fill_between(xs, m, se_upper, color=colors[method], alpha=0.15)
        ax.fill_between(xs, m, se_lower, color=colors[method], alpha=0.15)
        ax.set_xscale("log")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    ax.set_xlabel("Stepsize", labelpad=25)
