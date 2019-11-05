import copy

import numpy as np

from analysis.colormap import colors
from analysis.colormap import locs
from analysis.results import get_data_by


def get_WF(ax, result, FEATURES, INTEREST, cutoff):
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
        n_negatives = 0
        xs = []
        for stepsize in stepsizes:
            idx_exp_search_dict = copy.deepcopy(stepsize_search_dct)
            idx_exp_search_dict["alpha"] = stepsize
            idx_data = result.find_experiment_by(idx_exp_search_dict)
            data = result._load(idx_data)
            mean, _ = get_data_by(data, name="end")
            if mean <= cutoff:
                m.append(mean)
            else:
                m.append(cutoff)
                n_negatives += 1
            xs.append(np.random.uniform(locs[method] - 0.15, locs[method] + 0.15))
        m = np.array(m)
        data_learning_curve_methods.append({"mean": m, "algorithm": method})
        ax.scatter(
            xs,
            data_learning_curve_methods[idx_m]["mean"],
            facecolors="none",
            edgecolors=colors[method],
            s=100,
        )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.text(
            locs[method],
            cutoff,
            "{:2d}%".format(int(100 * n_negatives / len(stepsizes))),
            color=colors[method],
        )
    ax.set_xticks(ticks=[locs[m] for m in methods])
    ax.set_xticklabels(labels=[m.upper() for m in methods])
    ax.set_xlabel("Methods", labelpad=25)
