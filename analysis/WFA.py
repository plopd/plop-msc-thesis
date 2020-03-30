import numpy as np

from analysis.colormap import colors
from analysis.colormap import locs
from analysis.results import get_data_by


def get_WF(ax, result, config, metric, **param_dict):
    n_runs = param_dict.get("n_runs")
    config.pop("step_size", None)
    step_sizes = result.get_value_param("step_size", config, n_runs)
    step_sizes = sorted(step_sizes)
    means = []
    n_negatives = 0
    xs = []
    for stepsize in step_sizes:
        config["step_size"] = stepsize
        idx_data = result.find_experiment_by(config, n_runs)
        data = result._load(idx_data)
        mean, _ = get_data_by(data, metric)
        cutoff = data[:, 0].mean()
        if mean <= cutoff:
            means.append(mean)
        else:
            means.append(cutoff)
            n_negatives += 1
        xs.append(
            np.random.uniform(
                locs[config["algorithm"]] - 0.15, locs[config["algorithm"]] + 0.15
            )
        )
    means = np.array(means)
    color = colors[config["algorithm"]]
    ax.scatter(xs, means, facecolors=color, edgecolors=color, s=75)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return ax
