import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from alphaex.sweeper import Sweeper

from analysis.LCA import lca
from analysis.lineplot import lineplot
from analysis.results import Result
from analysis.SSA import ssa
from utils.utils import path_exists

matplotlib.use("agg")
matplotlib.rcParams.update({"font.size": 24})


def main():
    plot(sweep_id=int(sys.argv[1]), config_fn=sys.argv[2])


def plot(sweep_id, config_fn):
    config_root_path = Path(__file__).parents[1] / "configs"
    sweeper = Sweeper(config_root_path / f"{config_fn}.json")
    config = sweeper.parse(sweep_id)
    experiment = config.get("experiment")
    num_runs = config.get("num_runs")
    env = config.get("env")
    algorithms = config.get("algorithms").split(",")

    data_path = Path(f"~/scratch/{env}").expanduser()
    save_path = path_exists(Path(__file__).parents[0] / config_fn)
    n_cols = 2
    n_rows = 2
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 5, n_rows * 4),
        squeeze=False,
        sharex="col",
        sharey="all",
    )

    result = Result(experiment, data_path, num_runs)
    cutoff = result.data[:, 0].mean()

    for algorithm in algorithms:
        _config = {}
        _config["algorithm"] = algorithm
        step_sizes = result.get_value_param("step_size", _config)

        mean, standard_error, opt_step_size = lca(
            result, _config, step_sizes, "interim", percent=0.1
        )
        lineplot(
            axs[0, 0],
            np.arange(len(mean)),
            mean,
            standard_error,
            opt_step_size,
            n_std=2.5,
            show_legend=True,
            xscale={"value": "linear", "base": 10},
            ylim={"bottom": None, "top": cutoff - 0.01 * cutoff},
        )

        means, standard_errors = ssa(
            result, _config, step_sizes, "interim", percent=0.1, cutoff=cutoff
        )
        lineplot(
            axs[0, 1],
            step_sizes,
            means,
            standard_errors,
            _config.get("algorithm"),
            n_std=2.5,
            marker="o",
            show_legend=True,
            xscale={"value": "log", "base": 2},
            ylim={"bottom": None, "top": cutoff - 0.01 * cutoff},
        )

        fig.tight_layout()
        plt.savefig(save_path / f"{config_fn}")


if __name__ == "__main__":
    main()


# for i, step_size in enumerate(step_sizes):
#     _config["step_size"] = step_size
#     ids = result.find_experiment_by(_config)
#     data = result.data[ids, :]
#     mean = data.mean(axis=0)
#     standard_error = data.std(axis=0) / np.sqrt(num_runs)
#     label = f"{step_size}*" if opt_step_size == step_size else step_size
#     lineplot(axs[0,1], np.arange(len(mean)), mean, standard_error, label, n_std=2.5, show_legend=True,
#              xscale={"value": "linear", "base": 10})
