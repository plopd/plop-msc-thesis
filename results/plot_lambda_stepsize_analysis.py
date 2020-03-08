import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from alphaex.sweeper import Sweeper

from analysis.colormap import lmbdas
from analysis.results import get_data_by
from analysis.results import Result
from utils.utils import path_exists

matplotlib.use("agg")
matplotlib.rcParams.update({"font.size": 24})


def main():
    plot(sweep_id=int(sys.argv[1]), config_fn=sys.argv[2])


def plot(sweep_id, config_fn):
    config_root_path = Path(__file__).parents[1] / "configs"
    sweeper = Sweeper(config_root_path / f"{config_fn}.json")
    config = sweeper.parse(sweep_id)
    n_episodes = config.get("n_episodes")
    representations = config.get("representations").split(",")
    discount_rate = config.get("discount_rate")
    x_lim = [float(x) for x in config.get("x_lim").split(",")]
    _config_fn = config.get("experiment")
    sweeper = Sweeper(config_root_path / f"{_config_fn}.json")
    config = sweeper.parse(0)
    data_path = Path(f"~/scratch/{config.get('env')}").expanduser()
    save_path = path_exists(Path(__file__).parents[0] / _config_fn)
    results = Result(f"{_config_fn}.json", data_path, _config_fn)

    algorithms = sorted(list(results.get_param_val("algorithm", {}, 1)), reverse=True)
    trace_decays = list(results.get_param_val("trace_decay", {}, 1))

    n_cols = len(algorithms)
    n_rows = len(representations)
    # idx = np.arange(1, n_rows*n_cols+1).reshape((n_rows, n_cols))
    # fig = plt.figure()
    # for n_col in range(n_cols):
    #     for n_row in range(n_rows):
    #         fig.add_subplot(n_rows, n_cols, idx[n_row, n_col])

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey="all", sharex="all",
    )

    for row, representation in enumerate(representations):
        config = {}
        config["representations"] = representation
        config["discount_rate"] = discount_rate
        for col, algorithm in enumerate(algorithms):
            config["algorithm"] = algorithm
            for trace_decay in trace_decays:
                color = lmbdas.get(trace_decay)
                config.pop("step_size", None)
                config["trace_decay"] = trace_decay
                step_sizes = list(results.get_param_val("step_size", config, 1))
                step_sizes = sorted(step_sizes)
                step_sizes = np.array(step_sizes)
                step_sizes = step_sizes[np.where(step_sizes <= x_lim[1])[0]]
                num_step_size = len(step_sizes)
                means = np.zeros(num_step_size)
                se_errors = np.zeros(num_step_size)
                for i, step_size in enumerate(step_sizes):
                    config["step_size"] = step_size
                    ids = results.find_experiment_by(config, 1)
                    data = results.load(ids)
                    data = data[:, :n_episodes]
                    mean, se = get_data_by(data, name="auc", percent=1.0)
                    cutoff = data[:, 0].mean()
                    cutoff += 0.1
                    means[i] = mean
                    means = means.clip(0, cutoff)
                    se_errors[i] = se
                    se_errors[np.where(means >= cutoff)[0]] = 0.0

                axes[row, col].plot(step_sizes, means, c=color, label=f"{trace_decay}")
                axes[row, col].errorbar(
                    step_sizes, means, yerr=2.5 * se_errors, color=color
                )

            y_ticks = np.arange(0, cutoff, 0.1).astype(np.float32)
            x_ticks = np.arange(x_lim[0], x_lim[1], 2 * x_lim[1] / 10).astype(
                np.float32
            )
            for i in range(n_cols):
                axes[row, i].spines["right"].set_visible(False)
                axes[row, i].spines["top"].set_visible(False)
                axes[row, i].set_xticks(x_ticks)
                axes[row, i].set_xticklabels(x_ticks)
                axes[row, i].set_yticks(y_ticks)
                axes[row, i].set_yticklabels(y_ticks)
                axes[row, i].set_ylim(0.0, cutoff - 0.05)
                axes[row, i].set_xlim(x_lim[0], x_lim[1])

    plt.tight_layout()
    filename = f"RandomWalk_NumEp_{n_episodes}_StepSize_{x_lim[0]}-{x_lim[1]}_DiscountRate_{discount_rate}".replace(
        ".", "-"
    )
    plt.savefig(save_path / filename)


if __name__ == "__main__":
    main()
