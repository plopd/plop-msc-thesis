import os
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
    config_rootpath = Path(__file__).parents[1] / "configs"
    sweeper = Sweeper(config_rootpath / f"{config_fn}.json")
    config = sweeper.parse(sweep_id)
    n_episodes = config.get("n_episodes")
    representations = config.get("representations").split(",")
    discount_rate = config.get("discount_rate")
    xmin, xmax, xstep = tuple([float(x) for x in config.get("xlim").split(",")])
    ymin, ymax, ystep = tuple([float(y) for y in config.get("ylim").split(",")])
    experiment = config.get("experiment")
    sweeper = Sweeper(config_rootpath / f"{experiment}.json")
    config = sweeper.parse(0)
    data_path = Path(f"{os.environ.get('SCRATCH')}/{config.get('env')}").expanduser()
    save_rootpath = path_exists(Path(__file__).parents[0] / experiment)
    filename = f"RW-gamma-{discount_rate}-n_episode-{n_episodes}".replace(".", "")
    savepath = save_rootpath / filename
    results = Result(f"{experiment}.json", data_path, experiment)

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
    y_ticks = np.arange(ymin, ymax, ystep).astype(np.float32)[1:]
    x_ticks = np.arange(xmin, xmax, xstep).astype(np.float32)[1:]
    for row in range(n_rows):
        for col in range(n_cols):
            axes[row, col].spines["right"].set_visible(False)
            axes[row, col].spines["top"].set_visible(False)
            axes[row, col].set_xticks(x_ticks)
            axes[row, col].set_xticklabels(x_ticks)
            axes[row, col].set_yticks(y_ticks)
            axes[row, col].set_yticklabels(y_ticks)
            axes[row, col].set_ylim(ymin, ymax)
            axes[row, col].set_xlim(xmin, xmax)

    for row in range(n_rows):
        config = {}
        config["representations"] = representations[row]
        config["discount_rate"] = discount_rate
        for col in range(n_cols):
            config["algorithm"] = algorithms[col]
            for trace_decay in trace_decays:
                color = lmbdas.get(trace_decay)
                config.pop("step_size", None)
                config["trace_decay"] = trace_decay
                step_sizes = list(results.get_param_val("step_size", config, 1))
                step_sizes = sorted(step_sizes)
                step_sizes = np.array(step_sizes)
                step_sizes = step_sizes[np.where(step_sizes <= xmax)[0]]
                num_step_size = len(step_sizes)
                means = np.zeros(num_step_size)
                se_errors = np.zeros(num_step_size)
                for step_size_idx, step_size in enumerate(step_sizes):
                    config["step_size"] = step_size
                    ids = results.find_experiment_by(config, 1)
                    data = results.load(ids)
                    data = data[:, :n_episodes]
                    mean, se = get_data_by(data, name="auc", percent=1.0)
                    cutoff = data[:, 0].mean()
                    cutoff += cutoff * 0.1
                    means[step_size_idx] = mean
                    means = np.nan_to_num(means, nan=np.inf)
                    means = means.clip(0, cutoff)
                    se_errors[step_size_idx] = se
                    se_errors = np.nan_to_num(se_errors)
                    se_errors[np.where(means >= cutoff)[0]] = 0

                axes[row, col].plot(step_sizes, means, c=color, label=f"{trace_decay}")
                axes[row, col].errorbar(
                    step_sizes, means, yerr=2.5 * se_errors, color=color
                )
    plt.tight_layout()
    plt.savefig(savepath)


if __name__ == "__main__":
    main()
