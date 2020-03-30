import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from alphaex.sweeper import Sweeper

from analysis.results import Result
from utils.utils import path_exists

matplotlib.use("agg")
matplotlib.rcParams.update({"font.size": 24})


def main():
    plot(config_fn=sys.argv[1])


def plot(config_fn):
    config_root_path = Path(__file__).parents[1] / "configs"
    sweeper = Sweeper(config_root_path / f"{config_fn}.json")
    config = sweeper.parse(0)
    env = config.get("env")
    data_path = Path(f"~/scratch/{env}").expanduser()
    save_path = path_exists(Path(__file__).parents[0] / config_fn)
    fig, ax = plt.subplots(1, 1, sharey="all", sharex="col")
    num_runs = config.get("num_runs")
    results = Result(config_fn, data_path, num_runs)
    algorithm = "TD"
    _config = {}
    _config["algorithm"] = algorithm
    step_sizes = results.get_value_param("step_size", _config)

    for i, step_size in enumerate(step_sizes):
        _config["step_size"] = step_size
        ids = results.find_experiment_by(_config)

        data = results.data[ids, :]
        mean = data.mean(axis=0)
        se = data.std(axis=0) / np.sqrt(num_runs)
        ax.plot(mean, label=step_size)
        ax.fill_between(
            np.arange(len(mean)), mean + 2.5 * se, mean - 2.5 * se, alpha=0.15,
        )
    plt.yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
    plt.xticks([0, 25, 50, 75, 100])
    plt.legend()
    fig.tight_layout()
    filename = f"{config_fn}"
    filename = filename.replace(".", "_")
    plt.savefig(save_path / filename)


if __name__ == "__main__":
    main()
