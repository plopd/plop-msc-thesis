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
    sweeper = Sweeper(Path(__file__).parents[1] / "configs" / f"{config_fn}.json")
    config = sweeper.parse(sweep_id)
    data_path = Path(f"~/scratch/{config.get('env')}").expanduser()
    save_path = path_exists(Path(__file__).parents[0] / config_fn)
    results = Result(f"{config_fn}.json", data_path, config_fn)

    algorithms = sorted(list(results.get_param_val("algorithm", {}, 1)), reverse=True)
    representations = sorted(
        list(results.get_param_val("representations", {}, 1)), reverse=True
    )
    trace_decays = list(results.get_param_val("trace_decay", {}, 1))
    num_episodes = list(results.get_param_val("n_episodes", {}, 1))[0]

    fig, axes = plt.subplots(
        len(representations),
        len(algorithms),
        figsize=(10, 9),
        sharey="all",
        sharex="all",
    )

    for row, representation in enumerate(representations):
        config = {}
        config["representations"] = representation
        for col, algorithm in enumerate(algorithms):
            config["algorithm"] = algorithm
            for trace_decay in trace_decays:
                color = lmbdas.get(trace_decay)
                config.pop("step_size", None)
                config["trace_decay"] = trace_decay
                step_sizes = list(results.get_param_val("step_size", config, 1))
                step_sizes = sorted(step_sizes)
                num_step_size = len(step_sizes)
                means = np.zeros(num_step_size)
                se_errors = np.zeros(num_step_size)
                for i, step_size in enumerate(step_sizes):
                    config["step_size"] = step_size
                    ids = results.find_experiment_by(config, 1)
                    data = results.load(ids)
                    mean, se = get_data_by(data, name="auc", percent=1.0,)
                    cutoff = data[:, 0].mean()
                    means[i] = mean
                    se_errors[i] = se
                    means.clip(0, cutoff)

                # axes[row, col].set_title(f"{algorithm},{representation}")
                axes[row, col].scatter(
                    step_sizes, means, c=color, marker="o", label=f"{trace_decay}"
                )
                axes[row, col].errorbar(
                    step_sizes, means, yerr=2.5 * se_errors, color=color
                )
                # axes[-1, col].set_xlabel("Step size")
                # axes[row, 0].set_ylabel(f"RMSVE over {num_runs} 19 states \n and first {num_episodes} episodes")

            y_ticks = np.arange(0, 0.55, 0.1).astype(np.float32)
            for i in range(len(axes)):
                axes[row, i].spines["right"].set_visible(False)
                axes[row, i].spines["top"].set_visible(False)
                axes[row, i].set_yticks(y_ticks)
                axes[row, i].set_yticklabels(y_ticks)
                axes[row, i].set_ylim(0.0, 0.5)
                # axes[row, i].legend(loc='upper right')

            plt.tight_layout()
            filename = f"RandomWalk-Ep-{num_episodes}"
            filename = filename.replace(".", "_")
            plt.savefig(save_path / filename)


if __name__ == "__main__":
    main()
