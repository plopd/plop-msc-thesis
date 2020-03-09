import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from alphaex.sweeper import Sweeper

from analysis.colormap import colors
from analysis.colormap import linestyles
from analysis.results import get_data_by
from analysis.results import Result
from representations.configs import to_name
from utils.utils import path_exists

matplotlib.use("agg")
matplotlib.rcParams.update({"font.size": 24})


def main():
    plot(sweep_id=int(sys.argv[1]), config_fn=sys.argv[2])


def plot(sweep_id, config_fn):
    config_root_path = Path(__file__).parents[1] / "configs"
    sweeper = Sweeper(config_root_path / f"{config_fn}.json")
    config = sweeper.parse(sweep_id)
    experiments = config.get("experiment").split(",")
    algorithms = config.get("algorithms").split(",")
    num_runs = config.get("num_runs")
    env = config.get("env")
    data_path = Path(f"~/scratch/{env}").expanduser()
    save_path = path_exists(Path(__file__).parents[0] / config_fn)
    n_rows = len(experiments) + 1
    n_cols = len(algorithms)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_rows * 7, n_cols * 5), sharey="all", sharex="col"
    )

    for row, experiment in enumerate(experiments):
        _config = {}
        results = Result(
            config_filename=f"{experiment}.json",
            datapath=data_path,
            experiment=experiment,
        )
        representations = list(results.get_param_val("representations", {}, 1))[0]
        _config["representations"] = representations
        _config["num_states"] = config.get("num_states")
        for algorithm in algorithms:
            color = colors.get(algorithm)
            _config["algorithm"] = algorithm
            _config.pop("step_size", None)
            step_sizes = results.get_param_val("step_size", _config, num_runs)
            step_sizes = sorted(step_sizes)
            num_step_sizes = len(step_sizes)
            means = np.zeros(num_step_sizes)
            standard_errors = np.zeros(num_step_sizes)
            optimum_step_size = None
            optimum_error = np.inf

            for i, step_size in enumerate(step_sizes):
                _config["step_size"] = step_size
                ids = results.find_experiment_by(_config, num_runs)
                data = results.load(ids)
                mean, se = get_data_by(
                    data,
                    name=config.get("metric"),
                    percent=config.get("percent_metric"),
                )
                cutoff = data[:, 0].mean()
                # Find best instance of algorithm
                if mean <= optimum_error:
                    optimum_error = mean
                    optimum_step_size = step_size
                # Record step-size sensitivity
                means[i] = mean
                means = np.nan_to_num(means, nan=np.inf)
                means = means.clip(0, cutoff)
                standard_errors[i] = se
                standard_errors = np.nan_to_num(standard_errors)
                standard_errors[np.where(means >= cutoff)[0]] = 1e-8

            # ################ PLOT LCA ####################
            _config["step_size"] = optimum_step_size
            ids = results.find_experiment_by(_config, num_runs)
            data = results.load(ids)

            mean = data.mean(axis=0)
            se = data.std(axis=0) / np.sqrt(num_runs)
            steps = int(np.ceil(len(mean) * config.get("percent_plot")))
            mean = mean[:steps]
            se = se[:steps]
            axes[row, 0].plot(mean, c=color, label=algorithm)
            axes[row, 0].fill_between(
                np.arange(len(mean)),
                mean + 2.5 * se,
                mean - 2.5 * se,
                color=color,
                alpha=0.15,
            )
            axes[-1, 0].set_xlabel("Walks/Episodes")
            axes[0, 0].set_ylabel(
                f"RMSVE over {num_runs} runs\n" f"({config.get('metric')} performance)"
            )

            if _config.get("baseline") == 1:
                _config["algorithm"] = "LSTD" if algorithm == "TD" else "ELSTD"
                _config.pop("step_size", None)
                ids = results.find_experiment_by(_config, num_runs)
                data = results.load(ids)
                mean = data[
                    :, -1
                ].mean()  # During early learning the inverse of A may be unstable

                axes[row, 0].axhline(
                    mean,
                    xmin=0,
                    xmax=len(data),
                    color=color,
                    label=_config["algorithm"],
                    linestyle=linestyles.get(algorithm),
                )
            axes[row, 0].legend(loc="upper right")

            ################ PLOT SSA ####################
            axes[row, 1].errorbar(
                step_sizes, means, yerr=2.5 * standard_errors, color=color, fmt="o-"
            )
            axes[row, 1].set_xscale("log", basex=2)
            axes[-1, 1].set_xlabel("Step size")

            axes[row, 1].set_title(f"{to_name.get(representations)}", loc="left")

        y_tick_values = np.arange(
            config.get("y_min"), cutoff + config.get("step"), config.get("step"),
        ).astype(np.float32)
        for i in range(len(axes)):
            axes[row, i].spines["right"].set_visible(False)
            axes[row, i].spines["top"].set_visible(False)
            axes[row, i].set_yticks(y_tick_values)
            axes[row, i].set_yticklabels(y_tick_values)
            axes[row, i].set_ylim(config.get("y_min"), cutoff - 0.05)

        plt.tight_layout()
        filename = (
            f"plot_features_RW-{config.get('num_states')}-metric-{config.get('metric')}"
        )
        filename = filename.replace(".", "_")
        plt.savefig(save_path / filename)


if __name__ == "__main__":
    main()
