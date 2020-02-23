import json
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
from utils.utils import remove_keys_with_none_value

matplotlib.use("agg")
matplotlib.rcParams.update({"font.size": 24})


def main():
    ids = [int(x) for x in sys.argv[1].split(",")]
    fig, axes = plt.subplots(
        len(ids), 2, figsize=(12, 5 * len(ids)), sharey="all", sharex="col"
    )
    for row, sweep_id in enumerate(ids):
        config_filename = sys.argv[2]
        sweeper = Sweeper(
            Path(__file__).parents[1] / "configs" / f"{config_filename}.json"
        )
        param_cfg = sweeper.parse(sweep_id)

        exp_info = {
            "num_states": param_cfg.get("num_states"),
            "num_runs": param_cfg.get("num_runs"),
            "order": param_cfg.get("order"),
            "metric": param_cfg.get("metric"),
            "experiment": param_cfg.get("experiment"),
            "env": param_cfg.get("env"),
            "representations": param_cfg.get("representations"),
            "num_ones": param_cfg.get("num_ones"),
            "num_features": param_cfg.get("num_features"),
            "num_dims": param_cfg.get("num_dims"),
            "interest": param_cfg.get("interest"),
            "discount_rate": param_cfg.get("discount_rate"),
            "trace_decay": param_cfg.get("trace_decay"),
            "baseline": param_cfg.get("baseline"),
            "algorithms": param_cfg.get("algorithms"),
            "percent_metric": param_cfg.get("percent_metric"),
            "percent_plot": param_cfg.get("percent_plot"),
            "y_min": param_cfg.get("y_min"),
            "y_max": param_cfg.get("y_max"),
            "step": param_cfg.get("step"),
        }

        datapath = Path(f"~/scratch/{exp_info.get('env')}").expanduser()
        save_path = path_exists(Path(__file__).parents[0])

        results = Result(
            f"{exp_info.get('experiment')}.json", datapath, exp_info.get("experiment")
        )

        config = {
            "num_states": param_cfg.get("num_states"),
            "order": param_cfg.get("order"),
            "env": param_cfg.get("env"),
            "representations": param_cfg.get("representations"),
            "num_ones": param_cfg.get("num_ones"),
            "num_features": param_cfg.get("num_features"),
            "num_dims": param_cfg.get("num_dims"),
            "max_x": param_cfg.get("max_x"),
            "min_x": param_cfg.get("min_x"),
            "a": param_cfg.get("a"),
            "b": param_cfg.get("b"),
            "interest": param_cfg.get("interest"),
            "discount_rate": param_cfg.get("discount_rate"),
            "trace_decay": param_cfg.get("trace_decay"),
        }
        remove_keys_with_none_value(config)

        print(json.dumps(config, indent=4))

        for algo in exp_info.get("algorithms").split(","):
            color = colors.get(algo)
            config["algorithm"] = algo
            config.pop("step_size", None)
            step_sizes = results.get_param_val(
                "step_size", config, exp_info.get("num_runs")
            )
            step_sizes = sorted(step_sizes)
            num_step_sizes = len(step_sizes)
            means = np.zeros(num_step_sizes)
            std_errors = np.zeros(num_step_sizes)
            xs = np.zeros(num_step_sizes)
            optimum_step_size = None
            optimum_error = np.inf

            for i, step_size in enumerate(step_sizes):
                config["step_size"] = step_size
                ids = results.find_experiment_by(config, exp_info.get("num_runs"))
                data = results.load(ids)
                mean, se = get_data_by(
                    data,
                    name=exp_info.get("metric"),
                    percent=exp_info.get("percent_metric"),
                )
                cutoff = data[:, 0].mean()
                # Find best instance of algorithm
                if mean <= optimum_error:
                    optimum_error = mean
                    optimum_step_size = step_size
                # Record step-size sensitivity
                means[i] = mean
                std_errors[i] = se
                xs[i] = step_size
                means.clip(0, cutoff)

            ################ PLOT LCA ####################
            config["step_size"] = optimum_step_size
            ids = results.find_experiment_by(config, exp_info.get("num_runs"))
            data = results.load(ids)

            mean = data.mean(axis=0)
            se = data.std(axis=0) / np.sqrt(exp_info.get("num_runs"))
            steps = int(len(mean) * exp_info.get("percent_plot"))
            mean = mean[:steps]
            se = se[:steps]
            axes[row, 0].plot(mean, c=color, label=algo)
            axes[row, 0].fill_between(
                np.arange(len(mean)),
                mean + 2.5 * se,
                mean - 2.5 * se,
                color=color,
                alpha=0.15,
            )
            axes[-1, 0].set_xlabel("Walks/Episodes")
            axes[0, 0].set_ylabel(
                f"RMSVE over {exp_info.get('num_runs')} runs\n"
                f"({exp_info.get('metric')} performance)"
            )

            if exp_info.get("baseline") == 1:
                config["algorithm"] = "LSTD" if algo == "TD" else "ELSTD"
                config.pop("step_size", None)
                ids = results.find_experiment_by(config, exp_info.get("num_runs"))
                data = results.load(ids)
                mean = data[
                    :, -1
                ].mean()  # During early learning the inverse of A may be unstable

                axes[row, 0].axhline(
                    mean,
                    xmin=0,
                    xmax=len(data),
                    color=color,
                    label=config["algorithm"],
                    linestyle=linestyles.get(algo),
                )
            # axes[row, 0].legend(loc='upper right')

            ################ PLOT SSA ####################
            axes[row, 1].scatter(xs, means, c=color, marker="o")
            axes[row, 1].errorbar(
                xs, means, yerr=2.5 * std_errors, color=color, capsize=5,
            )
            axes[row, 1].set_xscale("log")
            axes[-1, 1].set_xlabel("Step size")

            axes[row, 1].set_title(
                f"{to_name.get(exp_info.get('representations'))}", loc="left"
            )

            ################ HOUSEKEEPING ####################
            print(
                f"----------- SSA:\ny-axis: "
                f"{[np.float32(label) for label in axes[row, 1].get_yticks()]},\nx-axis: "
                f"{[np.log2(label) for label in axes[row, 1].get_xticks()]}"
            )
            print(
                f"----------- LCA:\ny-axis: "
                f"{[np.float32(label) for label in axes[row, 0].get_yticks()]},"
                f"\nx-axis: {[label for label in axes[row, 0].get_xticks()]}"
            )
            print(
                f"optimum_step_size: 2^{np.float32((np.log2(optimum_step_size)))}, "
                f"{exp_info.get('metric')} error: {np.float32(optimum_error)}"
            )

        y_tick_values = np.arange(
            exp_info.get("y_min"),
            exp_info.get("y_max") + exp_info.get("step"),
            exp_info.get("step"),
        ).astype(np.float32)
        for i in range(len(axes)):
            axes[row, i].spines["right"].set_visible(False)
            axes[row, i].spines["top"].set_visible(False)
            axes[row, i].set_yticks(y_tick_values)
            axes[row, i].set_yticklabels(y_tick_values)
            axes[row, i].set_ylim(exp_info.get("y_min"), exp_info.get("y_max"))

        plt.tight_layout()
        filename = f"{exp_info.get('num_states')}-RandomWalk"
        filename = filename.replace(".", "_")
        plt.savefig(save_path / filename)


if __name__ == "__main__":
    main()
