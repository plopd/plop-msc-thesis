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
from utils.utils import path_exists
from utils.utils import remove_keys_with_none_value

# from utils.utils import emphasis_lim

matplotlib.use("agg")
matplotlib.rcParams.update({"font.size": 18})


def main():
    sweep_id = int(sys.argv[1].strip(","))
    config_filename = sys.argv[2]

    sweeper = Sweeper(Path(__file__).parents[1] / "configs" / f"{config_filename}.json")

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
        "algos": ["TD", "ETD"],
    }

    datapath = Path(f"~/scratch/{exp_info.get('env')}").expanduser()
    save_path = path_exists(Path(__file__).parents[0] / exp_info.get("experiment"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey="all")

    results = Result(
        f"{exp_info.get('experiment')}.json", datapath, exp_info.get("experiment")
    )

    config = exp_info.copy()
    config.pop("experiment", None)
    config.pop("algos", None)
    config.pop("metric", None)
    config.pop("num_runs", None)
    config.pop("baseline", None)
    config.pop("run", None)
    remove_keys_with_none_value(config)

    print(json.dumps(config, indent=4))

    for algo in exp_info.get("algos"):
        print(f"\n##### {algo}")
        config["algorithm"] = algo
        config.pop("step_size", None)
        step_sizes = results.get_param_val(
            "step_size", config, exp_info.get("num_runs")
        )
        step_sizes = sorted(step_sizes)
        means = []
        std_errors = []
        xs = []
        optim_step_size = None
        current_optim_err = np.inf

        for step_size in step_sizes:
            config["step_size"] = step_size
            ids = results.find_experiment_by(config, exp_info.get("num_runs"))
            data = results.load(ids)
            mean, se = get_data_by(data, name=exp_info.get("metric"))
            cutoff = data[:, 0].mean()
            # Find best instance of algorithm
            if mean < current_optim_err:
                current_optim_err = mean
                optim_step_size = step_size
            # Record step-size sensitivity
            if mean <= cutoff:
                means.append(mean)
                std_errors.append(se)
            else:
                means.append(cutoff)
                std_errors.append(0.0)
            xs.append(step_size)

        # if algo == "ETD":
        #     xs = np.array(xs) / emphasis_lim(1, args.discount_rate, args.trace_decay)
        means = np.array(means)
        std_errors = np.array(std_errors)

        axes[1].scatter(xs, means, c=colors.get(config.get("algorithm")), marker="o")
        axes[1].errorbar(
            xs,
            means,
            yerr=2.5 * std_errors,
            color=colors.get(config.get("algorithm")),
            capsize=5,
        )
        axes[1].tick_params(axis="x", colors=colors.get(config.get("algorithm")))
        axes[1].set_xscale("log", basex=2)
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["top"].set_visible(False)
        axes[1].tick_params(labelbottom=False, labelleft=False)
        axes[1].set_yticks([0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
        print(
            f"----------- SSA:\ny-axis: "
            f"{[np.float32(label) for label in axes[1].get_yticks()]},\nx-axis: "
            f"{[np.log2(label) for label in axes[1].get_xticks()]}"
        )

        config["step_size"] = optim_step_size
        ids = results.find_experiment_by(config, exp_info.get("num_runs"))
        data = results.load(ids)

        mean = data.mean(axis=0)
        se = data.std(axis=0) / np.sqrt(exp_info.get("num_runs"))

        if exp_info.get("metric") == "interim":
            steps = int(len(mean) * 0.1)
            mean = mean[:steps]
            se = se[:steps]
        axes[0].plot(mean, c=colors.get(config.get("algorithm")))
        axes[0].fill_between(
            np.arange(len(mean)),
            mean + 2.5 * se,
            mean - 2.5 * se,
            color=colors.get(config.get("algorithm")),
            alpha=0.15,
        )
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["top"].set_visible(False)
        # Hide ticklabeles
        axes[0].tick_params(labelbottom=False, labelleft=False)
        axes[0].set_yticks([0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])

        print(
            f"----------- LCA:\ny-axis: {[np.float32(label) for label in axes[0].get_yticks()]},"
            f"\nx-axis: {[label for label in axes[0].get_xticks()]}"
        )
        print(f"optim_step_size: 2^{np.float32((np.log2(optim_step_size)))}")

        if exp_info.get("baseline") == 1:
            config["algorithm"] = "LSTD" if algo == "TD" else "ELSTD"
            config.pop("step_size", None)
            ids = results.find_experiment_by(config, exp_info.get("num_runs"))
            data = results.load(ids)
            data = data[:, -1]
            mean = data.mean()

            axes[0].axhline(
                mean,
                xmin=0,
                xmax=len(data),
                color=colors.get(algo),
                linestyle=linestyles.get(algo),
            )

        plt.tight_layout()
        filename = "-".join(
            list(
                filter(
                    "".__ne__,
                    [
                        f"{kw}_{val}" if val is not None else ""
                        for (kw, val) in exp_info.items()
                    ],
                )
            )
        )
        filename = filename.replace(".", "_")
        plt.savefig(save_path / filename)


if __name__ == "__main__":
    main()
