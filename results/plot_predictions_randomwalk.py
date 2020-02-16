import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from alphaex.sweeper import Sweeper
from tqdm import tqdm

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

    # ax2 = axes[1].twiny()
    for algo in exp_info.get("algos"):
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

        for step_size in tqdm(step_sizes):
            config["step_size"] = step_size
            ids = results.find_experiment_by(config, exp_info.get("num_runs"))
            data = results.load(ids)
            mean, se = get_data_by(data, name=exp_info.get("metric"))
            cutoff = data[:, 0].mean()
            # print(algo, step_size, mean, cutoff)
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

        # ax = ax2 if algo == "ETD" else axes[1]

        ax = axes[1]

        ax.scatter(xs, means, c=colors.get(config.get("algorithm")), marker="o")
        ax.errorbar(
            xs,
            means,
            yerr=2.5 * std_errors,
            color=colors.get(config.get("algorithm")),
            capsize=5,
        )

        ax.tick_params(axis="x", colors=colors.get(config.get("algorithm")))

        # axes[1].set_xticks(
        #     [3.0517578125e-06, 2.44140625e-05, 0.0001953125, 0.0015625, 0.0125, 0.1])
        # axes[1].set_xticklabels(
        #     [3.0517578125e-06, 2.44140625e-05, 0.0001953125, 0.0015625, 0.0125, 0.1])

        ax.set_xscale("log", basex=2)
        # axes[1].spines["right"].set_visible(False)
        # axes[1].spines["top"].set_visible(False)

        config["step_size"] = optim_step_size
        ids = results.find_experiment_by(config, exp_info.get("num_runs"))
        data = results.load(ids)

        mean = data.mean(axis=0)

        if exp_info.get("metric") == "interim":
            steps = int(len(mean) * 0.1)
            mean = mean[:steps]
        axes[0].plot(
            mean,
            c=colors.get(config.get("algorithm")),
            label=f"{algo}, {int(np.log10(optim_step_size))}",
        )

        axes[0].legend()

        axes[0].spines["right"].set_visible(False)
        axes[0].spines["top"].set_visible(False)

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
