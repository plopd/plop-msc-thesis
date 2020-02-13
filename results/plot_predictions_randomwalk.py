import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from analysis.colormap import colors
from analysis.results import get_data_by
from analysis.results import Result
from utils.utils import path_exists
from utils.utils import remove_keys_with_none_value

matplotlib.use("agg")
matplotlib.rcParams.update({"font.size": 18})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_states", dest="num_states", type=int)
    parser.add_argument("--num_runs", dest="num_runs", type=int)
    parser.add_argument("--order", dest="order", type=int)
    parser.add_argument("--metric", dest="metric", type=str)
    parser.add_argument("--experiment", dest="experiment", type=str)
    parser.add_argument("--env", dest="env", type=str)
    parser.add_argument("--representations", dest="representations", type=str)
    parser.add_argument("--num_ones", dest="num_ones", type=int)
    parser.add_argument("--num_features", dest="num_features", type=int)
    parser.add_argument("--num_dims", dest="num_dims", type=int)
    parser.add_argument("--interest", dest="interest", type=str, default="UI")

    parser.add_argument("--algos", dest="algos", nargs="+", default=["TD", "ETD"])
    args = parser.parse_args()

    datapath = Path(f"~/scratch/{args.env}").expanduser()
    save_path = path_exists(Path(__file__).parents[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey="all")

    results = Result(f"{args.experiment}.json", datapath, args.experiment)

    config = args.__dict__.copy()
    config.pop("experiment", None)
    config.pop("algos", None)
    config.pop("metric", None)
    config.pop("num_runs", None)
    remove_keys_with_none_value(config)
    print(json.dumps(config, indent=4))

    for algo in args.algos:
        config["algorithm"] = algo
        config.pop("step_size", None)
        step_sizes = results.get_param_val("step_size", config, args.num_runs)
        step_sizes = sorted(step_sizes)
        means = []
        std_errors = []
        xs = []
        optim_step_size = None
        current_optim_err = np.inf
        for step_size in step_sizes:
            config["step_size"] = step_size
            ids = results.find_experiment_by(config, args.num_runs)
            data = results.load(ids)
            mean, se = get_data_by(data, name=args.metric)
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
        means = np.array(means)
        std_errors = np.array(std_errors)

        axes[1].plot(xs, means, c=colors.get(config.get("algorithm")), marker="o", ms=3)
        axes[1].errorbar(
            xs,
            means,
            yerr=2.5 * std_errors,
            color=colors.get(config.get("algorithm")),
            capsize=5,
            fmt="none",
        )

        axes[1].set_xscale("log", basex=2)
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["top"].set_visible(False)

        config["step_size"] = optim_step_size
        print(f"{algo}, {optim_step_size}")
        ids = results.find_experiment_by(config, args.num_runs)
        data = results.load(ids)

        mean = data.mean(axis=0)

        axes[0].plot(mean, c=colors.get(config.get("algorithm")))

        axes[0].spines["right"].set_visible(False)
        axes[0].spines["top"].set_visible(False)

        plt.tight_layout()
        filename = "-".join(
            list(
                filter(
                    "".__ne__,
                    [
                        f"{kw}_{val}" if val is not None else ""
                        for (kw, val) in args.__dict__.items()
                    ],
                )
            )
        )
        plt.savefig(save_path / filename)


if __name__ == "__main__":
    main()
