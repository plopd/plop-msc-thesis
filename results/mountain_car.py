from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from analysis.colormap import color_algorithms
from analysis.LCA import lca
from analysis.lineplot import lineplot
from analysis.results import Result
from analysis.SSA import ssa
from utils.decorators import timer
from utils.utils import path_exists

matplotlib.use("agg")
matplotlib.rcParams.update({"font.size": 24})


def main():
    fig_mountain_car_lca_ssa_lambdas()


@timer
def fig_mountain_car_lca_ssa_lambdas():
    experiment = "MountainCar"
    num_runs = 50
    env = "MountainCar"
    algorithms = ["TDTileCoding", "ETDTileCoding"]
    metrics = {
        "interim": {
            "percent_metric": 0.0025,
            "lca_ylim_bottom": 0.0,
            "lca_ylim_top": None,
            "lca_xlim_bottom": 0.0,
            "lca_xlim_top": 100,
            "ssa_ylim_bottom": 0.0,
            "ssa_ylim_top": None,
            "ssa_xlim_bottom": None,
            "ssa_xlim_top": 2,
        },
        "auc": {
            "percent_metric": 1.0,
            "lca_ylim_bottom": 0.0,
            "lca_ylim_top": None,
            "lca_xlim_bottom": 0.0,
            "ssa_ylim_bottom": 0.0,
            "ssa_ylim_top": None,
            "ssa_xlim_bottom": None,
            "ssa_xlim_top": 2,
        },
        "end": {
            "percent_metric": 0.0025,
            "lca_ylim_bottom": 0.0,
            "lca_ylim_top": None,
            "lca_xlim_bottom": 0.0,
            "ssa_ylim_bottom": 0.0,
            "ssa_ylim_top": None,
            "ssa_xlim_top": 2,
        },
    }
    lambdas = [0.0]

    data_path = Path(f"/Volumes/REINFORCE/Plop-Msc-2020/{env}").expanduser()
    save_path = path_exists(Path(__file__).parents[0] / "plots")
    n_cols = 2
    n_rows = len(lambdas)

    for metric, config in metrics.items():
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 5, n_rows * 4),
            squeeze=False,
            sharex="col",
            sharey="all",
            dpi=120,
        )
        for row, lmbda in enumerate(lambdas):
            result = Result(experiment, data_path, num_runs)
            cutoff = result.data[:, 0].mean()
            for algorithm in algorithms:
                _config = {}
                _config["trace_decay"] = lmbda
                _config["algorithm"] = algorithm
                step_sizes = result.get_value_param("step_size", _config)

                mean, standard_error, opt_step_size = lca(
                    result,
                    _config,
                    step_sizes,
                    metric,
                    percent=config.get("percent_metric"),
                )
                lineplot(
                    axs[row, 0],
                    np.arange(len(mean)),
                    mean,
                    standard_error,
                    opt_step_size,
                    n_std=2.5,
                    color=color_algorithms.get(algorithm),
                    show_legend=False,
                    xscale={"value": "linear", "base": 10},
                    ylim={
                        "bottom": config.get("lca_ylim_bottom"),
                        "top": config.get("lca_ylim_top"),
                    },
                    xlim={
                        "bottom": config.get("lca_xlim_bottom"),
                        "top": config.get("lca_xlim_top"),
                    },
                )

                means, standard_errors = ssa(
                    result,
                    _config,
                    step_sizes,
                    metric,
                    percent=config.get("percent_metric"),
                    cutoff=cutoff,
                )
                lineplot(
                    axs[row, 1],
                    step_sizes,
                    means,
                    standard_errors,
                    algorithm,
                    n_std=2.5,
                    color=color_algorithms.get(algorithm),
                    marker="o",
                    show_legend=False,
                    xscale={"value": "log", "base": 2},
                    ylim={
                        "bottom": config.get("ssa_ylim_bottom"),
                        "top": config.get("ssa_ylim_top"),
                    },
                    xlim={
                        "bottom": config.get("ssa_xlim_bottom"),
                        "top": config.get("ssa_xlim_top"),
                    },
                )

        for row in range(n_rows):
            for col in range(n_cols):
                axs[row, col].spines["right"].set_visible(False)
                axs[row, col].spines["top"].set_visible(False)

        fig.tight_layout()
        plt.savefig(save_path / f"MountainCar-{metric}")


if __name__ == "__main__":
    main()
