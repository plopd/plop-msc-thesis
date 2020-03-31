from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from analysis.colormap import lambdas
from analysis.lineplot import lineplot
from analysis.results import Result
from analysis.SSA import ssa
from utils.decorators import timer
from utils.utils import path_exists

matplotlib.use("agg")
matplotlib.rcParams.update({"font.size": 24})


def main():
    plot_random_walk_features()


@timer
def plot_random_walk_features():
    experiment = "RW19GammaLambdaAlpha"
    num_runs = 100
    env = "RandomWalk"
    algorithms = ["TD", "ETD"]
    features = ["TA", "D", "IN"]
    trace_decays = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]
    gammas = [0.99, 0.9995]

    metrics = {
        "auc": {
            "percent_metric": 1.0,
            "lca_ylim_bottom": 0.0,
            "lca_ylim_top": 0.5,
            "lca_xlim_bottom": 0.0,
            "ssa_ylim_bottom": 0.0,
            "ssa_ylim_top": 0.5,
            "ssa_xlim_bottom": 2 ** -1,
            "ssa_xlim_top": 2,
        },
    }
    episodes = [1000]

    data_path = Path(f"~/scratch/{env}").expanduser()
    save_path = path_exists(Path(__file__).parents[0] / "plots")
    n_cols = 2
    n_rows = len(features)
    result = Result(experiment, data_path, num_runs)

    for metric, config in metrics.items():
        for episode in episodes:
            for gamma in gammas:
                fig, axs = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(n_cols * 5, n_rows * 4),
                    squeeze=False,
                    sharex="col",
                    sharey="all",
                    dpi=120,
                )
                for feature in features:
                    cutoff = result.data[:, 0].mean()
                    for algorithm in algorithms:
                        for trace_decay in trace_decays:
                            _config = {}
                            _config["trace_decay"] = trace_decay
                            _config["representations"] = feature
                            _config["algorithm"] = algorithm
                            step_sizes = result.get_value_param("step_size", _config)

                            means, standard_errors = ssa(
                                result,
                                _config,
                                step_sizes,
                                metric,
                                percent=config.get("percent_metric"),
                                cutoff=cutoff,
                            )
                            lineplot(
                                axs[0, 1],
                                step_sizes,
                                means,
                                standard_errors,
                                algorithm,
                                n_std=2.5,
                                color=lambdas.get(trace_decay),
                                marker="o",
                                show_legend=True,
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
                plt.savefig(save_path / f"19-RandomWalk-{gamma}-{episode}-{metric}")


if __name__ == "__main__":
    main()
