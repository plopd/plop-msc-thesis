import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from analysis.learning_curve import get_LCA
from analysis.results import Result
from analysis.sensitivity_curve import get_SSA
from analysis.waterfall import get_WF
from utils.utils import path_exists

matplotlib.rcParams.update({"font.size": 18})


def main(num_states, features, experiment, order, num_ones, n, runs):

    INTEREST = "uniform"
    ENVIRONMENT = "chain"
    RUNS = runs
    CUTOFF = 0.45
    PERFORMANCE = "end"
    DATA_PATH = "/home/plopd/scratch/Chain"
    METHODS = ["td", "etd"]
    RESULTS_PATH = f"{Path(__file__).parents[0]}/Chain"
    path_exists(RESULTS_PATH)

    fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharey="row", dpi=80)
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    for m in METHODS:
        result = Result(f"{experiment}.json", DATA_PATH, experiment, runs=RUNS)

        stepsize_search_dct = {
            "algorithm": m,
            "env": ENVIRONMENT,
            "features": features,
            "interest": INTEREST,
            "order": order,
            "n": n,
            "num_ones": num_ones,
        }
        filtered = {k: v for k, v in stepsize_search_dct.items() if v is not None}
        stepsize_search_dct.clear()
        stepsize_search_dct.update(filtered)

        print(stepsize_search_dct)

        ax1 = get_LCA(ax1, result, stepsize_search_dct, name=PERFORMANCE)
        ax2 = get_SSA(ax2, result, stepsize_search_dct, cutoff=CUTOFF, name=PERFORMANCE)
        ax3 = get_WF(
            ax3,
            result,
            stepsize_search_dct,
            cutoff=CUTOFF,
            name=PERFORMANCE,
            methods=METHODS,
        )
    plt.tight_layout()
    plt.savefig(
        f"{RESULTS_PATH}/Chain_{num_states}_{features}_{order}_{n}_{num_ones}_{INTEREST}_{PERFORMANCE}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_states", required=True, type=int)
    parser.add_argument("--features", required=True, type=str)
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--order", default=None, type=int)
    parser.add_argument("--n_ones", default=None, type=int)
    parser.add_argument("--n", default=None, type=int)
    parser.add_argument("--runs", default=100, type=int)

    args = parser.parse_args()

    main(
        num_states=args.n_states,
        features=args.features,
        experiment=args.experiment,
        order=args.order,
        num_ones=args.n_ones,
        n=args.n,
        runs=args.runs,
    )
