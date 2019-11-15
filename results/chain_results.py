import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from analysis.learning_curve import get_LCA
from analysis.results import Result
from analysis.sensitivity_curve import get_SSA
from analysis.waterfall import get_WF
from utils.utils import path_exists

matplotlib.use("agg")
matplotlib.rcParams.update({"font.size": 18})


def main(kwargs):
    threshold_error = 0.45
    interest = kwargs.get("interest", "uniform")
    environment = kwargs.get("environment", "chain")
    experiment = kwargs.get("experiment")
    features = kwargs.get("features")
    order = kwargs.get("order")
    in_features = kwargs.get("in_features")
    num_ones = kwargs.get("ones")
    num_states = kwargs.get("states")
    num_runs = kwargs.get("runs")
    metric = kwargs.get("metric")
    datapath = Path("~/scratch/Chain").expanduser()
    methods = kwargs.get("methods")
    RESULTS_PATH = Path(__file__).parents[0] / "Chain"
    path_exists(RESULTS_PATH)

    fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharey="row", dpi=80)
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    for method in methods:
        result = Result(f"{experiment}.json", datapath, experiment, num_runs)

        stepsize_search_dct = {
            "algorithm": method,
            "env": environment,
            "in_features": in_features,
            "features": features,
            "order": order,
            "num_ones": num_ones,
            "N": num_states,
        }

        # remove any keys with value None
        filtered = {k: v for k, v in stepsize_search_dct.items() if v is not None}
        stepsize_search_dct.clear()
        stepsize_search_dct.update(filtered)

        # pretty print criteria
        print(json.dumps(stepsize_search_dct, indent=4))

        ax1 = get_LCA(ax1, result, stepsize_search_dct, metric)
        ax2 = get_SSA(ax2, result, stepsize_search_dct, threshold_error, metric)
        ax3 = get_WF(ax3, result, stepsize_search_dct, threshold_error, metric, methods)
    plt.tight_layout()
    plt.savefig(
        RESULTS_PATH
        / f"ChainStates{num_states}Features{features.title()}_Order{order}_"
        f"NumberFeaturesRandom{in_features}_NumberOnesRandom{num_ones}_"
        f"Interest{interest.title()}Performance{metric.upper()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--states", required=True, type=int)
    parser.add_argument("--features", required=True, type=str)
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--metric", default="end", type=str)
    parser.add_argument("--order", default=None, type=int)
    parser.add_argument("--ones", default=None, type=int)
    parser.add_argument("--in_features", default=None, type=int)
    parser.add_argument("--runs", default=100, type=int)
    parser.add_argument("--methods", nargs="+", type=str, default=["td", "etd"])

    args = parser.parse_args()
    main(vars(args))
