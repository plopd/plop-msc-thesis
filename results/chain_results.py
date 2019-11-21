import argparse
import copy
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from analysis.colormap import colors
from analysis.colormap import linestyles
from analysis.learning_curve import get_LCA
from analysis.results import Result
from analysis.sensitivity_curve import get_SSA
from analysis.waterfall import get_WF
from utils.utils import path_exists
from utils.utils import remove_keys_with_none_value

matplotlib.use("agg")
matplotlib.rcParams.update({"font.size": 18})


def main(kwargs):
    CUTOFF_ERR = 0.45
    interest = kwargs.get("interest", "uniform")
    environment = kwargs.get("env")
    experiment = kwargs.get("exp")
    experiment_bl = kwargs.get("exp_bl")
    features = kwargs.get("fts")
    order = kwargs.get("ord")
    in_features = kwargs.get("infts")
    num_ones = kwargs.get("one")
    num_states = kwargs.get("sts")
    n_runs = kwargs.get("rns")
    metric = kwargs.get("m")
    datapath = Path("~/scratch/Chain").expanduser()
    methods = kwargs.get("ms")
    lmbdas = kwargs.get("ls")
    RESULTS_PATH = Path(__file__).parents[0] / "Chain"
    path_exists(RESULTS_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey="row", dpi=80)

    for lmbda in lmbdas:
        for method in methods:
            result = Result(f"{experiment}.json", datapath, experiment)

            stepsize_search_dct = {
                "algorithm": method,
                "env": environment,
                "in_features": in_features,
                "features": features,
                "order": order,
                "num_ones": num_ones,
                "N": num_states,
                "lmbda": lmbda,
            }

            param_dict = {"n_runs": n_runs}

            # remove any keys with value None
            remove_keys_with_none_value(stepsize_search_dct)

            # pretty print criteria
            print(json.dumps(stepsize_search_dct, indent=4))

            # TODO How to combine experiments?
            if experiment_bl is not None:
                result_bl = Result(f"{experiment_bl}.json", datapath, experiment_bl)
                baseline_search_dct = copy.deepcopy(stepsize_search_dct)
                baseline_search_dct["algorithm"] = "lstd" if method == "td" else "elstd"
                print(json.dumps(baseline_search_dct, indent=4))
                idx_data = result_bl.find_experiment_by(baseline_search_dct, n_runs)
                data = result_bl.load(idx_data)
                if data is not None:
                    print("Adding baseline...")
                    data = data[:, -1]
                    m = data.mean()
                    axes[0].axhline(
                        m,
                        0,
                        50000,
                        color=colors[stepsize_search_dct["algorithm"]],
                        label=f"{stepsize_search_dct['algorithm']}-LS",
                        linestyle=linestyles[stepsize_search_dct["algorithm"]],
                    )

            get_LCA(axes[0], result, stepsize_search_dct, metric, **param_dict)
            get_SSA(
                axes[1], result, stepsize_search_dct, CUTOFF_ERR, metric, **param_dict
            )
            get_WF(
                axes[2],
                result,
                stepsize_search_dct,
                CUTOFF_ERR,
                metric,
                methods,
                **param_dict,
            )
            plt.tight_layout()
            plt.savefig(
                RESULTS_PATH / f"ChainStates{num_states}_"
                f"Lambda{str(lmbda).replace('.', '_')}_"
                f"Features{features.title()}_"
                f"Order{order}_"
                f"NumberFeaturesRandom{in_features}_"
                f"NumberOnesRandom{num_ones}_"
                f"Interest{interest.title()}Performance{metric.upper()}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sts", required=True, type=int)
    parser.add_argument("-fts", required=True, type=str)
    parser.add_argument("-exp", required=True, type=str)
    parser.add_argument("-exp_bl", type=str, help="Experiment name for baselines")
    parser.add_argument("-m", default="end", type=str)
    parser.add_argument("-ord", default=None, type=int)
    parser.add_argument("-one", default=None, type=int)
    parser.add_argument("-infts", default=None, type=int)
    parser.add_argument("-rns", default=100, type=int)
    parser.add_argument("-ls", nargs="+", type=float, default=[0.0])
    parser.add_argument("-ms", nargs="+", type=str, default=["td", "etd"])
    parser.add_argument("-env", type=str, default="chain")

    args = parser.parse_args()

    main(vars(args))
