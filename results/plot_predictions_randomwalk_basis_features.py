import argparse
import copy
import json
from collections import OrderedDict
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_states", dest="num_states", type=int)
    parser.add_argument("--num_runs", dest="num_runs", type=int)
    parser.add_argument("--num_timesteps", dest="num_timesteps", type=int)
    parser.add_argument("--metric", dest="metric", type=str)
    parser.add_argument(
        "--yticks",
        dest="yticks",
        type=float,
        nargs="+",
        default=[0.4, 0.3, 0.2, 0.1, 0.0],
    )
    parser.add_argument("--order", dest="order", type=int, nargs="+")
    args = parser.parse_args()
    print(args.__dict__)

    num_states = args.num_states
    environment = "chain"
    order = args.order

    experiments = OrderedDict(
        {
            "poly": {"experiment": f"Chain{num_states}Poly"},
            # "random-binary": {
            #     "experiment": "Chain5Random",
            #     "params": {
            #         "num_dims": 4,
            #         "num_ones": 3
            #     },
            # },
            # "random-nonbinary": {
            #     "experiment": "Chain5Random",
            #     "params": {
            #         "num_dims": 4,
            #     },
            # },
        }
    )

    n_runs = args.num_runs
    timesteps = args.num_timesteps
    metric = args.metric
    yticks = args.yticks
    datapath = Path("~/scratch/Chain").expanduser()
    save_path = Path(__file__).parents[0]

    path_exists(save_path)

    fig, axes = plt.subplots(len(order), 3, figsize=(20, 15), sharey="all", dpi=80)
    for row, o in enumerate(order):
        for features, params in experiments.items():
            experiment = params["experiment"]
            for method in ["TD", "ETD"]:
                result = Result(f"{experiment}.json", datapath, experiment)

                stepsize_search_dct = {
                    "algorithm": method,
                    "env": environment,
                    "representations": features,
                    "order": o,
                    "N": num_states,
                }
                print(stepsize_search_dct)

                param_dict = {
                    "n_runs": n_runs,
                    "yticks": yticks,
                    "timesteps": timesteps,
                }

                # remove any keys with value None
                remove_keys_with_none_value(stepsize_search_dct)

                # pretty print criteria
                print(json.dumps(stepsize_search_dct, indent=4))

                experiment_baseline = "ChainLSTD"
                if experiment_baseline is not None:
                    result_bl = Result(
                        f"{experiment_baseline}.json", datapath, experiment_baseline
                    )
                    baseline_search_dct = copy.deepcopy(stepsize_search_dct)
                    baseline_search_dct["algorithm"] = (
                        "LSTD" if method == "TD" else "ELSTD"
                    )
                    print(json.dumps(baseline_search_dct, indent=4))
                    idx_data = result_bl.find_experiment_by(baseline_search_dct, n_runs)
                    data = result_bl.load(idx_data)
                    if data is not None:
                        print("Adding baseline...")
                        data = data[:, -1]
                        m = data.mean()
                        axes[row, 0].axhline(
                            m,
                            0,
                            timesteps,
                            color=colors[stepsize_search_dct["algorithm"]],
                            linestyle=linestyles[stepsize_search_dct["algorithm"]],
                        )

                get_LCA(axes[row, 0], result, stepsize_search_dct, metric, **param_dict)
                get_SSA(axes[row, 1], result, stepsize_search_dct, metric, **param_dict)
                get_WF(
                    axes[row, 2],
                    result,
                    stepsize_search_dct,
                    metric,
                    ["TD", "ETD"],
                    **param_dict,
                )
                plt.tight_layout()
                plt.savefig(save_path / f"Chain-{num_states}-{metric}-{features}")


if __name__ == "__main__":
    main()
