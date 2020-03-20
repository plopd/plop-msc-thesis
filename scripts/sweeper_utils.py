import argparse
from pathlib import Path

from alphaex.sweeper import Sweeper

from utils.utils import remove_keys_with_none_value

parser = argparse.ArgumentParser()
parser.add_argument("--num_runs", type=int)
parser.add_argument("--num_states", type=int)
parser.add_argument("--representations", type=str)
parser.add_argument("--env", type=str)
parser.add_argument("--num_dims", type=int)
parser.add_argument("--num_features", type=int)
parser.add_argument("--interest", type=str)
parser.add_argument("--tilings", type=int)
parser.add_argument("--num_ones", type=int)
parser.add_argument("--step_size", type=float)
parser.add_argument("--trace_decay", type=float)
parser.add_argument("--discount_rate", type=float)
parser.add_argument("--algorithm", type=str)
parser.add_argument("--metric", type=str)
parser.add_argument("--order", type=float)
parser.add_argument("--config_filename", type=str, required=True)

args = parser.parse_args()


sweeper = Sweeper(
    Path(__file__).parents[1] / "configs" / f"{args.config_filename}.json"
)

search_dct = {
    "env": args.env,
    "num_states": args.num_states,
    "representations": args.representations,
    "num_dims": args.num_dims,
    "num_features": args.num_features,
    "interest": args.interest,
    "tilings": args.tilings,
    "num_ones": args.num_ones,
    "step_size": args.step_size,
    "trace_decay": args.trace_decay,
    "discount_rate": args.discount_rate,
    "algorithm": args.algorithm,
    "order": args.order,
    "metric": args.metric,
}

remove_keys_with_none_value(search_dct)

search_lst = sweeper.search(search_dct, args.num_runs)

lst_indices = []
for dct in search_lst:
    print(dct.get("ids"), dct)

print(
    f"Number of runs: {args.num_runs},\n"
    f"Total number of combinations (per run): {sweeper.total_combinations},\n"
    f"Found combinations / Total number of combinations (per run): {len(search_lst)}/{sweeper.total_combinations}, {lst_indices}\n"
    f"Total number of combinations (over all runs): {sweeper.total_combinations*args.num_runs},\n"
)
