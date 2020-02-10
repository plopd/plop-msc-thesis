import argparse
import json
from pathlib import Path

from alphaex.sweeper import Sweeper

from utils.utils import remove_keys_with_none_value

parser = argparse.ArgumentParser()
parser.add_argument("--num_runs", type=int, default=1)
parser.add_argument("--representations", type=str)
parser.add_argument("--env", type=str)
parser.add_argument("--num_dims", type=int)
parser.add_argument("--num_features", type=int)
parser.add_argument("--interest", type=str)
parser.add_argument("--tilings", type=int)
parser.add_argument("--num_ones", type=int)
parser.add_argument("--step_size", type=float)
parser.add_argument("--lmbda", type=float)
parser.add_argument("--algorithm", type=str)
parser.add_argument("--order", type=float)
parser.add_argument("--config_filename", type=str, required=True)

args = parser.parse_args()


sweeper = Sweeper(Path(__file__).parents[1] / "configs" / f"{args.config_filename}")

search_dct = {
    "env": args.env,
    "representations": args.representations,
    "num_dims": args.num_dims,
    "num_features": args.num_features,
    "interest": args.interest,
    "tilings": args.tilings,
    "num_ones": args.num_ones,
    "step_size": args.step_size,
    "trace_decay": args.lmbda,
    "algorithm": args.algorithm,
    "order": args.order,
}

remove_keys_with_none_value(search_dct)

search_lst = sweeper.search(search_dct, args.num_runs)

for exp in search_lst:
    print(json.dumps(exp, indent=4))

print(
    f"Runs: {args.num_runs}, "
    f"Unique instances: {sweeper.total_combinations}, "
    f"Total instances: {sweeper.total_combinations*args.num_runs}, "
    f"Instances/Total: {len(search_lst)}/{sweeper.total_combinations}"
)
