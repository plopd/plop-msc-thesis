import argparse
import json
from pathlib import Path

from alphaex.sweeper import Sweeper

from utils.utils import remove_keys_with_none_value

parser = argparse.ArgumentParser()
parser.add_argument("--num_runs", type=int, default=1)
parser.add_argument("--features", type=str)
parser.add_argument("--env", type=str)
parser.add_argument("--num_dims", type=int)
parser.add_argument("--interest", type=str)
parser.add_argument("--tilings", type=int)
parser.add_argument("--num_ones", type=int)
parser.add_argument("--alpha", type=float)
parser.add_argument("--lmbda", type=float)
parser.add_argument("--algorithm", type=str)
parser.add_argument("--order", type=float)
parser.add_argument("--config_filename", type=str, required=True)

args = parser.parse_args()


sweeper = Sweeper(Path(__file__).parents[1] / "configs" / f"{args.config_filename}")
print(f"Sweeping combinations: {sweeper.total_combinations}")
print(f"Total sweeping combinations: {sweeper.total_combinations * args.num_runs}")

search_dct = {
    "env": args.env,
    "features": args.features,
    "num_dims": args.num_dims,
    "interest": args.interest,
    "tilings": args.tilings,
    "num_ones": args.num_ones,
    "alpha": args.alpha,
    "lmbda": args.lmbda,
    "algorithm": args.algorithm,
    "order": args.order,
}

remove_keys_with_none_value(search_dct)

search_lst = sweeper.search(search_dct, args.num_runs)

for exp in search_lst:
    print(json.dumps(exp, indent=4))
