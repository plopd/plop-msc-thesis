import json
import sys
from pathlib import Path

from alphaex.sweeper import Sweeper


def test_sweeper(cfg_filename, num_runs):
    sweeper = Sweeper(f"{Path(__file__).parents[1]}/configs/{cfg_filename}")
    for sweep_id in range(sweeper.total_combinations * num_runs):
        rtn_dict = sweeper.parse(sweep_id)

        print(f"ID: {sweep_id}", json.dumps(rtn_dict, indent=8))

    print(f"-------------\nTotal sweeping combinations: {sweeper.total_combinations}")

    src_lst = sweeper.search(
        {
            # "v_min": 0,
            # "v_max": 19
            "algorithm": "lstd",
            "N": 5,
            "env": "chain",
            "features": "dependent",
            "interest": "uniform",
        },
        num_runs,
    )

    for src in src_lst:
        print(json.dumps(src, indent=4))


if __name__ == "__main__":
    cfg_filename = sys.argv[1]
    num_runs = int(sys.argv[2])
    test_sweeper(cfg_filename, num_runs)
