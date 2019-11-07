import sys
from pathlib import Path

from alphaex.sweeper import Sweeper


def test_sweeper(cfg_filename, num_runs):
    sweeper = Sweeper(f"{Path(__file__).parents[1]}/configs/{cfg_filename}")
    for sweep_id in range(sweeper.total_combinations * num_runs):
        rtn_dict = sweeper.parse(sweep_id)

        report = (
            "idx: %d\nrun: %d\nenv: %s\nN: %d\nalgorithm: %s\nalpha: "
            "%s\nfeatures: %s\ninterest: %s\norder: %s\nn: %s\nnum_ones: %s\n"
            % (
                sweep_id,
                rtn_dict.get("run", None),
                rtn_dict.get("env", None),
                rtn_dict.get("N", None),
                rtn_dict.get("algorithm", None),
                rtn_dict.get("alpha", None),
                rtn_dict.get("features", None),
                rtn_dict.get("interest", None),
                rtn_dict.get("order", None),
                rtn_dict.get("n", None),
                rtn_dict.get("num_ones", None),
            )
        )
        print(report)

    print(sweeper.total_combinations)

    # # test Sweeper.search
    sweeper.search(
        {"env": "chain", "algorithm": "td", "features": "random-binary"}, num_runs
    )


if __name__ == "__main__":
    cfg_filename = sys.argv[1]
    num_runs = int(sys.argv[2])
    test_sweeper(cfg_filename, num_runs)
