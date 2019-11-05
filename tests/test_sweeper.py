import sys
from pathlib import Path

from alphaex.sweeper import Sweeper


def test_sweeper(cfg_filename):
    sweep_file_name = f"{cfg_filename}.json"
    num_runs = 1
    # test Sweeper.parse
    sweeper = Sweeper(f"{Path(__file__).parents[1]}/configs/{sweep_file_name}")
    for sweep_id in range(0, sweeper.total_combinations * num_runs):
        rtn_dict = sweeper.parse(sweep_id)

        report = (
            "idx: %d\nrun: %d\nenv: %s\nN: %d\nalgorithm: %s\nalpha: "
            "%s\nfeatures: %s\ninterest: %s\norder: %s\n"
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
            )
        )
        print(report)

    print(sweeper.total_combinations)

    # # test Sweeper.search
    sweeper.search({"env": "chain", "algorithm": "etd"}, num_runs)
    # print(len(search_dct))


if __name__ == "__main__":
    cfg_filename = sys.argv[1]
    test_sweeper(cfg_filename)
