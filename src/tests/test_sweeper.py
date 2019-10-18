import os

from alphaex.sweeper import Sweeper


def test_sweeper():
    cfg_dir = "configs/"
    sweep_file_name = "chain.json"
    num_runs = 4
    # test Sweeper.parse
    sweeper = Sweeper(os.path.join(cfg_dir, sweep_file_name))
    for sweep_id in range(0, sweeper.total_combinations * num_runs):
        rtn_dict = sweeper.parse(sweep_id)

        report = (
            "idx: %d \nrun: %d\nenv: %s\nalgorithm: %s\nalpha: "
            "%s\nfeatures: %s\n"
            % (
                sweep_id,
                rtn_dict.get("run", None),
                rtn_dict.get("env", None),
                rtn_dict.get("algorithm", None),
                rtn_dict.get("alpha", None),
                rtn_dict.get("features", None),
            )
        )
        print(report)

    print(f"# unique param settings: {sweeper.total_combinations}\n")

    # test Sweeper.search
    print(
        sweeper.search(
            {"env": "chain", "algorithm": "td", "features": "dependent"}, num_runs
        )
    )


if __name__ == "__main__":
    test_sweeper()
