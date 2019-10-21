from analysis.results import Result

sweep_file_name = "test_chain.json"
datapath = "/Users/saipiens/repos/data-plop-msc-thesis"
experiment = "chain_five"
num_runs = 4


def test_find_param_by():
    result = Result(sweep_file_name, datapath, experiment, num_runs)

    print(
        result.find_experiment_by(
            {"algorithm": "td", "env": "chain", "features": "tabular"}
        )
    )


if __name__ == "__main__":
    test_find_param_by()
