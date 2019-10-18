from analysis.results import find_param_by
from analysis.results import Result

cfg_dir = "src/configs"
sweep_file_name = "chain.json"
path = "/home/plopd/scratch"
exp = "chain_five"
num_runs = 4


def test_find_param_by():
    result = Result(f"{cfg_dir}/{sweep_file_name}", path, exp, num_runs)

    print(result.find_exp_by({"algorithm": "td", "features": "tabular", "alpha": 0.5}))

    print(find_param_by(result.sweeper.search({"algorithm": "td"}, num_runs), "alpha"))


if __name__ == "__main__":
    test_find_param_by()
