from analysis.results import get_best_param_by
from analysis.results import Result

cfg_dir = "src/configs"
sweep_file_name = "chain.json"
path = "/Users/saipiens/repos/data-plop-msc-thesis"
exp = "chain_five"
num_runs = 4
algorithm = "td"

result = Result(f"{cfg_dir}/{sweep_file_name}", path, exp, num_runs)


lst_param_data = result.get_data_by_param(
    search_dct={"algorithm": algorithm, "features": "tabular"}, name="alpha"
)


best_param, best_val, best_data = get_best_param_by(lst_param_data, name="end")


print(algorithm, best_param, best_val, best_data)
