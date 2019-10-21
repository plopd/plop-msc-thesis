from analysis.results import Result

cfg_dir = "src/configs"
sweep_file_name = "chain.json"
path = "/Users/saipiens/repos/data-plop-msc-thesis"
exp = "chain"
num_runs = 4
algorithm = "td"

result = Result(f"{cfg_dir}/{sweep_file_name}", path, exp, num_runs)
