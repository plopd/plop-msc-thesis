# from analysis.results import Result
#
# sweep_file_name = "Chain19RandomBinary.json"
# datapath = "/Users/saipiens/repos/data-plop-msc-thesis"
# experiment = "chain_five"
# num_runs = 100
#
#
# def test_find_param_by():
#     result = Result(sweep_file_name, datapath, experiment, num_runs)
#
#     src_dct = result.find_experiment_by(
#         {"algorithm": "td", "env": "chain", "features": "random-binary", "alpha": 0.125}
#     )
#     print(len(src_dct))
#
#     experiments = result.find_experiment_by_idx(src_dct)
#
#     print(experiments)
#
#
# if __name__ == "__main__":
#     test_find_param_by()
