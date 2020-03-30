from pathlib import Path

import numpy as np
from alphaex.sweeper import Sweeper


class Result:
    def __init__(self, config_filename, datapath, num_runs):
        self.config_filename = config_filename
        self.datapath = datapath
        self.sweeper = Sweeper(
            Path(__file__).parents[1] / "configs" / f"{self.config_filename}.json"
        )
        self.num_experiments = self.sweeper.total_combinations * num_runs
        self.data = self._load(np.arange(0, self.num_experiments))
        self.num_episodes = self.data.shape[1]
        self.num_runs = num_runs

    def find_experiment_by(self, params):
        """
        Find all experiments which include `params` across `n_runs` runs.
        Args:
            params: dict,
            n_runs: int,

        Returns: list, sweeper indices

        """
        search_result_list = self.sweeper.search(params, self.num_runs)
        ids_experiments = []
        for search_result in search_result_list:
            ids_experiments = ids_experiments + search_result.get("ids")

        return list(set(ids_experiments))

    def get_value_param(self, name, config, n_runs=1):
        """
        Find values to parameter `name` by `config` across `runs`
        Args:
            name: str,
            config: dict,
            n_runs: int,
        Returns:

        """
        param_values = []
        search_result_list = self.sweeper.search(config, n_runs)
        for search_result in search_result_list:
            param_values.append(search_result.get(name))

        param_values = sorted(param_values)

        return param_values

    def _load(self, ids):
        data = []
        for idx in ids:
            filename = f"{idx}.npy"
            try:
                data.append(
                    np.load(self.datapath / f"{self.config_filename}" / filename)
                )
            except FileNotFoundError:
                print(f"File {filename} not found.")
        data = np.vstack(data)
        return data

    def get_data_auc(self, ids):
        auc_runs = self.data[ids, :].mean(axis=1)

        return auc_runs.mean(), (auc_runs.std() / np.sqrt(self.num_runs))

    def get_data_end(self, ids, percent):
        left = int(np.ceil(self.num_episodes * percent))
        end_data = self.data[ids, -left:]

        end_data = end_data.mean(axis=1)

        return end_data.mean(), (end_data.std() / np.sqrt(self.num_runs))

    def get_data_interim(self, ids, percent):
        right = int(np.ceil(self.num_episodes * percent))
        interim_data = self.data[ids, :right]

        interim_data = interim_data.mean(axis=1)

        return interim_data.mean(), (interim_data.std() / np.sqrt(self.num_runs))

    def get_data_by(self, name, ids, percent=None):
        if name == "end":
            return self.get_data_end(ids, percent)
        elif name == "interim":
            return self.get_data_interim(ids, percent)
        elif name == "auc":
            return self.get_data_auc(ids)
        raise Exception("Unknown name given.")
