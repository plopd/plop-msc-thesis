from pathlib import Path

import numpy as np
from alphaex.sweeper import Sweeper


class Result:
    def __init__(self, config_filename, datapath, experiment):
        self.config_filename = config_filename
        self.datapath = datapath
        self.experiment = experiment
        sweeper_config_file = (
            Path(__file__).parents[1] / "configs" / f"{self.config_filename}"
        )
        self.sweeper = Sweeper(sweeper_config_file)

    def find_experiment_by(self, params, n_runs):
        """
        Find all experiments which include `params` across `n_runs` runs.
        Args:
            params: dict,
            n_runs: int,

        Returns: list, sweeper indices

        """
        exps = self.sweeper.search(params, n_runs)
        ids_exp = []
        for ls in exps:
            ids_exp.extend(ls["ids"])

        return sorted(ids_exp)

    def find_experiment_by_idx(self, ids):
        exps = []
        for idx in ids:
            exp_config = self.sweeper.parse(idx)
            exps.append(exp_config)

        return exps

    def get_param_val(self, name, config, n_runs):
        """
        Find values to parameter `name` by `config` across `runs`
        Args:
            name: str,
            config: dict,
            n_runs: int,
        Returns:

        """
        param_vals = set()
        exps = self.sweeper.search(config, n_runs)
        for exp in exps:
            param_vals.add(exp[name])

        return param_vals

    def load(self, ids):
        data = []
        for idx in ids:
            try:
                data.append(
                    np.load(self.datapath / f"{self.experiment}" / f"{idx}.npy")
                )
            except FileNotFoundError:
                pass
        data = np.vstack(data)
        return data


def get_data_auc(data):
    n_runs, n_episodes = data.shape

    auc_runs = data.mean(axis=1)

    return auc_runs.mean(), auc_runs.std() / np.sqrt(n_runs)


def get_data_end(data, percent):
    n_runs, n_episodes = data.shape
    steps = int(n_episodes * percent)
    end_data = data[:, -steps:]

    end_data = end_data.mean(axis=1)

    return end_data.mean(), end_data.std() / np.sqrt(n_runs)


def get_data_interim(data, percent):
    n_runs, n_episodes = data.shape
    steps = int(n_episodes * percent)
    interim_data = data[:, :steps]

    interim_data = interim_data.mean(axis=1)

    return interim_data.mean(), interim_data.std() / np.sqrt(n_runs)


def get_data_by(data, name, percent=None):
    if name == "end":
        return get_data_end(data, percent)
    elif name == "interim":
        return get_data_interim(data, percent)
    elif name == "auc":
        return get_data_auc(data)
    raise Exception("Unknown name given.")
