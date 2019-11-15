from pathlib import Path

import numpy as np
from alphaex.sweeper import Sweeper


class Result(object):
    def __init__(self, config_filepath, datapath, experiment, runs=100):
        self.cfg = config_filepath
        self.datapath = datapath
        self.exp = experiment
        self.runs = runs
        self.sweeper = Sweeper(Path(__file__).parents[1] / "configs" / f"{self.cfg}")

    def find_experiment_by(self, params):
        """
        Find all experiments which include `params`
        Args:
            params: dict

        Returns: list, sweeper indices

        """
        lst_experiments = self.sweeper.search(params, self.runs)
        ids_experiments = []
        for ls in lst_experiments:
            ids_experiments.extend(ls["ids"])

        return sorted(ids_experiments)

    def find_experiment_by_idx(self, idxs):
        experiments = []
        for idx in idxs:
            rtn_dict = self.sweeper.parse(idx)
            experiments.append(rtn_dict)

        return experiments

    def get_param_val(self, name, criteria):
        """
        Find values to parameter with `name`
        Args:
            name: str,
            criteria: dict,

        Returns:

        """
        param_vals = set()
        lst_experiments = self.sweeper.search(criteria, self.runs)
        for exp in lst_experiments:
            param_vals.add(exp[name])

        return param_vals

    def load(self, idxs):
        lst_data = []
        for idx in idxs:
            try:
                lst_data.append(
                    np.load(self.datapath / f"{self.exp}" / f"{idx}_msve.npy")
                )
            except IOError:
                continue

        try:
            data = np.vstack(lst_data)
        except ValueError:
            return None
        return data


def get_data_auc(data):
    n_runs, n_episodes = data.shape

    auc_runs = data.mean(axis=1)

    return auc_runs.mean(), auc_runs.std() / np.sqrt(n_runs)


def get_data_end(data):
    n_runs, n_episodes = data.shape
    steps = int(n_episodes * 0.01)
    end_data = data[:, -steps:]

    end_data = end_data.mean(axis=1)

    return end_data.mean(), end_data.std() / np.sqrt(n_runs)


def get_data_by(data, name="end"):
    if name == "end":
        return get_data_end(data)
    elif name == "interim":
        raise NotImplementedError
    elif name == "auc":
        return get_data_auc(data)
    raise Exception("Unknown name given.")
