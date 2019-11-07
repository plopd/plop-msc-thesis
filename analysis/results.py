from pathlib import Path

import numpy as np
from alphaex.sweeper import Sweeper


class Result(object):
    def __init__(self, cfg, datapath, exp, runs=100):
        self.cfg = cfg
        self.datapath = datapath
        self.exp = exp
        self.runs = runs
        self.sweeper = Sweeper(f"{Path(__file__).parents[1]}/configs/{cfg}")

    def find_experiment_by(self, params):
        """
        Find all experiments which include `params`
        Args:
            params: dct

        Returns: list, sweeper indices

        """
        lst_experiments = self.sweeper.search(params, num_runs=self.runs)
        ids_experiments = []
        for ls in lst_experiments:
            ids_experiments.extend(ls["ids"])

        return sorted(ids_experiments)

    def find_experiment_by_idx(self, idcs):
        experiments = []
        for idx in idcs:
            rtn_dict = self.sweeper.parse(idx)
            experiments.append(rtn_dict)

        return experiments

    def get_param_val(self, name, search_dct={}):
        """
        Find values to parameter with `name`
        Args:
            name:
            search_dct:

        Returns:

        """
        param_vals = set()
        lst_experiments = self.sweeper.search(search_dct, num_runs=self.runs)
        for exp in lst_experiments:
            param_vals.add(exp[name])

        return param_vals

    def load(self, idxs):
        lst_data = []
        for idx in idxs:
            try:
                lst_data.append(
                    np.load(
                        f"{self.datapath}/{self.exp}/{idx}_msve.npy", allow_pickle=True
                    )
                )
            except IOError:
                continue

        data = None
        if lst_data:
            data = np.stack(lst_data)

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
