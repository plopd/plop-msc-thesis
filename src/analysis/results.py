from pathlib import Path

import numpy as np
from alphaex.sweeper import Sweeper


class Result(object):
    def __init__(self, cfg, path, exp, runs):
        self.cfg = cfg
        self.path = path
        self.exp = exp
        self.runs = runs
        self.sweeper = Sweeper(f"{Path(__file__).parents[2]}/{cfg}")

    def find_exp_by(self, params):
        lst_search = self.sweeper.search(params, num_runs=self.runs)
        idxs = []
        for ls in lst_search:
            idxs.extend(ls["ids"])

        return sorted(idxs)

    def _load(self, idxs):
        lst_data = []
        for idx in idxs:
            lst_data.append(
                np.load(f"{self.path}/{self.exp}/{idx}_error.npy", allow_pickle=True)
            )
        data = np.stack(lst_data)
        return data

    def get_data_by_param(self, search_dct, name):
        params = find_param_by(self.sweeper.search(search_dct, self.runs), name)

        lst_data_param = []
        for param in params:
            data_idxs = self.find_exp_by(search_dct)
            data = self._load(data_idxs)
            lst_data_param.append((param, data))

        return lst_data_param


def find_param_by(search_lst, name):
    param_set = set()
    for dct in search_lst:
        param_set.add(dct.get(name, None))

    return param_set


def get_best_end(data, fraction_end=0.1):
    best_val = np.inf
    best_param = None
    best_data = None

    for param_data in data:
        steps = int(param_data[1].shape[1] * fraction_end)
        val = np.mean(np.mean(param_data[1][:, -steps:], axis=1))
        if val <= best_val:
            best_val = val
            best_param = param_data[0]
            best_data = param_data[1]

    return (best_param, best_data)


def get_best_param_by(data, name="end"):

    if name == "end":
        best_param, best_data = get_best_end(data)
        return best_param, best_data

    raise Exception("Unknown name was given")
