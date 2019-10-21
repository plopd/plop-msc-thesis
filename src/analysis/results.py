from pathlib import Path

import numpy as np
from alphaex.sweeper import Sweeper


class Result(object):
    def __init__(self, cfg, datapath, exp, runs):
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

    def get_param_val(self, name, search_dct={}):
        """
        Find values to parameter with `name`
        Args:
            name:
            search_dct:

        Returns:

        """

    def _load(self, idxs):
        lst_data = []
        for idx in idxs:
            lst_data.append(
                np.load(
                    f"{self.datapath}/{self.exp}/{idx}_error.npy", allow_pickle=True
                )
            )
        data = np.stack(lst_data)
        return data
