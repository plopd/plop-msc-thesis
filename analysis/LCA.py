import copy

import numpy as np


def lca(result, config, step_sizes, name, percent=None):
    _config = copy.deepcopy(config)
    optimum_step_size = None
    optimum_error = np.inf

    for i, step_size in enumerate(step_sizes):
        _config["step_size"] = step_size
        ids = result.find_experiment_by(_config)

        mean, standard_error = result.get_data_by(name, ids, percent)

        if mean <= optimum_error:
            optimum_error = mean
            optimum_step_size = step_size

    _config["step_size"] = optimum_step_size
    ids = result.find_experiment_by(_config)
    data = result.data[ids]
    mean = data.mean(axis=0)
    standard_error = data.std(axis=0) / np.sqrt(data.shape[0])

    return mean, standard_error, optimum_step_size
