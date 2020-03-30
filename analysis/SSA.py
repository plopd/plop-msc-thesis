import copy

import numpy as np


def ssa(result, config, step_sizes, name, cutoff=None, percent=None):
    _config = copy.deepcopy(config)
    num_step_sizes = len(step_sizes)
    means = np.zeros(num_step_sizes)
    standard_errors = np.zeros(num_step_sizes)

    for i, step_size in enumerate(step_sizes):
        _config["step_size"] = step_size
        ids = result.find_experiment_by(_config)
        mean, standard_error = result.get_data_by(name, ids, percent)
        means[i] = mean
        means = np.nan_to_num(means, nan=np.inf)
        means = means.clip(0, cutoff)
        standard_errors[i] = standard_error
        standard_errors[np.where(means == cutoff)[0]] = 0

    return means, standard_errors
