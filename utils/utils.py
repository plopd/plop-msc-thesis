import logging
import os

import numpy as np


def path_exists(path):
    if not path.exists():
        os.makedirs(path, exist_ok=True)

    return path


def normalize_to_unit(x):
    return x / np.linalg.norm(x, axis=1).reshape((-1, 1))


def emphasis_limit(interest, discount_rate, trace_decay):
    return (interest - discount_rate * trace_decay * interest) / (1 - discount_rate)


def minmax_normalization_ab(x, min_x, max_x, a, b):
    # https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between
    # -1-and-1
    x_norm = a + ((x - min_x) * (b - a)) / (max_x - min_x)

    return x_norm


def per_feature_step_size_fourier_KOT(step_size, num_features, C):
    step_size = step_size * np.ones(num_features)
    step_size[1:] /= np.sqrt(np.sum(np.square(C[1:, :]), axis=1))

    return step_size


def remove_keys_with_none_value(dct):
    """
    Remove any keys with value None
    Returns:

    """
    filtered = {k: v for k, v in dct.items() if v is not None}
    dct.clear()
    dct.update(filtered)

    return dct


def get_simple_logger(module_name, output_filepath):
    # create logger on the current module and set its level
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    logging.basicConfig(
        filename=output_filepath,
        filemode="w",
        format="%(asctime)s-%(levelname)s-%(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    return logger
