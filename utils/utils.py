import logging
import os

import numpy as np


def path_exists(path):
    if not path.exists():
        os.makedirs(path, exist_ok=True)

    return path


def emphasis_lim(interest, discount_rate, trace_decay):
    emphasis_lim = (interest - discount_rate * trace_decay * interest) / (
        1 - discount_rate
    )
    return emphasis_lim


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


def get_interest(name, **kwargs):

    N, seed = kwargs.get("num_states"), kwargs.get("seed")

    if name == "UI":
        return np.ones(N)
    elif name == "random-binary":
        np.random.seed(seed)
        random_array = np.random.choice([0, 1], N)
        return random_array

    raise Exception("Unexpected interest given.")


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
