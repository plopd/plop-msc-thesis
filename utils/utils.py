import logging
import os

import numpy as np

from utils.features import get_bases_feature
from utils.features import get_dependent_feature
from utils.features import get_feature_state_aggregation
from utils.features import get_inverted_feature
from utils.features import get_random_features
from utils.features import get_tabular_feature


def path_exists(path):
    if not path.exists():
        os.makedirs(path, exist_ok=True)


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

    N, seed = kwargs.get("N"), kwargs.get("seed")

    if name == "uniform":
        return np.ones(N)
    elif name == "random-binary":
        np.random.seed(seed)
        random_array = np.random.choice([0, 1], N)
        return random_array

    raise Exception("Unexpected interest given.")


def get_feature(x, unit_norm=True, **kwargs):
    """ Construct various features from states.

    Args:
        x: ndarray, shape (k,)
        name: str,
        unit_norm: (boolean),

    Returns:

    """

    name = kwargs.get("features")
    order = kwargs.get("order")
    in_features = kwargs.get("in_features")
    num_ones = kwargs.get("num_ones", 0)
    seed = kwargs.get("seed")
    v_min = kwargs.get("v_min")
    v_max = kwargs.get("v_max")

    if name == "tabular":
        return get_tabular_feature(x, in_features)
    elif name == "inverted":
        return get_inverted_feature(x, in_features, unit_norm)
    elif name == "dependent":
        return get_dependent_feature(x, in_features, unit_norm)
    elif name == "poly" or name == "fourier":
        return get_bases_feature(x, name, order, in_features, v_min, v_max, unit_norm)
    elif name == "random-binary" or name == "random-nonbinary":
        return get_random_features(x, name, in_features, num_ones, seed, unit_norm)
    elif name == "SA":
        return get_feature_state_aggregation(x, in_features, seed, unit_norm)
    raise Exception("Unexpected name given.")


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
