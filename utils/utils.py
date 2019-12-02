import os

import numpy as np

from utils.features import get_bases_feature
from utils.features import get_dependent_feature
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
    num_states = kwargs.get("N")
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
        features = get_random_features(
            num_states, name, in_features, num_ones, seed, unit_norm
        )
        return features[x][0]
    raise Exception("Unexpected name given.")


def get_chain_states(N):
    return np.arange(N).reshape((-1, 1))
