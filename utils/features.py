import itertools

import numpy as np


def get_inverted_feature(x, in_features, unit_norm=True):
    representations = np.ones((in_features, in_features))
    representations[np.arange(in_features), np.arange(in_features)] = 0

    if unit_norm:
        representations = np.divide(
            representations, np.linalg.norm(representations, axis=1).reshape((-1, 1))
        )

    features = representations[x].squeeze()
    return features


def get_dependent_feature(x, in_features, unit_norm=True):

    upper = np.tril(np.ones((in_features, in_features)), k=0)
    lower = np.triu(np.ones((in_features - 1, in_features)), k=1)
    representations = np.vstack((upper, lower))

    if unit_norm:
        representations = np.divide(
            representations, np.linalg.norm(representations, axis=1).reshape((-1, 1))
        )

    features = representations[x].squeeze()
    return features


def get_random_features(x, name, in_features, num_ones, seed, unit_norm=True):

    if name != "random-binary" and name != "random-nonbinary":
        raise Exception("Unknown name given.")

    np.random.seed(seed + x)

    num_zeros = in_features - num_ones
    zeros = np.zeros(num_zeros)
    ones = np.ones(num_ones)
    features = np.hstack((zeros, ones))

    if name == "random-binary":
        np.random.shuffle(features)
    else:
        features = np.random.randn(in_features)

    if unit_norm:
        features = features / np.linalg.norm(features)

    return features


def get_tabular_feature(x, in_features):
    features = np.zeros(in_features)
    features[x] = 1

    return features


def get_bases_feature(x, name, order, in_features, v_min, v_max, unit_norm=True):
    if name != "poly" and name != "fourier":
        raise Exception("Unknown name given.")

    if v_min is not None and v_max is not None:
        x = (x - v_min) / (v_max - v_min)

    k = len(x)
    num_features = (order + 1) ** k

    assert in_features == num_features

    c = [i for i in range(order + 1)]
    C = np.array(list(itertools.product(*[c for _ in range(k)]))).reshape((-1, k))

    features = np.zeros(num_features)

    if name == "poly":
        features = np.prod(np.power(x, C), axis=1)
    elif name == "fourier":
        features = np.cos(np.pi * np.dot(C, x))

    if unit_norm:
        features = features / np.linalg.norm(features)

    return features


def get_feature_state_aggregation(x, in_features, seed, unit_norm=True):

    np.random.seed(seed)

    if in_features == 5:
        group_sizes = [1, 1, 1, 1, 1]
    elif in_features == 4:
        group_sizes = [2, 1, 1, 1]
    elif in_features == 3:
        group_sizes = [2, 2, 1]
    elif in_features == 2:
        group_sizes = [3, 2]
    elif in_features == 1:
        group_sizes = [5]
    else:
        raise ValueError("Wrong number of groups. Valid are 1, 2, 3, 4 and 5")

    representations = []
    for i_g, gs in enumerate(group_sizes):
        phi = np.zeros((gs, in_features))
        phi[:, i_g] = 1.0
        representations.append(phi)
    representations = np.concatenate(representations, axis=0)

    if unit_norm:
        representations = np.divide(
            representations, np.linalg.norm(representations, axis=1).reshape((-1, 1))
        )

    np.random.shuffle(representations)

    features = representations[x].squeeze()

    return features
