#!/usr/bin/env python
import numpy as np

from utils.utils import normalize_to_unit


class RandomClusterRepresentation:
    def __init__(self, num_features, seed, unit_norm=True):

        self.num_features = num_features

        rand_generator = np.random.RandomState(seed)

        if num_features == 5:
            group_sizes = [1, 1, 1, 1, 1]
        elif num_features == 4:
            group_sizes = [2, 1, 1, 1]
        elif num_features == 3:
            group_sizes = [2, 2, 1]
        elif num_features == 2:
            group_sizes = [3, 2]
        elif num_features == 1:
            group_sizes = [5]
        else:
            raise ValueError("Wrong number of groups. Valid are 1, 2, 3, 4 and 5.")

        self.features = []
        for i_g, gs in enumerate(group_sizes):
            phi = np.zeros((gs, num_features))
            phi[:, i_g] = 1.0
            self.features.append(phi)
        self.features = np.concatenate(self.features, axis=0)

        rand_generator.shuffle(self.features)

        if unit_norm:
            self.features = normalize_to_unit(self.features)

    def __getitem__(self, x):
        return self.features[x].squeeze()
