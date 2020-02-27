#!/usr/bin/env python
import numpy as np

from utils.utils import normalize_to_unit


class RandomBinaryRepresentations:
    def __init__(self, num_states, num_features, num_ones, seed, unit_norm=False):

        self.num_features = num_features
        self.num_ones = num_ones
        rand_generator = np.random.RandomState(seed)
        num_zeros = num_features - num_ones
        self.features = np.zeros((num_states, num_features))

        for j in range(num_states):
            random_array = np.array([1] * num_ones + [0] * num_zeros)
            rand_generator.shuffle(random_array)
            self.features[j, :] = random_array

        if unit_norm:
            self.features = normalize_to_unit(self.features)

    def __getitem__(self, x):
        return self.features[x].squeeze()
