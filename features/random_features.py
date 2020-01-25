#!/usr/bin/env python
import numpy as np


class RandomFeatures:
    def __init__(self, name, num_states, num_features, num_ones, seed, unit_norm=True):

        if name != "RB" and name != "RNB":
            raise Exception("Unknown name given.")

        np.random.seed(seed)

        if name == "RNB":
            self.features = np.random.randn(num_states, num_features)

        elif name == "RB":
            num_zeros = num_features - num_ones
            self.features = np.zeros((num_states, num_features))

            for j in range(num_states):
                random_array = np.array([0] * num_ones + [1] * num_zeros)
                np.random.shuffle(random_array)
                self.features[j, :] = random_array

        if unit_norm:
            self.features = np.divide(
                self.features, np.linalg.norm(self.features, axis=1).reshape((-1, 1))
            )

    def __getitem__(self, x):
        return self.features[x].squeeze()
