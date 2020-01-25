#!/usr/bin/env python
import numpy as np


class DependentFeatures:
    def __init__(self, num_states, num_features, unit_norm=True):

        upper = np.tril(np.ones((num_states // 2, num_features)), k=0)
        middle = np.ones(num_features)
        lower = np.triu(np.ones((num_states // 2, num_features)), k=1)
        self.features = np.vstack((upper, middle, lower))

        if unit_norm:
            self.features = np.divide(
                self.features, np.linalg.norm(self.features, axis=1).reshape((-1, 1))
            )

    def __getitem__(self, x):
        return self.features[x].squeeze()