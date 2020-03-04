#!/usr/bin/env python
import numpy as np


class InvertedRepresentations:
    def __init__(self, num_states, unit_norm=True):

        self.num_features = num_states
        self.features = np.ones((num_states, num_states)) - np.eye(num_states)

        if unit_norm:
            self.features = np.divide(
                self.features, np.linalg.norm(self.features, axis=1).reshape((-1, 1))
            )

    def __getitem__(self, x):
        return self.features[x].squeeze()
