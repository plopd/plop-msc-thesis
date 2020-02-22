#!/usr/bin/env python
import numpy as np


class TabularRepresentations:
    def __init__(self, num_states, unit_norm=False):

        self.num_features = num_states

        self.features = np.eye(num_states)

        if unit_norm:
            self.features = np.divide(
                self.features, np.linalg.norm(self.features, axis=1).reshape((-1, 1))
            )

    def __getitem__(self, x):
        return self.features[x].squeeze()
