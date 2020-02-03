#!/usr/bin/env python
import numpy as np


class TabularRepresentations:
    def __init__(self, num_states):

        self.num_features = num_states

        self.features = np.eye(num_states)

    def __getitem__(self, x):
        return self.features[x].squeeze()
