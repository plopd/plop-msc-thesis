#!/usr/bin/env python
import numpy as np


class TabularFeatures:
    def __init__(self, num_states):

        self.features = np.eye(num_states)

    def __getitem__(self, x):
        return self.features[x].squeeze()
