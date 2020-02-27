#!/usr/bin/env python
import numpy as np

from utils.utils import normalize_to_unit


class RandomRepresentations:
    def __init__(self, num_states, num_features, seed, unit_norm=True):

        self.num_features = num_features
        rand_generator = np.random.RandomState(seed)
        self.features = rand_generator.randn(num_states, num_features)

        if unit_norm:
            self.features = normalize_to_unit(self.features)

    def __getitem__(self, x):
        return self.features[x].squeeze()
