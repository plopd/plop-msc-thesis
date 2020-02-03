#!/usr/bin/env python
import itertools

import numpy as np


class PolynomialRepresentation:
    def __init__(self, name, num_dims, order, v_min, v_max, unit_norm=True):
        self.name = name
        self.unit_norm = unit_norm
        self.v_min = v_min
        self.v_max = v_max

        if name != "P" and name != "F":
            raise Exception("Unknown name given.")

        self.num_features = (order + 1) ** num_dims

        c = [i for i in range(order + 1)]
        self.C = np.array(
            list(itertools.product(*[c for _ in range(num_dims)]))
        ).reshape((-1, num_dims))

    def __getitem__(self, x):

        if self.v_min is not None and self.v_max is not None:
            x = (x - self.v_min) / (self.v_max - self.v_min)

        features = np.zeros(self.num_features)

        if self.name == "P":
            features = np.prod(np.power(x, self.C), axis=1)
        elif self.name == "F":
            features = np.cos(np.pi * np.dot(self.C, x))

        if self.unit_norm:
            features = features / np.linalg.norm(features)

        return features
