#!/usr/bin/env python
import itertools

import numpy as np


class BasisRepresentation:
    def __init__(self, name, num_dims, order, min_x, max_x, a, b, unit_norm=True):
        self.name = name
        self.unit_norm = unit_norm
        self.min_x = min_x
        self.max_x = max_x
        self.a = a
        self.b = b

        if name != "P" and name != "F":
            raise Exception("Unknown name given.")

        self.num_features = (order + 1) ** num_dims

        c = [i for i in range(order + 1)]
        self.C = np.array(
            list(itertools.product(*[c for _ in range(num_dims)]))
        ).reshape((-1, num_dims))

    def __getitem__(self, x):

        if (
            self.min_x is not None
            and self.max_x is not None
            and self.a is not None
            and self.b is not None
        ):
            # https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
            x = (self.b - self.a) * (x - self.min_x) / (
                self.max_x - self.min_x
            ) + self.a

        features = np.zeros(self.num_features)

        if self.name == "P":
            features = np.prod(np.power(x, self.C), axis=1)
        elif self.name == "F":
            features = np.cos(np.pi * np.dot(self.C, x))

        if self.unit_norm:
            features = features / np.linalg.norm(features)

        return features
