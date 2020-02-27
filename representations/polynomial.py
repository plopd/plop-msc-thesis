#!/usr/bin/env python
import itertools

import numpy as np

from utils.utils import minmax_normalization_ab


class PolynomialRepresentation:
    def __init__(self, num_dims, order, min_x, max_x, a, b, unit_norm=True):
        self.unit_norm = unit_norm
        self.min_x = min_x
        self.max_x = max_x
        self.a = a
        self.b = b

        self.num_features = (order + 1) ** num_dims

        c = [i for i in range(order + 1)]
        self.C = np.array(list(itertools.product(c, repeat=num_dims))).reshape(
            (self.num_features, num_dims)
        )

    def __getitem__(self, x):

        if (
            self.min_x is not None
            and self.max_x is not None
            and self.a is not None
            and self.b is not None
        ):
            x = minmax_normalization_ab(x, self.min_x, self.max_x, self.a, self.b)

        features = np.prod(np.power(x, self.C), axis=1)

        if self.unit_norm:
            features = features / np.linalg.norm(features)

        return features
