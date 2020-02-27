#!/usr/bin/env python
import numpy as np

from representations.polynomial import PolynomialRepresentation
from utils.utils import minmax_normalization_ab


class FourierRepresentation(PolynomialRepresentation):
    def __init__(self, num_dims, order, min_x, max_x, a, b, unit_norm=True):
        super().__init__(num_dims, order, min_x, max_x, a, b, unit_norm)

    def __getitem__(self, x):

        if (
            self.min_x is not None
            and self.max_x is not None
            and self.a is not None
            and self.b is not None
        ):
            x = minmax_normalization_ab(x, self.min_x, self.max_x, self.a, self.b)

        features = np.cos(np.pi * np.dot(self.C, x))

        if self.unit_norm:
            features = features / np.linalg.norm(features)

        return features
