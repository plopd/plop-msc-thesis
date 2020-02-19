#!/usr/bin/env python
import numpy as np

from representations.polynomial import PolynomialRepresentation
from utils.utils import minmax_normalization_ab


class FourierRepresentation(PolynomialRepresentation):
    def __init__(self, name, num_dims, order, min_x, max_x, a, b, unit_norm=True):
        super().__init__(name, num_dims, order, min_x, max_x, a, b, unit_norm)

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


# if __name__ == '__main__':
#     import representations.representations as repr
#     repr_dct = {
#             "min_x": 0,
#             "max_x": 1,
#             "a": 0,
#             "b": 1,
#             "num_dims": 2,
#             "order": 2,
#         }
#     F = repr.get_representation(
#         "F",
#         **repr_dct
#     )
#     num_states = 5
#     np.random.seed(0)
#     states = np.random.uniform(0, 1, 5*repr_dct.get("num_dims")).reshape((num_states, repr_dct.get("num_dims")))
#     print(F[states[0]])
