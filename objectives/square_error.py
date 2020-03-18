import numpy as np


class RootSquareValueErrorObjective:
    def __init__(self, true_values, mu, i):
        """
        Compute the Value Error
        Args:
            true_values: ndarray (N,)
            mu: ndarray (N,)
            i: ndarray (N,)
        """

        self.true_values = true_values
        self.mu = mu
        self.i = i

    def weight_norm(self, X, W):
        return np.sqrt(X.T.dot(W).dot(X))

    def value(self, estimated_values):
        weighting = self.mu * self.i
        D = np.diag(weighting)
        val = self.weight_norm(self.true_values - estimated_values, D)

        return val
