import numpy as np


def weight_norm(X, W):
    return np.sqrt(X.T.dot(W).dot(X))


def MSVE(true_v, est_v, mu):
    """
    Compute the Root Mean Square Value Error
    Args:
        true_v: ndarray (N,)
        est_v: ndarray (N,)
        mu: ndarray (N,)

    Returns:

    """
    D = np.diag(mu)
    res = weight_norm(true_v - est_v, D)

    return res
