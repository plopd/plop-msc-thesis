import itertools
import os

import numpy as np


def path_exists(path):
    if not path.exists():
        os.makedirs(path, exist_ok=True)


def get_interest(name, **kwargs):

    N, seed = kwargs.get("N"), kwargs.get("seed")

    if name == "uniform":
        return np.ones(N)
    elif name == "random-binary":
        np.random.seed(seed)
        random_array = np.random.choice([0, 1], N)
        return random_array

    raise Exception("Unexpected interest given.")


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


def calculate_M(P_pi, Gamma, Lmbda, i, d_mu):
    N = P_pi.shape[0]

    x1 = np.matmul(P_pi, Gamma)
    x1 = np.matmul(x1, Lmbda)
    x1 = np.eye(N) - x1
    x1 = np.linalg.inv(x1)

    x2 = np.eye(N) - np.matmul(P_pi, Gamma)

    P_pi_lmbda = np.eye(N) - np.matmul(x1, x2)

    i = np.multiply(d_mu, i)

    m = np.transpose(P_pi_lmbda)
    m = np.eye(N) - m
    m = np.linalg.inv(m)
    m = np.dot(m, i)

    M = np.diag(m)

    return M


def calculate_theta(P_pi, Gamma, Lmbda, Phi, r_pi, M):
    N = P_pi.shape[0]

    x1 = np.matmul(P_pi, Gamma)
    x1 = np.matmul(x1, Lmbda)
    x1 = np.eye(N) - x1
    x1 = np.linalg.inv(x1)

    x2 = np.matmul(P_pi, Gamma)
    x2 = np.eye(N) - x2

    x3 = np.matmul(x1, x2)

    A = np.matmul(M, x3)
    A = np.matmul(Phi.T, A)
    A = np.matmul(A, Phi)

    b = np.matmul(M, x1)
    b = np.matmul(Phi.T, b)
    b = np.dot(b, r_pi)

    theta = np.dot(np.linalg.inv(A), b)

    return theta, A, b


def calculate_v_pi(P_pi, Gamma, Lmbda, r_pi):
    N = P_pi.shape[0]

    x1 = np.matmul(P_pi, Gamma)
    x1 = np.matmul(x1, Lmbda)
    x1 = np.eye(N) - x1
    x1 = np.linalg.inv(x1)

    x2 = np.eye(N) - np.matmul(P_pi, Gamma)

    P_pi_lmbda = np.eye(N) - np.matmul(x1, x2)

    r_pi_lmbda = np.dot(x1, r_pi)

    v_pi = np.dot(np.linalg.inv(np.eye(N) - P_pi_lmbda), r_pi_lmbda)

    return v_pi


def run_exact_lstd(P_pi, Gamma, Lmbda, Phi, r_pi, d_mu, i, true_v, which):
    """

    Args:
        P_pi: (N,N)
        Gamma: (N,N)
        Lmbda: (N,N)
        Phi: (N, n)
        r_pi: (N, 1)
        d_mu: (N,)
        i: (N,)
        true_v: (N,)
        which: str, either "td" or "etd"
    Returns:

    """
    if which == "etd":
        M = calculate_M(P_pi, Gamma, Lmbda, i, d_mu)
        M = M / np.sum(M)
    elif which == "td":
        M = np.diag(d_mu)
    else:
        raise Exception("Unexpected which given.")

    theta, _, _ = calculate_theta(P_pi, Gamma, Lmbda, Phi, r_pi, M)
    msve = MSVE(true_v, np.dot(Phi, theta), d_mu)
    approx_v = np.dot(Phi, theta)

    return theta, msve, approx_v, M


def get_feature(x, unit_norm=True, **kwargs):
    """ Construct various features from states.

    Args:
        x: ndarray, shape (k,)
        name: str,

    Returns:

    """

    name = kwargs.get("features")
    order = kwargs.get("order")
    num_states = kwargs.get("N")
    in_features = kwargs.get("in_features")
    num_ones = kwargs.get("num_ones", 0)
    seed = kwargs.get("seed")

    if name == "tabular":
        return get_tabular_feature(x, in_features)
    elif name == "inverted":
        return get_inverted_feature(x, in_features, unit_norm)
    elif name == "dependent":
        return get_dependent_feature(x, in_features, unit_norm)
    elif name == "poly" or name == "fourier":
        return get_bases_feature(x, name, order, in_features, unit_norm)
    elif name == "random-binary":
        return get_random_feature(
            x, num_states, name, in_features, num_ones, seed, unit_norm=False
        )
    elif name == "random-nonbinary":
        return get_random_feature(
            x, num_states, name, in_features, num_ones, seed, unit_norm
        )
    raise Exception("Unexpected name given.")


def get_inverted_feature(x, in_features, unit_norm=True):
    representations = np.ones((in_features, in_features))
    representations[np.arange(in_features), np.arange(in_features)] = 0

    if unit_norm:
        representations = np.divide(
            representations, np.linalg.norm(representations, axis=1).reshape((-1, 1))
        )

    features = representations[x].squeeze()
    return features


def get_dependent_feature(x, in_features, unit_norm=True):

    upper = np.tril(np.ones((in_features, in_features)), k=0)
    lower = np.triu(np.ones((in_features - 1, in_features)), k=1)
    representations = np.vstack((upper, lower))

    if unit_norm:
        representations = np.divide(
            representations, np.linalg.norm(representations, axis=1).reshape((-1, 1))
        )

    features = representations[x].squeeze()
    return features


def get_random_feature(
    x, num_states, name, in_features, num_ones, seed=None, unit_norm=True
):

    if name != "random-binary" and name != "random-nonbinary":
        raise Exception("Unknown feature_type given.")

    np.random.seed(seed)
    num_zeros = in_features - num_ones
    representations = np.zeros((num_states, in_features))

    for i_s in range(num_states):
        if name == "random-binary":
            random_array = np.array([0] * num_zeros + [1] * num_ones)
            np.random.shuffle(random_array)
        else:
            random_array = np.random.randn(in_features)
        representations[i_s, :] = random_array

    if unit_norm:
        representations = np.divide(
            representations, np.linalg.norm(representations, axis=1).reshape((-1, 1))
        )

    features = representations[x].squeeze()
    return features


def get_tabular_feature(x, in_features):
    features = np.zeros(in_features)
    features[x] = 1

    return features


def get_bases_feature(x, name, order, in_features, unit_norm=True):
    """
    Construct order-n polynomial- or Fourier-basis features from states.

    Args:
        x: ndarray, shape (k,)
        name: str, 'poly' or 'fourier'
        order: int

    Returns: ndarray of size (num_features,), with num_features = (order+1)**k

    """

    if name != "poly" and name != "fourier":
        raise Exception("Unknown name given.")

    k = len(x)
    num_features = (order + 1) ** k

    assert in_features == num_features

    c = [i for i in range(order + 1)]
    C = np.array(list(itertools.product(*[c for _ in range(k)])))

    features = np.zeros(num_features)

    if name == "poly":
        features = np.prod(np.power(x, C), axis=1)
    elif name == "fourier":
        features = np.cos(np.pi * np.dot(C, x))

    if unit_norm:
        features = features / np.linalg.norm(features)

    return features


def get_chain_states(N):
    return np.arange(N).reshape((-1, 1))
