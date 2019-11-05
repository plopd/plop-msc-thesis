import itertools
import os

import numpy as np


def path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def calculate_auc(ys):
    auc = np.mean(ys)
    return auc


def get_interest(N, name, seed=None):
    if name == "uniform":
        return np.ones(N)
    elif name == "random-binary":
        np.random.seed(seed)
        random_array = np.random.choice([0, 1], N)
        return random_array

    raise Exception("Unexpected interest given.")


def weight_norm(X, W):
    return np.sqrt(X.T.dot(W).dot(X))


def MSVE(true_v, est_v, mu, i):
    """
    Compute the Root Mean Square Value Error
    Args:
        true_v: ndarray (N,)
        est_v: ndarray (N,)
        mu: ndarray (N,)
        i: ndarray (N,)

    Returns:

    """
    w = np.multiply(mu, i)
    w = w / np.sum(w)
    W = np.diag(w)
    res = weight_norm(true_v - est_v, W)

    return res


def MSPBE(v_theta, pi, p, r, gamma, Phi, W):
    raise NotImplementedError


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
        raise Exception("only td or etd acceptable.")

    theta, _, _ = calculate_theta(P_pi, Gamma, Lmbda, Phi, r_pi, M)
    msve = MSVE(true_v, np.dot(Phi, theta), d_mu, i)
    approx_v = np.dot(Phi, theta)

    return theta, msve, approx_v, M


def get_features(states, name=None, n=None, num_ones=None, order=None, seed=None):
    """ Construct various features from states.

    Args:
        order: int
        states: ndarray, shape (N, k)
        name: str,
        n: int,
        num_ones: int,
        seed: int,

    Returns:

    """
    if name == "tabular":
        return get_tabular_features(states)
    elif name == "inverted":
        return get_inverted_features(states)
    elif name == "dependent":
        return get_dependent_features(states)
    elif name == "poly":
        return get_bases_features(states, order=order, kind="poly")
    elif name == "fourier":
        return get_bases_features(states, order=order, kind="fourier")
    elif name == "random-binary":
        return get_random_features(
            states, n, num_ones=num_ones, kind="binary", seed=seed
        )
    elif name == "random-nonbinary":
        return get_random_features(states, n, num_ones=0, kind="non-binary", seed=seed)
    raise Exception("Unexpected features given.")


def get_inverted_features(states):
    N, n = states.shape

    features = np.ones((N, N))
    features[np.arange(N), np.arange(N)] = 0
    features = np.divide(features, np.linalg.norm(features, axis=1).reshape((-1, 1)))

    return features


def get_dependent_features(states):
    N, n = states.shape

    D = N // 2 + 1
    upper = np.tril(np.ones((D, D)), k=0)
    lower = np.triu(np.ones((D - 1, D)), k=1)
    features = np.vstack((upper, lower))
    features = np.divide(features, np.linalg.norm(features, axis=1).reshape((-1, 1)))

    return features


def get_random_features(states, n, num_ones, kind="binary", seed=None):
    N, _ = states.shape
    if kind != "binary" and kind != "non-binary":
        raise Exception("Unknown kind given.")

    np.random.seed(seed)
    num_zeros = n - num_ones
    features = np.zeros((N, n))

    for i_s in range(N):
        if kind == "binary":
            random_array = np.array([0] * num_zeros + [1] * num_ones)
        else:
            random_array = np.random.randn(n)
        np.random.shuffle(random_array)
        features[i_s, :] = random_array

    features = np.divide(features, np.linalg.norm(features, axis=1).reshape((-1, 1)))

    return features


def get_tabular_features(states):
    N, n = states.shape
    return np.eye(N)


def get_bases_features(states, order, kind=None, normalize=True):
    """
    Construct order-n polynomial- or Fourier-basis features from states.

    Args:
        states: ndarray, shape (N, k)
        order: int,
        kind: str, 'poly' or 'fourier'

    Returns: ndarray of size (N, num_features), with num_features = (order+1)**k

    """
    if kind != "poly" and kind != "fourier":
        raise Exception("Unknown kind given.")

    if normalize:
        # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
        states = np.divide(states - states.min(), (states.max() - states.min()))

    N, k = states.shape
    num_features = (order + 1) ** k

    c = [i for i in range(0, order + 1)]
    C = np.array(list(itertools.product(*[c for _ in range(k)])))

    X = np.zeros((N, num_features))

    for n in range(N):
        for i in range(num_features):
            if kind == "poly":
                X[n, i] = np.prod(np.power(states[n], C[i]))
            elif kind == "fourier":
                X[n, i] = np.cos(np.pi * np.dot(states[n], C[i]))

    return X
