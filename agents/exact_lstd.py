import numpy as np

from utils.objectives import MSVE


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
    if which == "ETD":
        M = calculate_M(P_pi, Gamma, Lmbda, i, d_mu)
        M = M / np.sum(M)
    elif which == "TD":
        M = np.diag(d_mu)
    else:
        raise Exception("Unexpected which given.")

    theta, _, _ = calculate_theta(P_pi, Gamma, Lmbda, Phi, r_pi, M)
    msve = MSVE(true_v, np.dot(Phi, theta), d_mu)
    approx_v = np.dot(Phi, theta)

    return theta, msve, approx_v, M
