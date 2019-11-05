import numpy as np

from utils.utils import run_exact_lstd


def TwoStateKeynoteConstant(which):
    P_pi = np.array([[0, 1], [0, 0]])
    Gamma = np.array([[1, 0], [0, 1]])
    Lmbda = np.array([[0, 0], [0, 0]])
    Phi = np.array([1, 1]).reshape((-1, 1))
    r_pi = np.array([1, 1]).reshape((-1, 1))
    d_mu = np.array([0.5, 0.5])
    i = np.ones_like(d_mu)
    true_v = np.array((2, 1)).reshape((-1, 1))
    print(f"{TwoStateKeynoteConstant.__name__}", "\n----------------")
    run_exact_lstd(P_pi, Gamma, Lmbda, Phi, r_pi, d_mu, i, true_v, which)


def TwoStateConstant(which):
    P_pi = np.array([[0, 1], [0, 0]])
    Gamma = np.array([[1, 0], [0, 1]])
    Lmbda = np.array([[0, 0], [0, 0]])
    Phi = np.array([1, 1]).reshape((-1, 1))
    r_pi = np.array([2, 0]).reshape((-1, 1))
    d_mu = np.array([0.5, 0.5])
    i = np.ones_like(d_mu)
    true_v = np.array((2, 0)).reshape((-1, 1))
    print(f"{TwoStateConstant.__name__}", "\n----------------")
    run_exact_lstd(P_pi, Gamma, Lmbda, Phi, r_pi, d_mu, i, true_v, which)


def TwoStateDependent(which):
    P_pi = np.array([[0, 1], [0, 0]])
    Gamma = np.array([[1, 0], [0, 0]])
    Lmbda = np.array([[0, 0], [0, 1]])
    Phi = np.array([1, 1]).reshape((-1, 1))
    r_pi = np.array([2, 0]).reshape((-1, 1))
    d_mu = np.array([0.5, 0.5])
    i = np.array([1, 0])
    true_v = np.array((2, 0)).reshape((-1, 1))
    print(f"{TwoStateDependent.__name__}", "\n----------------")
    run_exact_lstd(P_pi, Gamma, Lmbda, Phi, r_pi, d_mu, i, true_v, which)
