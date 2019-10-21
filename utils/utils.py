import os

import numpy as np


def path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def calculate_auc(ys):
    auc = np.mean(ys)
    return auc


def get_interest(N, name):
    if name == "uniform":
        return np.ones(N)
    elif name == "random binary":
        raise NotImplementedError

    raise Exception("Unexpected interest given")


def calc_rmsve(true_state_val, learned_state_val, state_distribution, interest):
    true_state_val = np.squeeze(true_state_val)
    learned_state_val = np.squeeze(learned_state_val)
    assert len(true_state_val) == len(learned_state_val)
    dmu_i = np.multiply(state_distribution, interest)
    deltas = np.square(true_state_val - learned_state_val)
    deltas = np.multiply(dmu_i, deltas)
    msve = np.sum(deltas, axis=0)
    msve = msve / np.sum(dmu_i)

    rmsve = np.sqrt(msve)

    return rmsve


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


def SolveExample(
    P_pi, Gamma, Lmbda, Phi, r_pi, d_mu, i, true_v, which, title, logging=True, log=None
):
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
        title: str,
    Returns:

    """
    if which == "etd":
        M = calculate_M(P_pi, Gamma, Lmbda, i, d_mu)
    elif which == "td":
        M = np.diag(d_mu)
    else:
        raise ValueError("only td or etd acceptable.")

    theta, _, _ = calculate_theta(P_pi, Gamma, Lmbda, Phi, r_pi, M)
    msve = calc_rmsve(true_v, np.dot(Phi, theta), d_mu, i)
    approx_v = np.dot(Phi, theta)

    if logging:
        log_output(
            log,
            theta=theta,
            msve=msve,
            approx_v=approx_v,
            M=M,
            which=which,
            title=title,
        )

    return theta, msve, approx_v, M


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
    SolveExample(P_pi, Gamma, Lmbda, Phi, r_pi, d_mu, i, true_v, which)


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
    SolveExample(P_pi, Gamma, Lmbda, Phi, r_pi, d_mu, i, true_v, which)


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
    SolveExample(P_pi, Gamma, Lmbda, Phi, r_pi, d_mu, i, true_v, which)


def log_output(log, theta, msve, approx_v, M, which, title):
    log.info(f"{title} -- {which.upper()}\n=================================")
    # log.info(f"Theta:\n{theta}")
    # log.info(f"Emphasis:\n{M}")
    # log.info(f"Approx_v:\n{approx_v}")
    log.info(f"MSVE:\t{msve}")
    log.info("-----------------------------------\n")


def get_features(N, name=None):
    if name == "tabular":
        return get_tabular_features(N)
    elif name == "inverted":
        return get_inverted_features(N)
    elif name == "dependent":
        return get_dependent_features(N)
    raise Exception("Unexpected features given")


def get_inverted_features(N):
    features = np.array(
        [0 if i == j else 1 / np.sqrt(N - 1) for i in range(N) for j in range(N)]
    ).reshape((N, N))

    return features


def get_dependent_features(N):
    N = int(np.ceil(N / 2))
    upper = np.tril(np.ones((N, N)), k=0)
    lower = np.triu(np.ones((N - 1, N)), k=1)
    features = np.vstack((upper, lower))
    features = np.divide(features, np.linalg.norm(features, axis=1).reshape((-1, 1)))

    return features


def get_random_binary_features(N, n, num_ones, seed):

    np.random.seed(seed)
    num_zeros = n - num_ones
    representations = np.zeros((N, n))

    for i_s in range(N):
        random_array = np.array([0] * num_zeros + [1] * num_ones)
        np.random.shuffle(random_array)
        representations[i_s, :] = random_array

    return representations


def calculate_phi_for_five_states_with_state_aggregation(n):

    if n == 5:
        group_sizes = [1, 1, 1, 1, 1]
    elif n == 4:
        group_sizes = [2, 1, 1, 1]
    elif n == 3:
        group_sizes = [2, 2, 1]
    elif n == 2:
        group_sizes = [3, 2]
    elif n == 1:
        group_sizes = [5]
    else:
        raise ValueError("Wrong number of groups. Valid are 1, 2, 3, 4 and 5")

    Phi = []
    for i_g, gs in enumerate(group_sizes):
        phi = np.zeros((gs, n))
        phi[:, i_g] = 1.0
        Phi.append(phi)
    Phi = np.concatenate(Phi, axis=0)

    return Phi


def get_tabular_features(N):
    features = np.eye(N)
    return features


if __name__ == "__main__":
    pass
    # calculate_v_chain(19)
    # calculate_state_distribution(19)
    # TwoStateKeynoteConstant(which="td")
    # TwoStateKeynoteConstant(which="etd")

    # TwoStateConstant(which="td")
    # TwoStateConstant(which="etd")
    #
    # TwoStateDependent(which="td")
    # TwoStateDependent(which="etd")

    # SolveExample(
    #     P_pi=np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]),
    #     Gamma=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    #     Lmbda=np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    #     Phi=np.array([[1, 0], [1, 0], [0, 1], [0, 1]]),
    #     r_pi=np.array([1, 1, 1, 1]).reshape((-1, 1)),
    #     d_mu=np.array([0.25, 0.25, 0.25, 0.25]),
    #     i=np.array([1, 0, 0, 0]),
    #     true_v=np.array([4, 3, 2, 1]).reshape((-1, 1)),
    #     which="etd",
    #     title="###### Example 9.5: Interest and Emphasis",
    # )

    # SolveExample(
    #     P_pi=np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]),
    #     Gamma=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    #     Lmbda=np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    #     Phi=np.array([[1, 0], [1, 0], [0, 1], [0, 1]]),
    #     r_pi=np.array([1, 1, 1, 1]).reshape((-1, 1)),
    #     d_mu=np.array([0.25, 0.25, 0.25, 0.25]),
    #     i=np.array([0, 0, 1, 0]),  # or [0, 0, 0, 1] results in Singular matrix!
    #     true_v=np.array([4, 3, 2, 1]).reshape((-1, 1)),
    #     which="etd",
    #     title="Example 9.5: Interest and Emphasis"
    # )

    # SolveExample(
    #     P_pi=np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]),
    #     Gamma=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    #     Lmbda=np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    #     Phi=np.array([[1, 0], [1, 0], [0, 1], [0, 1]]),
    #     r_pi=np.array([1, 1, 1, 1]).reshape((-1, 1)),
    #     d_mu=np.array([0.25, 0.25, 0.25, 0.25]),
    #     i=np.array([1, 0, 0, 1]),
    #     true_v=np.array([4, 3, 2, 1]).reshape((-1, 1)),
    #     which="etd",
    #     title="###### Example 9.5: Interest and Emphasis",
    # )
