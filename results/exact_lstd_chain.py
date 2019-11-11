import sys

import numpy as np

from utils.calculate_state_distribution_chain import calculate_state_distribution
from utils.utils import calculate_v_pi
from utils.utils import get_feature
from utils.utils import get_interest
from utils.utils import run_exact_lstd


def compute_solution(N, method, features, interest, n_runs=1, n=None, num_ones=None):
    if N == 5:
        d_mu = np.array([0.11143, 0.222_273, 0.333_124, 0.221_995, 0.111_178])
    elif N == 19:
        d_mu = np.array(
            [
                0.009_906,
                0.019_775,
                0.029_772,
                0.039_789,
                0.049_229,
                0.058_754,
                0.06882,
                0.079_348,
                0.089_988,
                0.10019,
                0.090_655,
                0.081_195,
                0.071_024,
                0.0606,
                0.050_468,
                0.040_169,
                0.030_098,
                0.02018,
                0.01004,
            ]
        )
    else:
        d_mu = calculate_state_distribution(N)
    r_pi = np.zeros(N)
    r_pi[0] = -0.5
    r_pi[-1] = 0.5

    states = np.arange(N).reshape(-1, 1)
    Phi = get_feature(states, name=features, n=n, num_ones=num_ones, seed=None)
    P_pi = np.array(
        [0.5 if i + 1 == j or i - 1 == j else 0 for i in range(N) for j in range(N)]
    ).reshape((N, N))
    true_v = calculate_v_pi(P_pi, np.eye(N), np.zeros(N), r_pi)

    theta_acc = np.zeros((Phi.shape[1], 1))
    msve_acc = 0.0
    M_acc = np.zeros((N, N))
    v_theta_acc = np.zeros((N, 1))
    for i in range(n_runs):
        theta, msve, v_theta, M = run_exact_lstd(
            P_pi=P_pi,
            Gamma=np.eye(N),
            Lmbda=np.zeros(N),
            Phi=get_feature(states, name=features, n=n, num_ones=num_ones, seed=i),
            r_pi=r_pi.reshape((-1, 1)),
            d_mu=d_mu,
            i=get_interest(N, name=interest, seed=i),
            true_v=true_v.reshape((-1, 1)),
            which=method,
        )
        theta_acc += theta
        msve_acc += msve
        M_acc += M
        v_theta_acc += v_theta

    theta_acc /= n_runs
    msve_acc /= n_runs
    M_acc /= n_runs
    v_theta_acc /= n_runs

    return theta_acc, msve_acc, M_acc, v_theta_acc


if __name__ == "__main__":
    N = int(sys.argv[1])
    method = sys.argv[2]
    features = sys.argv[3]
    interest = sys.argv[4]
    n_runs = int(sys.argv[5])
    n = int(sys.argv[6])
    num_ones = int(sys.argv[7])
    theta_acc, msve_acc, M_acc, v_theta_acc = compute_solution(
        N, method, features, interest, n_runs, n, num_ones
    )
    print(msve_acc)
