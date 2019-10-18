import logging
import os

import numpy as np
from utils.utils import SolveExample

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format="%(message)s")


def main():
    N = 5
    d_mu = np.array([0.11143, 0.222_273, 0.333_124, 0.221_995, 0.111_178])
    true_v = np.array(
        [0.170_387_83, 0.330_646_47, 0.502_222_93, 0.666_294_01, 0.836_713_85]
    )
    r_pi = np.zeros(N)
    r_pi[-1] = 0.5
    Phi_tabular = np.eye(N)
    Phi_inverted = np.array(
        [0 if i == j else 0.5 for i in range(N) for j in range(N)]
    ).reshape((N, N))
    Phi_dependent = np.array(
        [
            [1, 0, 0],
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
            [0, 0, 1],
        ]
    )
    P_pi = np.array(
        [0.5 if i + 1 == j or i - 1 == j else 0 for i in range(N) for j in range(N)]
    ).reshape((N, N))
    SolveExample(
        P_pi=P_pi,
        Gamma=np.eye(N),
        Lmbda=np.zeros(N),
        Phi=Phi_tabular,
        r_pi=r_pi.reshape((-1, 1)),
        d_mu=d_mu,
        i=np.ones(N),
        true_v=true_v.reshape((-1, 1)),
        which="td",
        title="Five-states Random Walk -- Tabular -- Uniform",
        logging=True,
        log=log,
    )

    SolveExample(
        P_pi=P_pi,
        Gamma=np.eye(N),
        Lmbda=np.zeros(N),
        Phi=Phi_tabular,
        r_pi=r_pi.reshape((-1, 1)),
        d_mu=d_mu,
        i=np.ones(N),
        true_v=true_v.reshape((-1, 1)),
        which="etd",
        title="Five-states Random Walk -- Tabular -- Uniform",
        logging=True,
        log=log,
    )

    msve_td = 0.0
    msve_etd = 0.0

    n_runs = 300
    for n_r in range(n_runs):
        np.random.seed(n_r)
        i = np.random.randint(0, 2, size=N)
        if np.sum(i) == 0:
            i = np.ones(N)
        _, msve, _, _ = SolveExample(
            P_pi=P_pi,
            Gamma=np.eye(N),
            Lmbda=np.zeros(N),
            Phi=Phi_inverted,
            r_pi=r_pi.reshape((-1, 1)),
            d_mu=d_mu,
            i=i,
            true_v=true_v.reshape((-1, 1)),
            which="td",
            title="Five-states Random Walk -- Inverted -- Random Binary",
            logging=False,
        )
        msve_td += msve

        _, msve, _, _ = SolveExample(
            P_pi=P_pi,
            Gamma=np.eye(N),
            Lmbda=np.zeros(N),
            Phi=Phi_inverted,
            r_pi=r_pi.reshape((-1, 1)),
            d_mu=d_mu,
            i=i,
            true_v=true_v.reshape((-1, 1)),
            which="etd",
            title="Five-states Random Walk -- Inverted -- Random Binary",
            logging=False,
        )
        msve_etd += msve
    log.info(
        f"Five-states Random Walk -- Inverted -- Random Binary\nFinal MSVE: TD: {msve_td / n_runs}, ETD: {msve_etd / n_runs}\n"
    )

    SolveExample(
        P_pi=P_pi,
        Gamma=np.eye(N),
        Lmbda=np.zeros(N),
        Phi=Phi_dependent,
        r_pi=r_pi.reshape((-1, 1)),
        d_mu=d_mu,
        i=np.ones(N),
        true_v=true_v.reshape((-1, 1)),
        which="td",
        title="Five-states Random Walk -- Dependent -- Uniform",
        logging=True,
        log=log,
    )

    SolveExample(
        P_pi=P_pi,
        Gamma=np.eye(N),
        Lmbda=np.zeros(N),
        Phi=Phi_dependent,
        r_pi=r_pi.reshape((-1, 1)),
        d_mu=d_mu,
        i=np.ones(N),
        true_v=true_v.reshape((-1, 1)),
        which="etd",
        title="Five-states Random Walk -- Dependent -- Uniform",
        logging=True,
        log=log,
    )

    SolveExample(
        P_pi=P_pi,
        Gamma=np.eye(N),
        Lmbda=np.zeros(N),
        Phi=Phi_dependent,
        r_pi=r_pi.reshape((-1, 1)),
        d_mu=d_mu,
        i=np.arange(N) + 1,
        true_v=true_v.reshape((-1, 1)),
        which="td",
        title="Five-states Random Walk -- Dependent -- Linearly Increasing",
        logging=True,
        log=log,
    )

    SolveExample(
        P_pi=P_pi,
        Gamma=np.eye(N),
        Lmbda=np.zeros(N),
        Phi=Phi_dependent,
        r_pi=r_pi.reshape((-1, 1)),
        d_mu=d_mu,
        i=np.arange(N) + 1,
        true_v=true_v.reshape((-1, 1)),
        which="etd",
        title="Five-states Random Walk -- Dependent -- Linearly Increasing",
        logging=True,
        log=log,
    )

    msve_td = 0.0
    msve_etd = 0.0

    n_runs = 300
    for n_r in range(n_runs):
        np.random.seed(n_r)
        i = np.random.randint(0, 2, size=N)
        if np.sum(i) == 0:
            i = np.ones(N)
        _, msve, _, _ = SolveExample(
            P_pi=P_pi,
            Gamma=np.eye(N),
            Lmbda=np.zeros(N),
            Phi=Phi_dependent,
            r_pi=r_pi.reshape((-1, 1)),
            d_mu=d_mu,
            i=i,
            true_v=true_v.reshape((-1, 1)),
            which="td",
            title="Five-states Random Walk -- Dependent -- Random Binary",
            logging=False,
        )
        msve_td += msve

        _, msve, _, _ = SolveExample(
            P_pi=P_pi,
            Gamma=np.eye(N),
            Lmbda=np.zeros(N),
            Phi=Phi_dependent,
            r_pi=r_pi.reshape((-1, 1)),
            d_mu=d_mu,
            i=i,
            true_v=true_v.reshape((-1, 1)),
            which="etd",
            title="Five-states Random Walk -- Dependent -- Random Binary",
            logging=False,
        )
        msve_etd += msve
    log.info(
        f"Five-states Random Walk -- Dependent -- Random Binary\nFinal MSVE: TD: "
        f"{msve_td / n_runs}, ETD: {msve_etd / n_runs}\n"
    )


if __name__ == "__main__":
    main()
