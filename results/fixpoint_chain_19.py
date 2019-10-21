import logging
import os

import numpy as np

from utils.utils import SolveExample

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format="%(message)s")


def main():
    N = 19
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
    true_v = np.array(
        [
            -0.906_711_97,
            -0.804_280_19,
            -0.708_252_43,
            -0.574_593_76,
            -0.451_977_53,
            -0.345_806_44,
            -0.224_006_71,
            -0.150_470_34,
            -0.097_719_99,
            0.005_349_21,
            0.120_276_07,
            0.218_865_73,
            0.314_948_7,
            0.444_041_99,
            0.575_407_42,
            0.676_515_53,
            0.758_742_48,
            0.843_161_59,
            0.911_814_87,
        ]
    )
    r_pi = np.zeros(N)
    r_pi[0] = -0.5
    r_pi[-1] = 0.5
    Phi_tabular = np.eye(N)
    Phi_inverted = np.array(
        [0 if i == j else 0.5 for i in range(N) for j in range(N)]
    ).reshape((N, N))
    Phi_dependent = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0],
            [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3), 0, 0, 0, 0, 0, 0, 0],
            [
                1 / np.sqrt(4),
                1 / np.sqrt(4),
                1 / np.sqrt(4),
                1 / np.sqrt(4),
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1 / np.sqrt(5),
                1 / np.sqrt(5),
                1 / np.sqrt(5),
                1 / np.sqrt(5),
                1 / np.sqrt(5),
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1 / np.sqrt(6),
                1 / np.sqrt(6),
                1 / np.sqrt(6),
                1 / np.sqrt(6),
                1 / np.sqrt(6),
                1 / np.sqrt(6),
                0,
                0,
                0,
                0,
            ],
            [
                1 / np.sqrt(7),
                1 / np.sqrt(7),
                1 / np.sqrt(7),
                1 / np.sqrt(7),
                1 / np.sqrt(7),
                1 / np.sqrt(7),
                1 / np.sqrt(7),
                0,
                0,
                0,
            ],
            [
                1 / np.sqrt(8),
                1 / np.sqrt(8),
                1 / np.sqrt(8),
                1 / np.sqrt(8),
                1 / np.sqrt(8),
                1 / np.sqrt(8),
                1 / np.sqrt(8),
                1 / np.sqrt(8),
                0,
                0,
            ],
            [
                1 / np.sqrt(9),
                1 / np.sqrt(9),
                1 / np.sqrt(9),
                1 / np.sqrt(9),
                1 / np.sqrt(9),
                1 / np.sqrt(9),
                1 / np.sqrt(9),
                1 / np.sqrt(9),
                1 / np.sqrt(9),
                0,
            ],
            [
                1 / np.sqrt(10),
                1 / np.sqrt(10),
                1 / np.sqrt(10),
                1 / np.sqrt(10),
                1 / np.sqrt(10),
                1 / np.sqrt(10),
                1 / np.sqrt(10),
                1 / np.sqrt(10),
                1 / np.sqrt(10),
                1 / np.sqrt(10),
            ],
            [
                0,
                1 / np.sqrt(9),
                1 / np.sqrt(9),
                1 / np.sqrt(9),
                1 / np.sqrt(9),
                1 / np.sqrt(9),
                1 / np.sqrt(9),
                1 / np.sqrt(9),
                1 / np.sqrt(9),
                1 / np.sqrt(9),
            ],
            [
                0,
                0,
                1 / np.sqrt(8),
                1 / np.sqrt(8),
                1 / np.sqrt(8),
                1 / np.sqrt(8),
                1 / np.sqrt(8),
                1 / np.sqrt(8),
                1 / np.sqrt(8),
                1 / np.sqrt(8),
            ],
            [
                0,
                0,
                0,
                1 / np.sqrt(7),
                1 / np.sqrt(7),
                1 / np.sqrt(7),
                1 / np.sqrt(7),
                1 / np.sqrt(7),
                1 / np.sqrt(7),
                1 / np.sqrt(7),
            ],
            [
                0,
                0,
                0,
                0,
                1 / np.sqrt(6),
                1 / np.sqrt(6),
                1 / np.sqrt(6),
                1 / np.sqrt(6),
                1 / np.sqrt(6),
                1 / np.sqrt(6),
            ],
            [
                0,
                0,
                0,
                0,
                0,
                1 / np.sqrt(5),
                1 / np.sqrt(5),
                1 / np.sqrt(5),
                1 / np.sqrt(5),
                1 / np.sqrt(5),
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1 / np.sqrt(4),
                1 / np.sqrt(4),
                1 / np.sqrt(4),
                1 / np.sqrt(4),
            ],
            [0, 0, 0, 0, 0, 0, 0, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            [0, 0, 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
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
