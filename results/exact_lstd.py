import matplotlib.pyplot as plt
import numpy as np


def main(n_states):

    if n_states == 5:
        M_td = np.array([0.11143, 0.222273, 0.333124, 0.221995, 0.111178])
        M_etd = np.array([0.13049204, 0.23191021, 0.27533378, 0.23183994, 0.13042403])
    elif n_states == 19:
        M_td = np.array(
            [
                0.009906,
                0.019775,
                0.029772,
                0.039789,
                0.049229,
                0.058754,
                0.06882,
                0.079348,
                0.089988,
                0.10019,
                0.090655,
                0.081195,
                0.071024,
                0.0606,
                0.050468,
                0.040169,
                0.030098,
                0.02018,
                0.01004,
            ]
        )
        M_etd = np.array(
            [
                0.01193134,
                0.02362549,
                0.03484617,
                0.04535403,
                0.05490921,
                0.0632857,
                0.07025544,
                0.07557743,
                0.07899958,
                0.08026715,
                0.07913586,
                0.07583402,
                0.07058813,
                0.06364171,
                0.05524434,
                0.04563861,
                0.03507112,
                0.02378299,
                0.01201169,
            ]
        )
    else:
        raise Exception("Unexpected number of states given.")

    xs = np.arange(0, n_states)
    plt.plot(xs, M_td, label="d_mu")
    plt.plot(xs, M_etd, label="M")
    plt.xticks(xs, xs)
    plt.xlabel("States")
    plt.ylabel("State update probability")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(5)
    main(19)
