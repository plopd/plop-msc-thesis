from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_fig(N):
    path = Path("~/scratch/Chain").expanduser() / f"true_v_{N}.npy"
    true_v = np.load(path)

    path = Path("~/scratch/Chain").expanduser() / f"state_distribution_{N}.npy"
    state_distribution = np.load(path)

    fig, ax = plt.subplots(figsize=(12, 9))
    true_v = [f"{true_v[i]:.2f}" for i in range(len(true_v))]
    ax.plot(np.arange(len(true_v)), true_v, color="red", marker="o")
    ax.set_title("True state value function")
    ax.set_xlabel("State")
    ax.set_ylabel("Value scale")

    ax.set_yticks(true_v)
    ax.set_yticklabels(true_v)

    x_values = np.arange(len(state_distribution))
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values)

    ax.margins(x=0)
    ax.tick_params(axis="y")

    ax2 = ax.twinx()

    ax2.plot(np.arange(len(true_v)), state_distribution, color="grey", alpha=0.05)
    ax2.fill_between(
        np.arange(len(true_v)), state_distribution, color="grey", alpha=0.05
    )
    ax2.set_ylabel("Distribution scale")
    ax2.set_yticks([np.min(state_distribution), np.max(state_distribution)])
    ax2.set_yticklabels([np.min(state_distribution), np.max(state_distribution)])
    ax2.tick_params(axis="y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    get_fig(5)
    get_fig(19)
