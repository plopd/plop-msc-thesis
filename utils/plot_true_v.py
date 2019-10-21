import matplotlib.pyplot as plt
import numpy as np

import utils.const as const


def plot_true_v_and_state_distribution(true_v, state_distribution):
    fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
    true_v = [f"{true_v[i]:.4f}" for i in range(len(true_v))]
    ax.plot(np.arange(len(true_v)) + 1, true_v, color="red", marker="o")
    # ax.set_title("True state value function (policy evaluation)")
    # ax.set_xlabel("State")
    # ax.set_ylabel("Value scale")

    ax.set_yticks(true_v)
    ax.set_yticklabels(true_v)

    x_values = np.arange(len(state_distribution) + 1)
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values)

    ax.margins(x=0)
    ax.tick_params(axis="y")

    ax2 = ax.twinx()

    ax2.plot(np.arange(len(true_v)) + 1, state_distribution, color="grey", alpha=0.05)
    ax2.fill_between(
        np.arange(len(true_v)) + 1, state_distribution, color="grey", alpha=0.05
    )
    # ax2.set_ylabel("Distribution scale")
    ax2.set_yticks([np.min(state_distribution), np.max(state_distribution)])
    ax2.set_yticklabels([np.min(state_distribution), np.max(state_distribution)])
    ax2.tick_params(axis="y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    true_v = np.load(
        f"{const.PATHS['project_path']}/data/true_v_5_states_random_walk.npy",
        allow_pickle=True,
    )
    state_distribution = np.load(
        f"{const.PATHS['project_path']}/data/state_distribution_5_states_random_walk.npy",
        allow_pickle=True,
    )
    plot_true_v_and_state_distribution(true_v, state_distribution)
