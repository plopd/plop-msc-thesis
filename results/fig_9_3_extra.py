import matplotlib.pyplot as plt
import numpy as np

from representations.representations import get_representation


def get_fig():
    num_states = 19
    num_dims = 1
    order = 5
    num_features = (order + 1) ** num_dims
    states = np.arange(num_states).reshape((-1, num_dims))

    FR = get_representation(
        name="P",
        **{
            "order": order,
            "num_dims": num_dims,
            "min_x": 0,
            "max_x": len(states) - 1,
            "a": -1,
            "b": 1,
        }
    )

    features = np.array([FR[states[i]] for i in range(num_states)])

    fig = plt.figure(figsize=(7, 5), dpi=150)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax = fig.add_subplot(1, 1, 1)
    for i in range(1, num_features + 1):
        ax.scatter(states[:, 0], features[:, i - 1], s=5)
        plt.plot(states[:, 0], features[:, i - 1])
        ax.set_xticks(np.arange(num_states).tolist())
        ax.set_xticklabels(np.arange(num_states).tolist())
        ax.set_yticks([-1, 1], [-1, 1])
    plt.show()


if __name__ == "__main__":
    get_fig()
