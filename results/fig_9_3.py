import matplotlib.pyplot as plt
import numpy as np

from representations.representations import get_representation


def get_fig():
    num_states = 19
    num_dims = 1
    order = 9
    num_features = (order + 1) ** num_dims
    states = np.arange(num_states).reshape((-1, num_dims))

    FR = get_representation(
        name="F",
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

    fig = plt.figure(figsize=(4, order * 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(1, num_features + 1):
        ax = fig.add_subplot(num_features, 1, i)
        ax.scatter(states[:, 0], features[:, i - 1], s=5)
        ax.set_xticks(np.arange(num_states).tolist())
        ax.set_xticklabels(np.arange(num_states).tolist())
        ax.set_yticks([-1, 1], [-1, 1])
    plt.show()


if __name__ == "__main__":
    get_fig()
