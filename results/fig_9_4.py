import matplotlib.pyplot as plt
import numpy as np

from representations.representations import get_representation


def get_fig(name):
    num_states = 10000
    num_dims = 2
    num_features = 36
    states = np.random.uniform(0, 1, (num_states, num_dims))

    FR = get_representation(
        name=name,
        **{
            "order": 5,
            "num_dims": num_dims,
            "min_x": states.min(),
            "max_x": states.max(),
        }
    )

    features = np.array([FR[states[i]] for i in range(num_states)])

    fig = plt.figure(figsize=(25, 25))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(1, num_features + 1):
        ax = fig.add_subplot(int(np.sqrt(num_features)), int(np.sqrt(num_features)), i)
        ax.scatter(states[:, 0], states[:, 1], c=features[:, i - 1], cmap="bone")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    get_fig("F")
    get_fig("P")
