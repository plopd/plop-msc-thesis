import matplotlib.pyplot as plt
import numpy as np

from representations.representations import get_representation


def get_fig():
    num_states = 5
    num_dims = 1
    orders = [1, 2, 3]

    fig, axes = plt.subplots(
        ncols=1, nrows=len(orders), figsize=(7, 10), sharey="row", sharex="row"
    )

    for row, order in enumerate(orders):
        num_features = (order + 1) ** num_dims
        states = np.arange(num_states).reshape((-1, num_dims))

        kwargs_representation = {
            "order": order,
            "num_dims": num_dims,
            "min_x": 0,
            "max_x": len(states) - 1,
            "a": -1,
            "b": 1,
        }

        FR = get_representation(name="P", **kwargs_representation, unit_norm=True)
        print(FR.C.shape, FR.C)

        features = np.array([FR[states[i]] for i in range(num_states)])

        for i in range(1, num_features + 1):
            axes[row].scatter(states[:, 0], features[:, i - 1], s=5)
            axes[row].plot(states[:, 0], features[:, i - 1])
            axes[row].set_xticks(np.arange(num_states).tolist())
            axes[row].set_xticklabels(np.arange(num_states).tolist())
            axes[row].set_yticks(
                [kwargs_representation.get("a"), kwargs_representation.get("b")],
                [kwargs_representation.get("a"), kwargs_representation.get("b")],
            )
    plt.show()


if __name__ == "__main__":
    get_fig()
