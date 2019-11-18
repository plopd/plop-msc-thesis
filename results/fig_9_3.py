import matplotlib.pyplot as plt
import numpy as np

from utils.utils import get_feature


def get_fig():
    n_states = 500
    in_features = 1
    order = 15
    out_features = (order + 1) ** in_features
    states = np.random.uniform(0, 1, (n_states, in_features))

    features = np.array(
        [
            get_feature(
                states[i],
                **{
                    "order": order,
                    "features": "fourier",
                    "in_features": out_features,
                    "v_min": 0,
                    "v_max": n_states - 1,
                },
                unit_norm=True
            )
            for i in range(n_states)
        ]
    )

    fig = plt.figure(figsize=(order * 3, 4))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(1, out_features + 1):
        ax = fig.add_subplot(1, out_features, i)
        ax.scatter(states[:, 0], features[:, i - 1])
    plt.show()


if __name__ == "__main__":
    get_fig()
