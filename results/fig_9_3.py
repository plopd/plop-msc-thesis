import matplotlib.pyplot as plt
import numpy as np

from utils.utils import get_feature


def get_fig():
    n_states = 1000
    in_features = 1
    out_features = 5
    states = np.random.uniform(0, 1, (n_states, in_features))

    features = np.array(
        [
            get_feature(
                states[i],
                **{"order": 4, "features": "fourier", "in_features": out_features},
                unit_norm=False
            )
            for i in range(n_states)
        ]
    )

    fig = plt.figure(figsize=(20, 4))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(1, out_features + 1):
        ax = fig.add_subplot(1, out_features, i)
        ax.scatter(states[:, 0], features[:, i - 1])
    plt.show()


if __name__ == "__main__":
    get_fig()
