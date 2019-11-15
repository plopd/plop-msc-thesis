import matplotlib.pyplot as plt
import numpy as np

from utils.utils import get_feature


def get_fig(name):
    n_states = 10000
    in_features = 2
    out_features = 36
    states = np.random.uniform(0, 1, (n_states, in_features))

    features = np.array(
        [
            get_feature(states[i], **{"order": 5, "features": name})
            for i in range(n_states)
        ]
    )

    fig = plt.figure(figsize=(25, 25))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(1, out_features + 1):
        ax = fig.add_subplot(int(np.sqrt(out_features)), int(np.sqrt(out_features)), i)
        ax.scatter(states[:, 0], states[:, 1], c=features[:, i - 1], cmap="bone")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    get_fig("fourier")
    get_fig("poly")
