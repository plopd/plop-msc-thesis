import matplotlib.pyplot as plt
import numpy as np

from utils.utils import get_bases_features

states = np.random.uniform(0, 1, 500).reshape(-1, 1)
features = get_bases_features(states, order=4, kind="fourier")
N, n = features.shape

fig = plt.figure(figsize=(20, 4))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(1, n + 1):
    ax = fig.add_subplot(1, n, i)
    ax.scatter(states[:, 0], features[:, i - 1])
plt.show()
