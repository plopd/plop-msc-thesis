import time

import matplotlib.pyplot as plt
import numpy as np

from utils.utils import get_bases_features

states = np.random.uniform(0, 1, 100_000).reshape(-1, 2)

# start = time.time()
# features = get_bases_features(states, order=5, kind="fourier")
# end = time.time()
# print(f"{end-start}s")
# N, n = features.shape
#
# fig = plt.figure(figsize=(25, 25))
# fig.subplots_adjust(hspace=0.4, wspace=0.4)
#
# for i in range(1, n + 1):
#     ax = fig.add_subplot(int(np.sqrt(n)), int(np.sqrt(n)), i)
#     ax.scatter(states[:, 0], states[:, 1], c=features[:, i - 1], cmap="bone")
# plt.title("Fourier Basis")
# plt.tight_layout()
# plt.show()

start = time.time()
features = get_bases_features(states, order=5, kind="poly")
end = time.time()
print(f"{end-start}s")

N, n = features.shape
fig = plt.figure(figsize=(25, 25))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(1, n + 1):
    ax = fig.add_subplot(int(np.sqrt(n)), int(np.sqrt(n)), i)
    ax.scatter(states[:, 0], states[:, 1], c=features[:, i - 1], cmap="bone")
plt.title("Polynomial Basis")
plt.tight_layout()
plt.show()
