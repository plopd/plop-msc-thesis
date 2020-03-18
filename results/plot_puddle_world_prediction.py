import os
from pathlib import Path

import gym
import gym_puddle  # noqa f401
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.utils import minmax_normalization_ab

matplotlib.rcParams.update({"font.size": 24})
sns.set()

save_rootpath = Path(f"{os.environ.get('SCRATCH')}") / "PuddleWorld"
num_obs = 500
env_id = "PuddleWorld-v0"
discount_rate = 0.99
policy_name = "semi-random-policy"

observations = np.load(save_rootpath / f"S_{num_obs}.npy")
env = gym.make(env_id)

# Plot states S
plt.figure()
plt.scatter(observations[:, 0], observations[:, 1], alpha=0.15)
plt.xlim((env.observation_space.low[0], env.observation_space.high[0]))
plt.ylim((env.observation_space.low[1], env.observation_space.high[1]))
plt.savefig((Path(save_rootpath) / "observation_space"))

# Plot true values
filename = f"true_values_{discount_rate}".replace(".", "")
true_values = np.load(Path(save_rootpath) / f"{filename}.npy")
colors = minmax_normalization_ab(
    true_values, true_values.min(), true_values.max(), -1, 1
)
plt.figure()
sc = plt.scatter(observations[:, 0], observations[:, 1], c=colors, cmap="hot")
plt.xlim((env.observation_space.low[0], env.observation_space.high[0]))
plt.ylim((env.observation_space.low[1], env.observation_space.high[1]))
plt.colorbar(sc)
plt.title(f"{env_id} {policy_name} Prediction")
plt.tight_layout()
plt.savefig((Path(save_rootpath) / f"true_values_{discount_rate}".replace(".", "")))
