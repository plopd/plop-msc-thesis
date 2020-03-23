import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--num_obs", type=int, required=True)
parser.add_argument("--env_id", type=str, required=True)
parser.add_argument("--discount_rate", type=float, required=True)
parser.add_argument("--policy_name", type=str, required=True)
parser.add_argument("--problem", type=str, required=True)

args = parser.parse_args()


save_rootpath = Path(f"{os.environ.get('SCRATCH')}") / args.problem
num_obs = args.num_obs
env_id = args.env_id
discount_rate = args.discount_rate
policy_name = args.policy_name

observations = np.load(save_rootpath / "S.npy")
env = gym.make(env_id)

# Plot states S
plt.figure()
plt.scatter(observations[:, 0], observations[:, 1], alpha=0.15)
plt.xlim((env.observation_space.low[0], env.observation_space.high[0]))
plt.ylim((env.observation_space.low[1], env.observation_space.high[1]))
plt.savefig((Path(save_rootpath) / "observation_space"))

# Plot true values
filename = f"true_values-discount_rate_{discount_rate}".replace(".", "_")
true_values = np.load(Path(save_rootpath) / f"{filename}.npy")
colors = minmax_normalization_ab(
    true_values,
    true_values.min(),
    true_values.max(),
    true_values.min(),
    true_values.max(),
)
plt.figure()
sc = plt.scatter(observations[:, 0], observations[:, 1], c=colors, cmap="hot")
plt.xlim((env.observation_space.low[0], env.observation_space.high[0]))
plt.ylim((env.observation_space.low[1], env.observation_space.high[1]))
plt.colorbar(sc)
plt.title(f"{env_id} {policy_name} Prediction")
plt.tight_layout()
plt.savefig(
    (
        Path(save_rootpath)
        / f"true_values-discount_rate_{discount_rate}".replace(".", "_")
    )
)
