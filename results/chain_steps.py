from pathlib import Path

import numpy as np

N = 5
episode_steps = np.load(
    Path(__file__).parents[0] / "Chain" / f"Chain{N}_episode_steps.npy"
)
n_runs, n_episodes = episode_steps.shape
avg_episode = np.mean(episode_steps, axis=1)
print(
    f"Avg. episode length over {n_runs} runs and {n_episodes} episodes in Chain{N} with a random policy: "
    f"mean: {np.mean(avg_episode):.0f},\t"
    f"se: {np.std(avg_episode) / np.sqrt(n_runs):.2f},\t"
    f"\tmax: {np.max(avg_episode):.0f},"
    f"\tmin: {np.min(avg_episode):.0f}"
)
