from pathlib import Path

import numpy as np

episode_steps = np.load(Path(__file__).parents[0] / "PuddleWorld" / "episode_steps.npy")
n_runs = len(episode_steps)
print(
    f"Avg. episode length over {n_runs} runs in PuddleWorld with a random policy: "
    f"mean: {np.mean(episode_steps):.0f},\t"
    f"se: {np.std(episode_steps) / np.sqrt(n_runs):.2f},\t"
    f"\tmax: {np.max(episode_steps):.0f},"
    f"\tmin: {np.min(episode_steps):.0f}"
)
