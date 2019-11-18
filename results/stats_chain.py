import sys
from pathlib import Path

import numpy as np


def main():
    N = int(sys.argv[1])
    episode_steps = np.load(
        Path(__file__).parents[0] / "Chain" / f"Chain{N}_episode_steps.npy"
    )
    n_runs, n_episodes = episode_steps.shape
    avg_episode = np.mean(episode_steps, axis=1)
    print(
        f"Avg. episode length over {n_runs} runs "
        f"and {n_episodes} episodes in Chain{N} with a random policy: "
        f"Mean: {np.mean(avg_episode):.0f},\t"
        f"Se: {np.std(avg_episode) / np.sqrt(n_runs):.2f},\t"
        f"\tMax: {np.max(episode_steps):.0f},"
        f"\tMin: {np.min(episode_steps):.0f}"
    )


if __name__ == "__main__":
    main()
