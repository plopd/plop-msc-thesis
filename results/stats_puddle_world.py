from pathlib import Path

import numpy as np


def main():
    episode_steps = np.load(
        Path(__file__).parents[0] / "PuddleWorld" / "episode_steps.npy"
    )
    n_runs, n_episodes = episode_steps.shape
    avg_episode = np.mean(episode_steps, axis=1)
    print(
        f"Avg. episode length over {n_runs} runs and "
        f"{n_episodes} episodes in PuddleWorld with a random policy: "
        f"Mean: {np.mean(avg_episode):.0f},\t"
        f"Se: {np.std(avg_episode) / np.sqrt(n_runs):.2f},\t"
        f"\tMax: {np.max(episode_steps):.0f},"
        f"\tMin: {np.min(episode_steps):.0f}"
    )


if __name__ == "__main__":
    main()
