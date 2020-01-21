import sys
from pathlib import Path

import gym
import gym_puddle  # noqa f401
import numpy as np
from tqdm import tqdm

from utils.decorators import timer
from utils.utils import path_exists


@timer
def main():
    env = gym.make("PuddleWorld-v0")
    n_runs = 1
    n_episodes = 1
    # How to use TimeLimit's _elapsed_steps instead of own increment?
    steps = np.zeros((n_runs, n_episodes))

    for i in range(n_runs):
        for j in tqdm(range(n_episodes)):
            env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                print("Action:", action)
                observation, reward, done, info = env.step(action)
                steps[i][j] += 1
        print(f"\rRun: {i},\tAvg. Steps: {np.mean(steps[i]):.2f} done.", end="")
        sys.stdout.flush()
        path = Path(__file__).parents[1] / "results" / "PuddleWorld"
        path_exists(path)
        np.save(path / f"episode_steps.npy", steps)
    env.close()


if __name__ == "__main__":
    main()