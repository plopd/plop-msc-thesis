from pathlib import Path

import gym
import gym_puddle  # noqa f401
import numpy as np

from agents.policies import get_action_from_policy  # noqa f401
from utils.decorators import timer
from utils.utils import path_exists


@timer
def main():
    env = gym.make("PuddleWorld-v0")
    n_runs = 30
    n_episodes = 1000
    # How to use TimeLimit's _elapsed_steps instead of own increment?
    steps = np.zeros((n_runs, n_episodes))

    for i in range(n_runs):
        rand_generator = np.random.RandomState(i)
        env.reset()
        for j in range(n_episodes):
            done = False
            while not done:
                # action = env.action_space.sample()
                action = get_action_from_policy("semi-random-puddle", rand_generator)
                observation, reward, done, info = env.step(action)
                # env.render("human")
                steps[i][j] += 1
                if done:
                    env.reset()
            print(f"Episode: {j}: Length: {steps[i][j]}")
        print(f"Run: {i},\tAvg. Steps: {np.mean(steps[i]):.2f} done.", end="\n")
        # sys.stdout.flush()
        path = Path(__file__).parents[1] / "results" / "PuddleWorld"
        path_exists(path)
        np.save(path / f"episode_steps.npy", steps)
    env.close()


if __name__ == "__main__":
    main()
