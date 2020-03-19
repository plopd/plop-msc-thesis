import argparse
import os
from pathlib import Path

import gym
import gym_puddle  # noqa f401
import numpy as np
from tqdm import tqdm

from agents.policies import get_action_from_policy
from utils.utils import path_exists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--num_obs", type=int, required=True)
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--discount_rate", type=float, required=True)
    parser.add_argument("--num_episode", type=int, required=True)
    parser.add_argument("--policy_name", type=str, required=True)
    parser.add_argument("--problem", type=str, required=True)

    args = parser.parse_args()

    save_rootpath = Path(f"{os.environ.get('SCRATCH')}") / f"{args.problem}"
    save_rootpath = path_exists(save_rootpath)
    args.__dict__["save_rootpath"] = save_rootpath

    simulate_on_policy(**args.__dict__)
    compute_value_function(**args.__dict__)


def simulate_on_policy(**kwargs):
    env_id = kwargs.get("env_id")
    steps = kwargs.get("steps")
    policy_name = kwargs.get("policy_name")
    save_rootpath = kwargs.get("save_rootpath")

    env = gym.make(env_id)
    seed = 0
    env.seed(seed=seed)
    rand_generator = np.random.RandomState(seed)
    obs = env.reset()
    observations = []
    for _ in tqdm(range(steps)):
        observations.append(obs)
        action = get_action_from_policy(policy_name, obs, rand_generator)
        obs, reward, done, info = env.step(action)
        # env.render("human")
        if done:
            obs = env.reset()
    env.close()
    np.save(save_rootpath / "S", observations)


def compute_value_function(**kwargs):
    env_id = kwargs.get("env_id")
    steps = kwargs.get("steps")
    policy_name = kwargs.get("policy_name")
    save_rootpath = kwargs.get("save_rootpath")
    num_obs = kwargs.get("num_obs")
    num_episode = kwargs.get("num_episode")
    discount_rate = kwargs.get("discount_rate")

    observations = np.load(save_rootpath / "S.npy")
    idxs = np.random.choice(
        np.arange(steps // 2, steps), size=(num_obs,), replace=False
    )
    observations = observations[idxs, :]
    np.save(save_rootpath / f"S", observations)

    # Get true values by averaging returns
    true_values = np.zeros(num_obs)
    for i in tqdm(range(num_obs)):
        obs = observations[i]
        env = gym.make(env_id)
        env.seed(i)
        Gs = np.zeros(num_episode)
        for n_e in range(num_episode):
            rand_generator = np.random.RandomState(n_e)
            rewards = []
            done = False
            env.reset()
            env.state = obs
            while not done:
                action = get_action_from_policy(policy_name, obs, rand_generator)
                obs, reward, done, info = env.step(action)
                rewards.append(reward)
                if done:
                    G = np.sum([discount_rate ** i * r for i, r in enumerate(rewards)])
                    Gs[n_e] = G
        true_value = np.mean(Gs)
        true_values[i] = true_value
    np.save(save_rootpath / f"true_values", true_values)


if __name__ == "__main__":
    main()
