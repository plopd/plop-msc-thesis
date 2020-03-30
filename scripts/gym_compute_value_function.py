import os
import sys
from pathlib import Path

import gym
import gym_puddle  # noqa f401
import numpy as np
from alphaex.sweeper import Sweeper

from agents.policies import get_action_from_policy
from utils.utils import path_exists


def main():
    gym_compute_value_function(sweep_id=int(sys.argv[1]), config_fn=sys.argv[2])


def gym_compute_value_function(sweep_id, config_fn):
    config_root_path = Path(__file__).parents[1] / "configs"
    sweeper = Sweeper(config_root_path / f"{config_fn}.json")
    config = sweeper.parse(sweep_id)
    env_id = config.get("env_id")
    policy_name = config.get("policy_name")
    save_rootpath = Path(f"{os.environ.get('SCRATCH')}") / f"{config.get('problem')}"
    save_rootpath = path_exists(save_rootpath)
    num_episode = config.get("num_episode")
    discount_rate = config.get("discount_rate")

    observations = np.load(save_rootpath / "S.npy")

    # Get true values by averaging returns
    true_values = np.zeros(1,)
    env = gym.make(env_id)
    env.seed(sweep_id)
    Gs = np.zeros(num_episode)
    for n_e in range(num_episode):
        rand_generator = np.random.RandomState(n_e)
        rewards = []
        done = False
        env.reset()
        env.state = np.copy(observations[sweep_id])
        obs = env.state
        while not done:
            action = get_action_from_policy(policy_name, obs, rand_generator)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
        if done:
            G = np.sum([discount_rate ** j * r for j, r in enumerate(rewards)])
            Gs[n_e] = G
    true_value = np.mean(Gs)
    true_values[0] = true_value
    np.save(
        save_rootpath / f"{sweep_id}-discount_rate_{discount_rate}".replace(".", "_"),
        true_values,
    )


if __name__ == "__main__":
    main()
