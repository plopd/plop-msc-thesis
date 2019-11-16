import sys

import numpy as np
from tqdm import tqdm

import agents.agents as agents
import environments.environments as envs
from rl_glue.rl_glue import RLGlue


def calculate_state_distribution(N):
    agent_info = {
        "N": N,
        "algorithm": "td",
        "features": "tabular",
        "gamma": 1,
        "lmbda": 0,
        "alpha": 0.125,
        "seed": 0,
        "interest": "uniform",
    }

    env_info = {"env": "chain", "N": N}

    exp_info = {
        "max_timesteps_episode": 1000000,
        "episode_eval_freq": 1,
        "n_episodes": 1,
    }

    rl_glue = RLGlue(
        envs.get_environment(env_info["env"]), agents.get_agent(agent_info["algorithm"])
    )

    rl_glue.rl_init(agent_info, env_info)

    eta = np.zeros(env_info["N"])
    last_state, _ = rl_glue.rl_start()
    for _ in tqdm(range(1, int(exp_info["max_timesteps_episode"]) + 1)):
        eta[last_state] += 1
        _, last_state, _, term = rl_glue.rl_step()
        if term:
            last_state, _ = rl_glue.rl_start()

    state_distribution = eta / np.sum(eta)

    return state_distribution


if __name__ == "__main__":
    N = int(sys.argv[1])
    print(calculate_state_distribution(N))
