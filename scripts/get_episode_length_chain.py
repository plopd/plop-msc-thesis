import sys
from pathlib import Path

import numpy as np

import agents.agents as agents
import environments.environments as envs
from rl_glue.rl_glue import RLGlue
from utils.decorators import timer
from utils.utils import path_exists


@timer
def main():
    n_runs = 30
    n_episodes = 10000
    steps = np.zeros((n_runs, n_episodes))
    N = int(sys.argv[1])

    agent_info = {
        "N": N,
        "algorithm": "td",
        "features": "tabular",
        "in_features": N,
        "gamma": 1,
        "lmbda": 0,
        "alpha": 0.125,
        "seed": None,
        "interest": "uniform",
    }

    env_info = {"env": "chain", "N": N}

    exp_info = {
        "max_timesteps_episode": 0,
        "episode_eval_freq": 1,
        "n_episodes": n_episodes,
    }

    for i in range(n_runs):
        agent_info["seed"] = i
        rl_glue = RLGlue(
            envs.get_environment(env_info["env"]),
            agents.get_agent(agent_info["algorithm"]),
        )
        rl_glue.rl_init(agent_info, env_info)
        for j in range(n_episodes):
            num_steps_before_episode = rl_glue.num_steps
            rl_glue.rl_episode(exp_info.get("max_timesteps_episode"))
            num_steps_after_episode = rl_glue.num_steps
            steps[i, j] = num_steps_after_episode - num_steps_before_episode
        print(f"\rRun: {i},\tAvg. Steps: {np.mean(steps[i]):.2f} done.", end="")
        sys.stdout.flush()

    path = Path(__file__).parents[1] / "results" / "Chain"
    path_exists(path)
    np.save(path / f"Chain{N}_episode_steps.npy", steps)


if __name__ == "__main__":
    main()
