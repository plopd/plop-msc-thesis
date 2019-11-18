import numpy as np
import pytest

from agents.agents import get_agent
from environments.environments import get_environment
from rl_glue.rl_glue import RLGlue

agent_info = {
    "N": 5,
    "algorithm": "etd",
    "features": "tabular",
    "in_features": 5,
    "order": None,
    "n": None,
    "num_ones": None,
    "gamma": 1.0,
    "lmbda": 0.0,
    "alpha": 2 ** -7,
    "seed": None,
    "interest": "uniform",
}

env_info = {"env": "chain", "N": 5}


@pytest.mark.parametrize("N", [5, 19])
def test_chain_init(N):
    environment = get_environment(env_info["env"])
    env_info["N"] = N
    agent_info["N"] = N
    agent_info["in_features"] = N
    agent = get_agent(agent_info["algorithm"])
    rl_glue = RLGlue(environment, agent)
    rl_glue.rl_init(agent_init_info=agent_info, env_init_info=env_info)

    (last_state, _) = rl_glue.rl_start()

    assert last_state == N // 2


@pytest.mark.parametrize("method", ["td", "etd"])
def test_same_episodes_for_each_run_for_different_methods(method):
    runs_with_episodes = {
        0: [[2, 1, 2, 3, 2, 3, 4], [2, 3, 4], [2, 3, 2, 1, 2, 1, 0]],
        1: [
            [2, 3, 4, 3, 2, 3, 4],
            [2, 3, 4, 3, 2, 3, 2, 3, 4, 3, 2, 3, 2, 1, 0, 1, 0],
            [2, 3, 2, 1, 0, 1, 0],
        ],
    }

    env_info["log_episodes"] = 1
    env_info["N"] = 5
    agent_info["N"] = 5
    agent_info["in_features"] = 5
    for i in range(len(runs_with_episodes)):
        agent_info["algorithm"] = method
        agent_info["seed"] = i
        rl_glue = RLGlue(
            get_environment(env_info["env"]), get_agent(agent_info["algorithm"])
        )
        rl_glue.rl_init(agent_info, env_info)
        for j in range(len(runs_with_episodes[i])):
            rl_glue.rl_episode(0)
            arr = np.array(rl_glue.rl_env_message("get episode")).squeeze().tolist()
            assert runs_with_episodes[i][j] == arr
