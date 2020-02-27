import numpy as np
import pytest

from agents.agents import get_agent
from environments.environments import get_environment
from rl_glue.rl_glue import RLGlue

agent_info = {
    "num_states": 5,
    "algorithm": "ETD",
    "representations": "TA",
    "num_dims": 5,
    "discount_rate": 1.0,
    "trace_decay": 0.0,
    "step_size": 0.001,
    "seed": 42,
    "interest": "UI",
    "policy": "random-chain",
}

env_info = {"env": "Chain", "num_states": 5}


@pytest.mark.parametrize("num_states", [5, 19])
def test_chain_init(num_states):
    environment = get_environment(env_info["env"])
    env_info["num_states"] = num_states
    agent_info["num_states"] = num_states
    agent_info["num_dims"] = num_states
    agent = get_agent(agent_info["algorithm"])
    rl_glue = RLGlue(environment, agent)
    rl_glue.rl_init(agent_init_info=agent_info, env_init_info=env_info)

    (last_state, _) = rl_glue.rl_start()

    assert last_state == num_states // 2


@pytest.mark.parametrize("algorithm", ["TD", "ETD"])
def test_same_walks_per_run_for_each_algorithm(algorithm):
    runs_with_episodes = {
        0: [[2, 1, 2, 3, 2, 3, 4], [2, 3, 4], [2, 3, 2, 1, 2, 1, 0]],
        1: [
            [2, 3, 4, 3, 2, 3, 4],
            [2, 3, 4, 3, 2, 3, 2, 3, 4, 3, 2, 3, 2, 1, 0, 1, 0],
            [2, 3, 2, 1, 0, 1, 0],
        ],
    }

    env_info["log_episodes"] = 1
    env_info["num_states"] = 5
    agent_info["num_states"] = 5
    agent_info["num_dims"] = 5
    for i in range(len(runs_with_episodes)):
        agent_info["algorithm"] = algorithm
        agent_info["seed"] = i
        rl_glue = RLGlue(
            get_environment(env_info["env"]), get_agent(agent_info["algorithm"])
        )
        rl_glue.rl_init(agent_info, env_info)
        for j in range(len(runs_with_episodes[i])):
            rl_glue.rl_episode(0)
            assert np.array_equiv(
                runs_with_episodes[i][j],
                np.array(rl_glue.rl_env_message("get episode")).squeeze(),
            )
