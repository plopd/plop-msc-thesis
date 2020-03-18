import pytest

from agents.agents import get_agent
from environments.environments import get_environment
from rl_glue.rl_glue import RLGlue

agent_info = {
    "num_states": 19,
    "representations": "TA",
    "num_dims": 19,
    "discount_rate": 0.99,
    "trace_decay": 0.5,
    "interest": "UI",
    "seed": 0,
    "policy": "random-chain",
}

env_info = {"env": "RandomWalk", "num_states": 19}


@pytest.mark.parametrize("algorithm", ["ELSTD"])
def test_increasing_steps_over_episodes(algorithm):
    environment = get_environment(env_info["env"])
    agent_info["algorithm"] = algorithm
    agent = get_agent(agent_info["algorithm"])

    rl_glue = RLGlue(environment, agent)
    rl_glue.rl_init(agent_info, env_info)

    for episode in range(1, 10):
        total_timesteps_before_episode = rl_glue.rl_agent_message("get steps")
        rl_glue.rl_episode(0)
        total_timesteps_after_episode = rl_glue.rl_agent_message("get steps")
        assert total_timesteps_after_episode - total_timesteps_before_episode > 0
