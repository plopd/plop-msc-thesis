import pytest

from agents.agents import get_agent
from environments.environments import get_environment
from rl_glue.rl_glue import RLGlue

agent_info = {
    "N": 19,
    "algorithm": "etd",
    "features": "tabular",
    "order": None,
    "n": None,
    "num_ones": None,
    "gamma": 1.0,
    "lmbda": 0.0,
    "alpha": 2 ** -7,
    "seed": None,
    "interest": "uniform",
}

env_info = {"env": "chain", "N": 19}


@pytest.mark.parametrize("N", [5, 19])
def test_chain_init(N):
    environment = get_environment(env_info["env"])
    env_info["N"] = N
    agent_info["N"] = N
    agent = get_agent(agent_info["algorithm"])
    rl_glue = RLGlue(environment, agent)
    rl_glue.rl_init(agent_init_info=agent_info, env_init_info=env_info)

    (last_state, _) = rl_glue.rl_start()

    assert last_state == N // 2
