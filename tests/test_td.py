import numpy as np
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


@pytest.mark.parametrize("algorithm", ["td", "etd"])
def test_agent_start(algorithm):
    environment = get_environment(env_info["env"])
    agent_info["algorithm"] = algorithm
    agent = get_agent(agent_info["algorithm"])

    rl_glue = RLGlue(environment, agent)
    rl_glue.rl_init(agent_init_info=agent_info, env_init_info=env_info)

    z = rl_glue.rl_agent_message("get eligibility trace")
    w = rl_glue.rl_agent_message("get weight vector")

    try:
        M = rl_glue.rl_agent_message("get emphasis vector")
        F = rl_glue.rl_agent_message("get weight vector")
        assert F == 0.0
        assert M == 0.0
    except Exception:
        pass

    assert np.allclose(z, np.zeros(z.shape[0]))

    assert np.allclose(w, np.zeros(w.shape[0]))


def test_followon_trace_for_constant_gamma_lambda_interest():
    environment = get_environment(env_info["env"])
    agent = get_agent("etd")

    rl_glue = RLGlue(environment, agent)

    for episode in range(1, 3):
        rl_glue.rl_init(agent_init_info=agent_info, env_init_info=env_info)
        rl_glue.rl_episode(0)
        assert rl_glue.rl_agent_message("get followon trace") - 1 == rl_glue.num_steps


@pytest.mark.parametrize("algorithm", ["td", "etd"])
def test_eligibility_trace_reset_at_start_of_episode(algorithm):
    environment = get_environment(env_info["env"])
    agent_info["algorithm"] = algorithm
    agent = get_agent(agent_info["algorithm"])

    rl_glue = RLGlue(environment, agent)

    for episode in range(1, 3):
        rl_glue.rl_init(agent_init_info=agent_info, env_init_info=env_info)
        assert np.allclose(
            rl_glue.rl_agent_message("get eligibility trace"),
            np.zeros(rl_glue.rl_agent_message("get weight vector").shape[0]),
        )
        rl_glue.rl_episode(0)
