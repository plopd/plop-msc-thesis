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
    "in_features": 19,
    "num_ones": None,
    "gamma": 1.0,
    "lmbda": 0.0,
    "interest": "uniform",
    "alpha": 2 ** -7,
    "seed": None,
    "policy": "random-chain",
}

env_info = {"env": "chain", "N": 19}


@pytest.mark.parametrize("algorithm", ["elstd"])
def test_agent_start(algorithm):
    environment = get_environment(env_info["env"])
    agent_info["algorithm"] = algorithm
    agent = get_agent(agent_info["algorithm"])

    rl_glue = RLGlue(environment, agent)
    rl_glue.rl_init(agent_info, env_info)
    rl_glue.rl_start()

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


def test_linear_followon_trace():
    agent_info["gamma"] = 1.0
    agent_info["lmbda"] = 0.0
    agent_info["interest"] = "uniform"
    environment = get_environment(env_info["env"])
    agent = get_agent("etd")

    rl_glue = RLGlue(environment, agent)

    for episode in range(1, 3):
        rl_glue.rl_init(agent_init_info=agent_info, env_init_info=env_info)
        rl_glue.rl_episode(0)
        assert rl_glue.rl_agent_message("get followon trace") - 1 == rl_glue.num_steps


def test_constant_emphasis():
    agent_info["gamma"] = 1.0
    agent_info["lmbda"] = 1.0
    agent_info["interest"] = "uniform"
    environment = get_environment(env_info["env"])
    agent = get_agent("etd")

    rl_glue = RLGlue(environment, agent)

    for episode in range(1, 3):
        rl_glue.rl_init(agent_init_info=agent_info, env_init_info=env_info)
        rl_glue.rl_episode(0)
        assert rl_glue.rl_agent_message("get emphasis trace") == 1.0


@pytest.mark.parametrize("algorithm", ["td", "etd", "lstd", "elstd"])
def test_eligibility_trace_reset_at_start_of_episode(algorithm):
    environment = get_environment(env_info["env"])
    agent_info["algorithm"] = algorithm
    agent = get_agent(agent_info["algorithm"])

    rl_glue = RLGlue(environment, agent)
    rl_glue.rl_init(agent_init_info=agent_info, env_init_info=env_info)
    rl_glue.rl_start()
    e = rl_glue.rl_agent_message("get eligibility trace")
    assert np.allclose(e, np.zeros(e.shape[0]))


@pytest.mark.parametrize("algorithm", ["etd", "elstd"])
def test_emphasis_reset_at_start_of_episode(algorithm):
    environment = get_environment(env_info["env"])
    agent_info["algorithm"] = algorithm
    agent = get_agent(agent_info["algorithm"])

    rl_glue = RLGlue(environment, agent)

    rl_glue.rl_init(agent_init_info=agent_info, env_init_info=env_info)
    rl_glue.rl_start()
    assert rl_glue.rl_agent_message("get emphasis trace") == 0.0


@pytest.mark.parametrize(
    "env, algorithm", [("deterministic-chain", "td"), ("deterministic-chain", "etd")]
)
def test_one_step_td_update(env, algorithm):
    agent_info["algorithm"] = algorithm
    agent_info["lmbda"] = 1.0
    agent_info["gamma"] = 0.0
    env_info["env"] = env
    agent = get_agent(agent_info["algorithm"])
    environment = get_environment(env_info["env"])

    rl_glue = RLGlue(environment, agent)

    for episode in range(1, 2):
        rl_glue.rl_init(agent_init_info=agent_info, env_init_info=env_info)
        rl_glue.rl_episode(0)
        weight_vector = rl_glue.rl_agent_message("get weight vector")

        assert np.allclose(
            weight_vector[:-episode], np.zeros(weight_vector.shape[0] - episode)
        )
