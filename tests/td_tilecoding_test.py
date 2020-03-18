import numpy as np
import pytest

from agents.agents import get_agent
from environments.environments import get_environment
from rl_glue.rl_glue import RLGlue

agent_info = {
    "num_states": 19,
    "representations": "TC",
    "tiles_per_dim": "18",
    "tilings": 1,
    "min_x": "0",
    "max_x": "18",
    "discount_rate": 0.99,
    "trace_decay": 0.5,
    "interest": "UI",
    "step_size": 0.001,
    "policy": "random-chain",
}

env_info = {"env": "RandomWalk", "num_states": 19}
environment = get_environment(env_info["env"])


@pytest.mark.parametrize("algorithm", ["ETDTileCoding", "TDTileCoding"])
def test_agent_start(algorithm):
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

    assert np.array_equal(z, np.zeros(z.shape[0]))

    assert np.array_equal(w, np.zeros(w.shape[0]))


def test_linear_followon_trace():
    agent_info["discount_rate"] = 1.0
    agent_info["trace_decay"] = 0.0
    agent_info["interest"] = "UI"
    agent = get_agent("ETDTileCoding")

    rl_glue = RLGlue(environment, agent)

    for episode in range(1, 3):
        rl_glue.rl_init(agent_init_info=agent_info, env_init_info=env_info)
        rl_glue.rl_episode(0)
        assert rl_glue.rl_agent_message("get followon trace") - 1 == rl_glue.num_steps


def test_constant_emphasis():
    agent_info["discount_rate"] = 1.0
    agent_info["trace_decay"] = 1.0
    agent_info["interest"] = "UI"
    agent = get_agent("ETDTileCoding")

    rl_glue = RLGlue(environment, agent)

    for episode in range(1, 3):
        rl_glue.rl_init(agent_init_info=agent_info, env_init_info=env_info)
        rl_glue.rl_episode(0)
        assert rl_glue.rl_agent_message("get emphasis trace") == 1.0


@pytest.mark.parametrize("algorithm", ["TDTileCoding", "ETDTileCoding"])
def test_eligibility_trace_reset_at_start_of_episode(algorithm):
    agent_info["algorithm"] = algorithm
    agent = get_agent(agent_info["algorithm"])

    rl_glue = RLGlue(environment, agent)
    rl_glue.rl_init(agent_init_info=agent_info, env_init_info=env_info)
    rl_glue.rl_start()
    e = rl_glue.rl_agent_message("get eligibility trace")
    assert np.allclose(e, np.zeros(e.shape[0]))


@pytest.mark.parametrize("algorithm", ["ETDTileCoding"])
def test_emphasis_reset_at_start_of_episode(algorithm):
    agent_info["algorithm"] = algorithm
    agent = get_agent(agent_info["algorithm"])

    rl_glue = RLGlue(environment, agent)

    rl_glue.rl_init(agent_init_info=agent_info, env_init_info=env_info)
    rl_glue.rl_start()
    assert rl_glue.rl_agent_message("get emphasis trace") == 0.0
