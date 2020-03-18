import numpy as np
import pytest

from agents.agents import get_agent
from environments.environments import get_environment
from representations.representations import get_representation
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

env_info = {"env": "RandomWalk", "num_states": 5}


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
    num_runs = len(runs_with_episodes)
    for i in range(num_runs):
        agent_info["algorithm"] = algorithm
        agent_info["seed"] = i
        rl_glue = RLGlue(
            get_environment(env_info["env"]), get_agent(agent_info["algorithm"])
        )
        rl_glue.rl_init(agent_info, env_info)
        num_episodes = len(runs_with_episodes[i])
        for j in range(num_episodes):
            rl_glue.rl_episode(0)
            assert np.array_equiv(
                runs_with_episodes[i][j],
                np.array(rl_glue.rl_env_message("get episode")).squeeze(),
            )


@pytest.mark.parametrize("representations", ["RB", "R"])
def test_same_feature_representation_for_one_trial(representations):
    agent_info = {
        "num_states": 19,
        "algorithm": "ETD",
        "representations": representations,
        "num_features": 18,
        "num_ones": 10,
        "discount_rate": 0.95,
        "trace_decay": 0.5,
        "step_size": 0.0001,
        "interest": "UI",
        "policy": "random-chain",
    }
    env_info = {"env": "RandomWalk", "num_states": 19}
    num_states = agent_info.get("num_states")
    for seed in np.arange(10):
        agent_info["seed"] = seed
        states = np.arange(num_states).reshape(-1, 1)
        RF = get_representation(agent_info.get("representations"), **agent_info)
        rl_glue = RLGlue(
            get_environment(env_info["env"]), get_agent(agent_info["algorithm"])
        )
        random_features = np.vstack([RF[states[i]] for i in range(num_states)])
        rl_glue.rl_init(agent_info, env_info)
        max_steps_this_episode = 0
        for i in range(10):
            is_terminal = False

            rl_glue.rl_start()

            while (not is_terminal) and (
                (max_steps_this_episode == 0)
                or (rl_glue.num_steps < max_steps_this_episode)
            ):
                rl_step_result = rl_glue.rl_step()
                is_terminal = rl_step_result[3]
                last_state = rl_step_result[2]
                np.array_equiv(
                    rl_glue.agent.FR[last_state], random_features[last_state]
                )
