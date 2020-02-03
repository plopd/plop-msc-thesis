import sys

import numpy as np
from tqdm import tqdm

import agents.agents as agents
import environments.environments as envs
from rl_glue.rl_glue import RLGlue


def calculate_state_distribution(N):
    agent_info = {
        "N": N,
        "algorithm": "TD",
        "representations": "tabular",
        "gamma": 1,
        "lmbda": 0,
        "alpha": 0.125,
        "seed": 0,
        "interest": "UI",
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


N_STATES = 1000

# all states
STATES = np.arange(0, N_STATES)

# start from a central state
START_STATE = 500

# terminal states
END_STATES = [-1, N_STATES]

# possible actions
ACTION_LEFT = -1
ACTION_RIGHT = 1
ACTIONS = [ACTION_LEFT, ACTION_RIGHT]

# maximum stride for an action
STEP_RANGE = 100


def compute_true_value():
    # true state value, just a promising guess
    true_value = np.arange(-1001, 1003, 2) / 1001.0

    # Dynamic programming to find the true state values, based on the promising guess above
    # Assume all rewards are 0, given that we have already given value -1 and 1 to terminal states
    while True:
        old_value = np.copy(true_value)
        for state in STATES:
            true_value[state] = 0
            for action in ACTIONS:
                for step in range(1, STEP_RANGE + 1):
                    step *= action
                    next_state = state + step
                    next_state = max(min(next_state, N_STATES + 1), 0)
                    # asynchronous update for faster convergence
                    true_value[state] += 1.0 / (2 * STEP_RANGE) * true_value[next_state]
        error = np.sum(np.abs(old_value - true_value))
        if error < 1e-2:
            break
    # correct the state value for terminal states to 0
    true_value[0] = true_value[-1] = 0

    return true_value


if __name__ == "__main__":
    N = int(sys.argv[1])
    print(calculate_state_distribution(N))
    # true_value = compute_true_value()
    # np.save("/Users/saipiens/scratch/Chain/true_v_1000", true_value, allow_pickle=True)
