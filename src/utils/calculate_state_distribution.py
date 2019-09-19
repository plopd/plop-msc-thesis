import sys

import numpy as np
import utils.const as const
import yaml
from rl_glue.rl_glue import RLGlue
from tqdm import tqdm


def calculate_state_distribution(agent_info, env_info, experiment_info):

    rl_glue = RLGlue(const.ENVS[env_info["env"]], const.AGENTS[agent_info["agent"]])

    rl_glue.rl_init(agent_info, env_info)

    eta = np.zeros(env_info["N"])
    last_state, _ = rl_glue.rl_start()
    for _ in tqdm(range(1, int(experiment_info["max_timesteps_episode"]) + 1)):
        eta[last_state - 1] += 1
        _, last_state, _, term = rl_glue.rl_step()
        if term:
            last_state, _ = rl_glue.rl_start()

    state_distribution = eta / np.sum(eta)

    return state_distribution


def main():
    cfg = sys.argv[1]
    with open(f"{const.PATHS['project_path']}/src/configs/{cfg}.yaml", "r") as stream:
        try:
            struct = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    state_distribution = calculate_state_distribution(
        struct["agent_info"], struct["env_info"], struct["experiment_info"]
    )
    np.save(
        f"{const.PATHS['project_path']}/data/state_distribution_{struct['agent_info']['N']}_states_random_walk",
        state_distribution,
    )


if __name__ == "__main__":
    main()
