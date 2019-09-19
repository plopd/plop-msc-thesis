import sys

import numpy as np
import utils.const as const
import yaml
from rl_glue.rl_glue import RLGlue
from tqdm import tqdm


def calculate_true_v(agent_info, env_info, experiment_info):

    rl_glue = RLGlue(const.ENVS[env_info["env"]], const.AGENTS[agent_info["agent"]])

    rl_glue.rl_init(agent_info, env_info)
    for _ in tqdm(range(1, experiment_info["n_episodes"] + 1)):
        rl_glue.rl_episode(experiment_info["max_timesteps_episode"])
    return rl_glue.rl_agent_message("get state value")


def main():
    cfg = sys.argv[1]
    with open(f"{const.PATHS['project_path']}/src/configs/{cfg}.yaml", "r") as stream:
        try:
            struct = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    true_v = calculate_true_v(
        struct["agent_info"], struct["env_info"], struct["experiment_info"]
    )
    print(true_v)
    np.save(
        f"{const.PATHS['project_path']}/data/true_v_{struct['agent_info']['N']}_states_random_walk",
        true_v,
    )


if __name__ == "__main__":
    main()
