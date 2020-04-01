import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

import agents.agents as agents
import environments.environments as envs
from rl_glue.rl_glue import RLGlue
from utils.utils import path_exists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--num_obs", type=int, required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--discount_rate", type=float, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--policy_name", type=str, required=True)
    parser.add_argument("--problem", type=str, required=True)

    args = parser.parse_args()

    save_rootpath = Path(f"{os.environ.get('SCRATCH')}") / f"{args.problem}"
    save_rootpath = path_exists(save_rootpath)
    args.__dict__["save_rootpath"] = save_rootpath

    simulate_on_policy(**args.__dict__)
    # compute_value_function(**args.__dict__)


def simulate_on_policy(**kwargs):
    env_id = kwargs.get("env")
    steps = kwargs.get("steps")
    policy_name = kwargs.get("policy_name")
    save_rootpath = kwargs.get("save_rootpath")
    discount_rate = kwargs.get("discount_rate")
    n_samples = kwargs.get("num_obs")

    agent_info = {
        "algorithm": "TDTileCoding",
        "representations": "TC",
        "max_x": "0.6,0.07",
        "min_x": "-1.2,-0.07",
        "tiles_per_dim": "4,4",
        "tilings": 5,
        "discount_rate": discount_rate,
        "trace_decay": 0.0,
        "step_size": 0.0001,
        "seed": 0,
        "interest": "UI",
        "policy": policy_name,
    }

    env_info = {"env": env_id, "seed": 0}

    agent = agents.get_agent(agent_info.get("algorithm"))
    env = envs.get_environment(env_info.get("env"))

    rl_glue = RLGlue(env, agent)
    rl_glue.rl_init(agent_info, env_info)
    last_state, _ = rl_glue.rl_start()
    states = []
    for _ in tqdm(range(steps)):
        states.append(last_state)
        reward, last_state, last_action, term = rl_glue.rl_step()
        if term:
            last_state, _ = rl_glue.rl_start()
    states = np.vstack(states)

    rand_generator = np.random.RandomState(0)
    idxs = rand_generator.choice(
        np.arange(steps // 2, steps), size=(n_samples,), replace=False
    )
    states = states[idxs, :]
    np.save(save_rootpath / "S", states)


def compute_value_function(**kwargs):
    pass
    # env_id = f"{kwargs.get('env')}-v0"
    # steps = kwargs.get("steps")
    # policy_name = kwargs.get("policy_name")
    # save_rootpath = kwargs.get("save_rootpath")
    # num_episodes = kwargs.get("num_episodes")
    # discount_rate = kwargs.get("discount_rate")
    # states = np.load(save_rootpath / "S.npy")
    #
    # np.save(
    #     save_rootpath / f"true_values-discount_rate_{discount_rate}".replace(".", "_"),
    #     true_values,
    # )


if __name__ == "__main__":
    main()
