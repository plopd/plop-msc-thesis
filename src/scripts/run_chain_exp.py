import os
import sys

from alphaex.sweeper import Sweeper
from experiments.chain_experiment import ChainExp


def main():
    sweep_id = int(sys.argv[1])  # sys.argv returns str
    cfg_dir = "configs/"
    sweep_file_name = "chain.json"
    sweeper = Sweeper(os.path.join(cfg_dir, sweep_file_name))
    param_cfg = sweeper.parse(sweep_id)

    print(param_cfg)

    agent_info = {
        "N": param_cfg["N"],
        "algorithm": param_cfg["algorithm"],
        "features": param_cfg["features"],
        "gamma": param_cfg["gamma"],
        "lmbda": param_cfg["lmbda"],
        "alpha": param_cfg["alpha"],
        "seed": sweep_id,
        "interest": param_cfg["interest"],
    }

    env_info = {"env": param_cfg["env"], "N": param_cfg["N"]}

    exp_info = {
        "id": sweep_id,
        "max_timesteps_episode": param_cfg["max_timesteps_episode"],
        "episode_eval_freq": param_cfg["episode_eval_freq"],
        "n_episodes": param_cfg["n_episodes"],
        "output_dir": param_cfg["output_dir"],
    }

    exp = ChainExp(agent_info=agent_info, env_info=env_info, experiment_info=exp_info)

    exp.run_experiment()


if __name__ == "__main__":
    main()
