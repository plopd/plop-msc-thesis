import sys
from pathlib import Path

from alphaex.sweeper import Sweeper

from experiments.chain_experiment import ChainExp


def main():
    sweep_id = int(sys.argv[1])
    sweep_file_name = "Chain5RandomNonBinary.json"
    sweeper = Sweeper(f"{Path(__file__).parents[1]}/configs/{sweep_file_name}")
    param_cfg = sweeper.parse(sweep_id)

    agent_info = {
        "N": param_cfg.get("N", None),
        "algorithm": param_cfg.get("algorithm", None),
        "features": param_cfg.get("features", None),
        "order": param_cfg.get("order", None),
        "n": param_cfg.get("n", None),
        "num_ones": param_cfg.get("num_ones", None),
        "gamma": param_cfg.get("gamma", None),
        "lmbda": param_cfg.get("lmbda", None),
        "alpha": param_cfg.get("alpha", None),
        "seed": param_cfg.get("seed", None),
        "interest": param_cfg.get("interest", None),
    }

    env_info = {"env": param_cfg.get("env", None), "N": param_cfg.get("N", None)}

    exp_info = {
        "id": sweep_id,
        "max_episode_steps": param_cfg.get("max_episode_steps", None),
        "episode_eval_freq": param_cfg.get("episode_eval_freq", None),
        "n_episodes": param_cfg.get("n_episodes", None),
        "output_dir": param_cfg.get("output_dir", None),
    }

    exp = ChainExp(agent_info=agent_info, env_info=env_info, experiment_info=exp_info)

    exp.run_experiment()


if __name__ == "__main__":
    main()
