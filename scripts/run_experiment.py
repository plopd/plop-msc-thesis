import sys
from pathlib import Path

from alphaex.sweeper import Sweeper

from experiments.chain_experiment import ChainExp


def main():
    sweep_id = int(sys.argv[1].strip(","))
    sweep_file_name = sys.argv[2]

    sweeper = Sweeper(Path(__file__).parents[1] / "configs" / f"{sweep_file_name}")

    param_cfg = sweeper.parse(sweep_id)

    agent_info = {
        "N": param_cfg.get("N"),
        "algorithm": param_cfg.get("algorithm"),
        "features": param_cfg.get("features"),
        "in_features": param_cfg.get("in_features"),
        "order": param_cfg.get("order"),
        "num_ones": param_cfg.get("num_ones", 0),
        "v_min": param_cfg.get("v_min"),
        "v_max": param_cfg.get("v_max"),
        "gamma": param_cfg.get("gamma"),
        "lmbda": param_cfg.get("lmbda"),
        "alpha": param_cfg.get("alpha"),
        "seed": param_cfg.get("run"),
        "interest": param_cfg.get("interest"),
        "policy": param_cfg.get("policy"),
    }

    env_info = {
        "env": param_cfg.get("env"),
        "N": param_cfg.get("N"),
        "seed": param_cfg.get("run"),
    }

    exp_info = {
        "id": sweep_id,
        "max_episode_steps": param_cfg.get("max_episode_steps"),
        "episode_eval_freq": param_cfg.get("episode_eval_freq"),
        "n_episodes": param_cfg.get("n_episodes"),
        "output_dir": param_cfg.get("output_dir"),
    }

    exp = ChainExp(agent_info, env_info, exp_info)
    exp.run_experiment()


if __name__ == "__main__":
    main()
