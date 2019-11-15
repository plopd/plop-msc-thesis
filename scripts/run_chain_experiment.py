import json
import sys
from pathlib import Path

from alphaex.sweeper import Sweeper

from experiments.chain_experiment import ChainExp


def main():
    sweep_file_name = "Test.json"
    sweeper = Sweeper(Path(__file__).parents[1] / "configs" / f"{sweep_file_name}")
    sweep_id = int(sys.argv[1])
    param_cfg = sweeper.parse(sweep_id)

    print(json.dumps(param_cfg, indent=4))

    agent_info = {
        "N": param_cfg.get("N"),
        "algorithm": param_cfg.get("algorithm"),
        "features": param_cfg.get("features"),
        "in_features": param_cfg.get("in_features"),
        "order": param_cfg.get("order"),
        "num_ones": param_cfg.get("num_ones"),
        "gamma": param_cfg.get("gamma"),
        "lmbda": param_cfg.get("lmbda"),
        "alpha": param_cfg.get("alpha"),
        "seed": param_cfg.get("seed"),
        "interest": param_cfg.get("interest"),
    }

    env_info = {"env": param_cfg.get("env"), "N": param_cfg.get("N")}

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
