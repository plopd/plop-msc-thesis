import sys
from pathlib import Path

from alphaex.sweeper import Sweeper

from experiments.chain_experiment import ChainExp


def main():
    sweep_id = int(sys.argv[1])  # sys.argv returns str
    sweep_file_name = "chain.json"
    sweeper = Sweeper(f"{Path(__file__).parents[1]}/configs/{sweep_file_name}")
    param_cfg = sweeper.parse(sweep_id)

    report = (
        "idx: %d\nrun: %d\nenv: %s\nN: %d\nalgorithm: %s\nalpha: "
        "%s\nfeatures: %s\n"
        % (
            sweep_id,
            param_cfg.get("run", None),
            param_cfg.get("env", None),
            param_cfg.get("N", None),
            param_cfg.get("algorithm", None),
            param_cfg.get("alpha", None),
            param_cfg.get("features", None),
        )
    )
    print(report)

    agent_info = {
        "N": param_cfg["N"],
        "algorithm": param_cfg["algorithm"],
        "features": param_cfg["features"],
        "gamma": param_cfg["gamma"],
        "lmbda": param_cfg["lmbda"],
        "alpha": param_cfg["alpha"],
        "seed": param_cfg["run"],
        "interest": param_cfg["interest"],
    }

    env_info = {"env": param_cfg["env"], "N": param_cfg["N"]}

    exp_info = {
        "id": sweep_id,
        "max_episode_steps": param_cfg["max_episode_steps"],
        "episode_eval_freq": param_cfg["episode_eval_freq"],
        "n_episodes": param_cfg["n_episodes"],
        "output_dir": param_cfg["output_dir"],
    }

    exp = ChainExp(agent_info=agent_info, env_info=env_info, experiment_info=exp_info)

    exp.run_experiment()


if __name__ == "__main__":
    main()
