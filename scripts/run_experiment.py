import sys
from pathlib import Path

from alphaex.sweeper import Sweeper

from experiments.experiments import get_experiment


def main():
    sweep_id = int(sys.argv[1].strip(","))
    sweep_file_name = sys.argv[2]

    sweeper = Sweeper(Path(__file__).parents[1] / "configs" / f"{sweep_file_name}")

    param_cfg = sweeper.parse(sweep_id)

    agent_info = {
        "num_states": param_cfg.get("num_states"),
        "algorithm": param_cfg.get("algorithm"),
        "representations": param_cfg.get("representations"),
        "num_dims": param_cfg.get("num_dims"),
        "num_features": param_cfg.get("num_features"),
        "order": param_cfg.get("order"),
        "num_ones": param_cfg.get("num_ones", 0),
        "min_x": param_cfg.get("min_x"),
        "max_x": param_cfg.get("max_x"),
        "a": param_cfg.get("a"),
        "b": param_cfg.get("b"),
        "discount_rate": param_cfg.get("discount_rate"),
        "trace_decay": param_cfg.get("trace_decay"),
        "step_size": param_cfg.get("step_size"),
        "seed": param_cfg.get("run"),
        "interest": param_cfg.get("interest"),
        "policy": param_cfg.get("policy"),
        "tilings": param_cfg.get("tilings"),
        "tiles_per_dim": param_cfg.get("tiles_per_dim"),
    }

    env_info = {
        "env": param_cfg.get("env"),
        "num_states": param_cfg.get("num_states"),
        "seed": param_cfg.get("run"),
    }

    exp_info = {
        "problem": param_cfg.get("problem"),
        "id": sweep_id,
        "max_episode_steps": param_cfg.get("max_episode_steps"),
        "episode_eval_freq": param_cfg.get("episode_eval_freq"),
        "n_episodes": param_cfg.get("n_episodes"),
        "output_dir": param_cfg.get("output_dir"),
        "log_every_nth_episode": param_cfg.get("log_every_nth_episode", 1000),
    }

    experiment = get_experiment(exp_info.get("problem"), agent_info, env_info, exp_info)
    experiment.start()


if __name__ == "__main__":
    main()
