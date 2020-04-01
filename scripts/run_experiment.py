import sys
from pathlib import Path

from alphaex.sweeper import Sweeper

from experiments.experiments import get_experiment


def main():
    run(sweep_id=int(sys.argv[1]), config_fn=sys.argv[2])


def run(sweep_id, config_fn):

    sweeper = Sweeper(Path(__file__).parents[1] / "configs" / f"{config_fn}.json")

    param_cfg = sweeper.parse(sweep_id)

    agent_info = {
        "num_states": param_cfg.get("num_states"),
        "algorithm": param_cfg.get("algorithm"),
        "representations": param_cfg.get("representations"),
        "num_dims": param_cfg.get("num_dims"),
        "num_features": param_cfg.get("num_features"),
        "order": param_cfg.get("order"),
        "num_ones": param_cfg.get("num_ones"),
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
        "log_episodes": param_cfg.get("log_episodes", 0),
    }

    output_dir = Path(param_cfg.get("output_dir")) / config_fn
    exp_info = {
        "seed": param_cfg.get("run"),
        "problem": param_cfg.get("problem"),
        "id": sweep_id,
        "max_episode_steps": param_cfg.get("max_episode_steps", 0),
        "episode_eval_freq": param_cfg.get("episode_eval_freq"),
        "n_episodes": param_cfg.get("n_episodes"),
        "output_dir": str(output_dir),
        "save_representations": param_cfg.get("save_representations"),
        "runs": param_cfg.get("runs", 1),
    }

    experiment = get_experiment(exp_info.get("problem"), agent_info, env_info, exp_info)
    experiment.run()
    return experiment


if __name__ == "__main__":
    main()
