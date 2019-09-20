import argparse
from pathlib import Path

import numpy as np
from experiments.experiment import BaseExperiment
from rl_glue.rl_glue import RLGlue
from tqdm import tqdm
from utils import const
from utils.utils import calculate_irmsve
from utils.utils import path_exists


class RandomWalkExperiment(BaseExperiment):
    def __init__(self, agent_info, env_info, experiment_info):
        super(RandomWalkExperiment, self).__init__()
        self.agent_info = agent_info
        self.env_info = env_info
        self.experiment_info = experiment_info

        self.agent = const.AGENTS[agent_info["agent"]]
        self.max_timesteps_episode = agent_info["max_timesteps_episode"]
        self.alpha = agent_info["alpha"]

        self.N = env_info["N"]
        self.env = const.ENVS[env_info["env"]]

        self.n_runs = experiment_info["n_runs"]
        self.n_episodes = experiment_info["n_episodes"]
        self.episode_eval_freq = experiment_info["episode_eval_freq"]
        self.exp_name = experiment_info["exp_name"]
        self.output_path = Path(const.PATHS["project_path"]) / "output" / self.exp_name
        path_exists(self.output_path)

        self.rl_glue = None
        self.data_path = None
        self.learning_history = np.zeros(
            (self.n_runs, self.n_episodes // self.episode_eval_freq + 1)
        )
        self.approx_v = np.zeros((self.n_runs, self.N))

        self.episode = 0
        self.run = 0

        self.true_v = np.load(
            const.PATHS["project_path"] / "data" / f"true_v_"
            f"{agent_info['N']}_states_random_walk.npy",
            allow_pickle=True,
        )
        self.state_distribution = np.load(
            const.PATHS["project_path"] / "data" / f"state_distribution_"
            f"{agent_info['N']}_states_random_walk.npy",
            allow_pickle=True,
        )

    def init_experiment(self):
        self.rl_glue = RLGlue(self.env, self.agent)

    def run_experiment(self):
        self.init_experiment()
        for self.run in tqdm(range(1, self.n_runs + 1)):
            current_approx_v = self._learn_one_run()
            self.approx_v[self.run - 1] = current_approx_v
            self._save()

        self.make_plots()

    def _learn_one_run(self):
        self.agent_info["seed"] = self.run
        self.rl_glue.rl_init(
            agent_init_info=self.agent_info, env_init_info=self.env_info
        )

        current_approx_v = self.rl_glue.rl_agent_message("get state value")
        self.learning_history[self.run - 1, 0] = calculate_irmsve(
            true_state_val=self.true_v,
            learned_state_val=current_approx_v,
            state_distribution=self.state_distribution,
            num_states=self.N,
        )
        for episode in range(1, self.n_episodes + 1):
            current_approx_v = self._learn_one_episode()

        return current_approx_v

    def _learn_one_episode(self):
        current_approx_v = None
        self.rl_glue.rl_episode(self.max_timesteps_episode)

        if self.episode % self.episode_eval_freq == 0:
            current_approx_v = self.rl_glue.rl_agent_message("get state value")
            self.learning_history[
                self.run - 1, self.episode // self.episode_eval_freq
            ] = calculate_irmsve(
                true_state_val=self.true_v,
                learned_state_val=current_approx_v,
                state_distribution=self.state_distribution,
                num_states=self.N,
            )
        elif self.episode == self.n_episodes:
            current_approx_v = self.rl_glue.rl_agent_message("get state value")

        return current_approx_v

    def _save(self):
        self.data_path = Path(self.output_path) / f"alpha_{self.alpha}".replace(
            ".", "_"
        )
        path_exists(self.data_path)

        np.save(
            f"{self.data_path}/learning_history",
            self.learning_history,
            allow_pickle=True,
        )
        np.save(f"{self.data_path}/approx_v", self.approx_v, allow_pickle=True)

    def _save_stats(self):
        learning_curve, learning_final, learning_speed, early_learning = (
            self._calculate_stats()
        )

        np.save(f"{self.data_path}/learning_curve", learning_curve, allow_pickle=True)
        np.save(f"{self.data_path}/learning_final", learning_final, allow_pickle=True)
        np.save(f"{self.data_path}/learning_speed", learning_speed, allow_pickle=True)
        np.save(f"{self.data_path}/early_learning", early_learning, allow_pickle=True)

    def _calculate_stats(self):

        learning_curve = {
            "avg": np.mean(self.learning_history, axis=0),
            "sd": np.std(self.learning_history, axis=0),
            "se": np.std(self.learning_history, axis=0) / np.sqrt(self.n_runs),
        }

        data_final = np.mean(
            self.learning_history[
                :,
                self.learning_history.shape[1]
                - int(0.01 * self.learning_history.shape[1]) :,
            ],
            axis=1,
        )

        learning_final = {
            "avg": np.mean(data_final, axis=0),
            "sd": np.std(data_final, axis=0),
            "se": np.std(data_final, axis=0) / np.sqrt(self.n_runs),
        }

        auc = np.mean(self.learning_history, axis=1)
        learning_speed = {
            "avg": np.mean(auc, axis=0),
            "sd": np.std(auc, axis=0),
            "se": np.std(auc, axis=0) / np.sqrt(self.n_runs),
        }

        data_start = np.mean(
            self.learning_history[:, : int(0.01 * self.learning_history.shape[1])],
            axis=1,
        )
        early_learning = {
            "avg": np.mean(data_start, axis=0),
            "sd": np.std(data_start, axis=0),
            "se": np.std(data_start, axis=0) / np.sqrt(self.n_runs),
        }

        return learning_curve, learning_final, learning_speed, early_learning

    def make_plots(self):
        pass

    def cleanup_experiment(self):
        pass

    def message_experiment(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, required=True)
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, required=True)
    parser.add_argument("--s0", type=int, required=True)
    parser.add_argument("--s_right_term", type=int, required=True)
    parser.add_argument("--s_left_term", type=int, default=0)
    parser.add_argument("--r_left", type=int, required=True)
    parser.add_argument("--r_right", type=int, default=1)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument(
        "--phi",
        type=str,
        required=True,
        help="tabular; " "random-binary; " "random-non-binary; " "state-aggregation",
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--env", type=str, default="RandomWalkEnvironment")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lmbda", type=float, default=0.0)
    parser.add_argument("--episode_eval_freq", type=int, default=1)
    parser.add_argument(
        "--max_timesteps_episode",
        type=int,
        default=0,
        help="zero is equivalent to no limit.",
    )

    args = parser.parse_args()

    agent_info = {
        "N": args.N,
        "n": args.n,
        "phi": args.phi,
        "agent": args.agent,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "lmbda": args.lmbda,
        "max_timesteps_episode": args.max_timesteps_episode,
    }

    env_info = {
        "env": args.env,
        "N": args.N,
        "s0": args.s0,
        "s_left_term": args.s_left_term,
        "s_right_term": args.s_right_term,
        "r_left": args.r_left,
        "r_right": args.r_right,
    }

    experiment_info = {
        "n_runs": args.n_runs,
        "n_episodes": args.n_episodes,
        "episode_eval_freq": args.episode_eval_freq,
        "exp_name": args.exp_name,
    }

    experiment = RandomWalkExperiment(agent_info, env_info, experiment_info)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
