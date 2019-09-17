import numpy as np
from experiments.experiment import BaseExperiment
from rl_glue.rl_glue import RLGlue
from tqdm import tqdm
from utils import const
from utils.utils import calculate_irmsve


class RandomWalkExperiment(BaseExperiment):
    def __init__(self, agent_info, env_info, experiment_info):
        super(RandomWalkExperiment, self).__init__()
        self.agent_info = agent_info
        self.env_info = env_info
        self.experiment_info = experiment_info

        self.rl_glue = None
        self.learn_history = np.zeros(
            (self.n_runs, self.n_episodes // self.episode_eval_freq + 1)
        )
        self.approx_v = np.zeros((self.n_runs, self.N))

        self.episode = 0
        self.run = 0

        self.agent = const.AGENTS[agent_info["agent"]]
        self.true_v = agent_info["true_v"]
        self.state_distribution = agent_info["state_distribution"]

        self.N = env_info["N"]
        self.env = const.ENVS[env_info["env"]]

        self.n_runs = experiment_info["n_runs"]
        self.n_episodes = experiment_info["n_episodes"]
        self.episode_eval_freq = experiment_info["episode_eval_freq"]
        self.i = experiment_info["i"]
        self.save_path = experiment_info["save_path"]

    def init_experiment(self):
        self.rl_glue = RLGlue(self.env, self.agent)

    def run_experiment(self):
        for self.run in tqdm(range(1, self.n_runs + 1)):
            current_approx_v = self._learn_one_episode()
            self.approx_v[self.run - 1] = current_approx_v

            self._save()

    def _learn_one_run(self):
        self.agent_info["seed"] = self.run
        self.rl_glue.rl_init(
            agent_init_info=self.agent_info, env_init_info=self.env_info
        )

        current_approx_v = self.rl_glue.rl_agent_message("get state value")
        self.learn_history[self.run - 1, 0] = calculate_irmsve(
            true_state_val=self.true_v,
            learned_state_val=current_approx_v,
            state_distribution=self.state_distribution,
            interest=self.i,
            num_states=self.N,
        )
        for episode in range(1, self.n_episodes + 1):
            current_approx_v = self._learn_one_episode()

        return current_approx_v

    def _learn_one_episode(self):
        current_approx_v = None
        self.rl_glue.rl_episode(0)

        if self.episode % self.episode_eval_freq == 0:
            current_approx_v = self.rl_glue.rl_agent_message("get state value")
            self.learn_history[
                self.run - 1, self.episode // self.episode_eval_freq
            ] = calculate_irmsve(
                true_state_val=self.true_v,
                learned_state_val=current_approx_v,
                state_distribution=self.state_distribution,
                interest=self.i,
                num_states=self.N,
            )
        elif self.episode == self.n_episodes:
            current_approx_v = self.rl_glue.rl_agent_message("get state value")

        return current_approx_v

    def _save(self):
        learning_curve, learning_final, learning_speed, early_learning = (
            self._calculate_stats()
        )

        np.save(
            f"{self.save_path}/learn_history", self.learn_history, allow_pickle=True
        )
        np.save(f"{self.save_path}/approx_v", self.approx_v, allow_pickle=True)

        np.save(
            f"{self.save_path}/learning_curve.npy", learning_curve, allow_pickle=True
        )
        np.save(
            f"{self.save_path}/learning_final.npy", learning_final, allow_pickle=True
        )
        np.save(
            f"{self.save_path}/learning_speed.npy", learning_speed, allow_pickle=True
        )
        np.save(
            f"{self.save_path}/early_learning.npy", early_learning, allow_pickle=True
        )

    def _calculate_stats(self):

        learning_curve = {
            "avg": np.mean(self.learn_history, axis=0),
            "sd": np.std(self.learn_history, axis=0),
            "se": np.std(self.learn_history, axis=0) / np.sqrt(self.n_runs),
        }

        data_final = np.mean(
            self.learn_history[
                :,
                self.learn_history.shape[1] - int(0.01 * self.learn_history.shape[1]) :,
            ],
            axis=1,
        )

        learning_final = {
            "avg": np.mean(data_final, axis=0),
            "sd": np.std(data_final, axis=0),
            "se": np.std(data_final, axis=0) / np.sqrt(self.n_runs),
        }

        auc = np.mean(self.learn_history, axis=1)
        learning_speed = {
            "avg": np.mean(auc, axis=0),
            "sd": np.std(auc, axis=0),
            "se": np.std(auc, axis=0) / np.sqrt(self.n_runs),
        }

        data_start = np.mean(
            self.learn_history[:, : int(0.01 * self.learn_history.shape[1])], axis=1
        )
        early_learning = {
            "avg": np.mean(data_start, axis=0),
            "sd": np.std(data_start, axis=0),
            "se": np.std(data_start, axis=0) / np.sqrt(self.n_runs),
        }

        return learning_curve, learning_final, learning_speed, early_learning

    def cleanup_experiment(self):
        pass

    def message_experiment(self):
        pass
