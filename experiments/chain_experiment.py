from pathlib import Path

import numpy as np
from tqdm import tqdm

import agents.agents as agents
import environments.environments as envs
from experiments.base_experiment import BaseExperiment
from rl_glue.rl_glue import RLGlue
from utils.calculate_state_distribution_chain import calculate_state_distribution
from utils.calculate_value_function_chain import calculate_v_chain
from utils.utils import get_interest
from utils.utils import MSVE
from utils.utils import path_exists


class ChainExp(BaseExperiment):
    def __init__(self, agent_info, env_info, experiment_info):
        super().__init__()
        self.agent_info = agent_info
        self.env_info = env_info
        self.experiment_info = experiment_info

        self.agent = agents.get_agent(agent_info["algorithm"])
        self.alpha = agent_info["alpha"]

        self.N = env_info["N"]
        self.env = envs.get_environment(env_info["env"])

        self.n_episodes = experiment_info["n_episodes"]
        self.episode_eval_freq = experiment_info["episode_eval_freq"]
        self.id = experiment_info["id"]
        self.max_episode_steps = experiment_info["max_episode_steps"]

        self.i = get_interest(agent_info["interest"], **agent_info)

        self.output_dir = Path(experiment_info["output_dir"]).expanduser()
        path_exists(self.output_dir)

        self.rl_glue = None
        self.msve_error = np.zeros(self.n_episodes // self.episode_eval_freq + 1)

        path = self.output_dir.parents[0] / f"true_v_{self.N}.npy"
        if not path.is_file():
            true_v = calculate_v_chain(agent_info["N"])
            np.save(path, true_v)
        self.true_v = np.load(path, allow_pickle=True)

        path = self.output_dir.parents[0] / f"state_distribution_{self.N}.npy"
        if not path.is_file():
            state_dist = calculate_state_distribution(agent_info["N"])
            np.save(path, state_dist)
        self.state_distribution = np.load(path, allow_pickle=True)

    def init_experiment(self):
        self.rl_glue = RLGlue(self.env, self.agent)
        self.rl_glue.rl_init(
            agent_init_info=self.agent_info, env_init_info=self.env_info
        )

    def run_experiment(self):
        self.init_experiment()
        self.learn()
        self.save_experiment()

    def learn(self):
        current_approx_v = self.rl_glue.rl_agent_message("get state value")
        self.msve_error[0] = MSVE(
            self.true_v, current_approx_v, self.state_distribution, self.i
        )

        for episode in tqdm(range(1, self.n_episodes + 1)):
            self._learn(episode)

    def _learn(self, episode):
        self.rl_glue.rl_episode(0)

        if episode % self.episode_eval_freq == 0:
            current_approx_v = self.rl_glue.rl_agent_message("get state value")
            self.msve_error[episode // self.episode_eval_freq] = MSVE(
                self.true_v, current_approx_v, self.state_distribution, self.i
            )
        elif episode == self.n_episodes:
            self.rl_glue.rl_agent_message("get state value")

    def save_experiment(self):
        np.save(self.output_dir / f"{self.id}_msve", self.msve_error)

    def cleanup_experiment(self):
        pass

    def message_experiment(self):
        pass
