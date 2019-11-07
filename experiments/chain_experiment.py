import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

import agents.agents as agents
import environments.environments as envs
from experiments.experiment import BaseExperiment
from rl_glue.rl_glue import RLGlue
from utils.calculate_state_distribution_chain import calculate_state_distribution
from utils.calculate_value_function_chain import calculate_v_chain
from utils.utils import get_interest
from utils.utils import MSVE
from utils.utils import path_exists


class ChainExp(BaseExperiment):
    def __init__(self, agent_info, env_info, experiment_info):
        super(ChainExp, self).__init__()
        self.agent_info = agent_info
        self.env_info = env_info
        self.experiment_info = experiment_info

        self.agent = agents.get_agent(agent_info["algorithm"])
        self.alpha = agent_info["alpha"]

        self.N = env_info["N"]
        self.env = envs.get_environment(env_info["env"])

        self.n_episodes = experiment_info["n_episodes"]
        self.episode_eval_freq = experiment_info["episode_eval_freq"]
        self.output_dir = f"{experiment_info['output_dir']}"
        self.id = experiment_info["id"]
        self.max_episode_steps = experiment_info["max_episode_steps"]

        self.i = get_interest(
            self.N, agent_info["interest"], seed=agent_info.get("seed")
        )

        path_exists(self.output_dir)

        self.rl_glue = None
        self.msve_error = np.zeros(self.n_episodes // self.episode_eval_freq + 1)

        self.episode = 0

        path = f"{Path(experiment_info['output_dir']).parents[0]}/true_v_{self.N}.npy"

        if not os.path.isfile(path):
            true_v = calculate_v_chain(agent_info["N"])
            np.save(path, true_v)
        self.true_v = np.load(path, allow_pickle=True)

        path = f"{Path(experiment_info['output_dir']).parents[0]}/state_distribution_{self.N}.npy"

        if not os.path.isfile(path):
            state_dist = calculate_state_distribution(agent_info["N"])
            np.save(path, state_dist)
        self.state_distribution = np.load(path, allow_pickle=True)

    def init_experiment(self):
        self.rl_glue = RLGlue(self.env, self.agent)

    def run_experiment(self):
        self.init_experiment()
        self.learn_run()
        self.save_experiment()

    def learn_run(self):
        self.rl_glue.rl_init(
            agent_init_info=self.agent_info, env_init_info=self.env_info
        )

        current_approx_v = self.rl_glue.rl_agent_message("get state value")
        self.msve_error[0] = MSVE(
            true_v=self.true_v,
            est_v=current_approx_v,
            mu=self.state_distribution,
            i=self.i,
        )
        for self.episode in tqdm(range(1, self.n_episodes + 1)):
            self.learn_episode()

    def learn_episode(self):
        self.rl_glue.rl_episode(0)

        if self.episode % self.episode_eval_freq == 0:
            current_approx_v = self.rl_glue.rl_agent_message("get state value")
            self.msve_error[self.episode // self.episode_eval_freq] = MSVE(
                true_v=self.true_v,
                est_v=current_approx_v,
                mu=self.state_distribution,
                i=self.i,
            )
        elif self.episode == self.n_episodes:
            self.rl_glue.rl_agent_message("get state value")

    def save_experiment(self):
        np.save(f"{self.output_dir}/{self.id}_msve", self.msve_error)

    def cleanup_experiment(self):
        pass

    def message_experiment(self):
        pass
