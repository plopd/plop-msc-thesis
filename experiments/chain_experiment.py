import json
from pathlib import Path

import numpy as np

import agents.agents as agents
import environments.environments as envs
from experiments.base_experiment import BaseExperiment
from representations.representations import get_representation
from rl_glue.rl_glue import RLGlue
from utils.objectives import MSVE
from utils.utils import get_simple_logger
from utils.utils import path_exists


class ChainExp(BaseExperiment):
    def __init__(self, agent_info, env_info, experiment_info):
        super().__init__()
        self.agent_info = agent_info
        self.env_info = env_info
        self.experiment_info = experiment_info

        self.agent = agents.get_agent(agent_info.get("algorithm"))

        self.N = env_info["N"]
        self.env = envs.get_environment(env_info.get("env"))

        self.n_episodes = experiment_info.get("n_episodes")
        self.episode_eval_freq = experiment_info.get("episode_eval_freq")
        self.id = experiment_info.get("id")
        self.max_episode_steps = experiment_info.get("max_episode_steps")
        self.output_dir = Path(experiment_info.get("output_dir")).expanduser()
        self.log_every_nth_episode = experiment_info.get("log_every_nth_episode", 1000)
        path_exists(self.output_dir)
        self.logger = get_simple_logger(
            __name__, self.output_dir / f"{self.id}_info.log"
        )
        self.logger.info(
            json.dumps([self.agent_info, self.env_info, self.experiment_info], indent=4)
        )
        path = self.output_dir.parents[0] / f"true_v_{self.N}.npy"
        self.true_v = np.load(path)
        self.states = np.arange(self.N).reshape((-1, 1))
        self.state_distribution = np.ones_like(self.true_v) * 1 / len(self.states)
        self.msve_error = np.zeros(self.n_episodes // self.episode_eval_freq + 1)
        FR = get_representation(name=agent_info.get("representations"), **agent_info)

        self.representations = np.array(
            [FR[self.states[i]] for i in range(self.states.shape[0])]
        ).reshape(self.states.shape[0], -1)

    def init(self):
        self.rl_glue = RLGlue(self.env, self.agent)
        self.rl_glue.rl_init(self.agent_info, self.env_info)

    def start(self):
        self.init()
        self.learn()
        self.save()

    def learn(self):
        current_approx_v = self.message("get state value")
        self.msve_error[0] = MSVE(
            self.true_v, current_approx_v, self.state_distribution
        )

        for episode in range(1, self.n_episodes + 1):
            self._learn(episode)

    def _learn(self, episode):
        self.rl_glue.rl_episode(self.max_episode_steps)

        if episode % self.episode_eval_freq == 0:
            current_approx_v = self.message("get state value")
            self.msve_error[episode // self.episode_eval_freq] = MSVE(
                self.true_v, current_approx_v, self.state_distribution
            )

        if episode % self.experiment_info.get("log_every_nth_episode") == 0:
            self.logger.info(
                f"Episodes: "
                f"{episode}/{self.n_episodes}, "
                f"MSVE: {self.msve_error[episode // self.episode_eval_freq]:.4f}"
            )

    def save(self):
        np.save(self.output_dir / f"{self.id}_msve", self.msve_error)

    def cleanup(self):
        pass

    def message(self, message):
        if message == "get state value":
            current_theta = self.rl_glue.rl_agent_message("get weight vector")
            current_approx_v = np.dot(self.representations, current_theta)
            return current_approx_v
        raise Exception("Unexpected message given.")
