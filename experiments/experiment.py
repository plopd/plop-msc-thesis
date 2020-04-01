from pathlib import Path

import numpy as np

import agents.agents as agents
import environments.environments as envs
from experiments.base_experiment import BaseExperiment
from objectives.objectives import get_objective
from representations.representations import get_representation
from rl_glue.rl_glue import RLGlue
from utils.utils import path_exists


class Exp(BaseExperiment):
    def __init__(self, agent_info, env_info, experiment_info):
        super().__init__()
        self.agent_info = agent_info
        self.env_info = env_info
        self.experiment_info = experiment_info
        self.agent = agents.get_agent(agent_info.get("algorithm"))
        self.env = envs.get_environment(env_info.get("env"))
        self.num_episodes = experiment_info.get("n_episodes")
        self.episode_eval_freq = experiment_info.get("episode_eval_freq")
        self.id = experiment_info.get("id")
        self.max_episode_steps = experiment_info.get("max_episode_steps")
        self.output_dir = Path(experiment_info.get("output_dir")).expanduser()
        self.output_dir = path_exists(self.output_dir)
        self.true_values = np.load(self.output_dir / "true_values.npy")
        self.obs = np.load(self.output_dir / "states.npy")
        self.num_obs = len(self.obs)
        self.on_policy_dist = np.ones(self.num_obs) * 1 / self.num_obs
        self.msve_error = np.zeros((self.num_episodes // self.episode_eval_freq + 1,))
        self.log_episodes = self.env_info.get("log_episodes")
        if self.log_episodes:
            self.episodes = [[] for i in range(self.experiment_info.get("runs"))]

        self.objective = get_objective(
            "RMSVE", self.true_values, self.on_policy_dist, np.ones(self.num_obs),
        )

    def init(self):
        feature_representation = get_representation(
            name=self.agent_info.get("representations"), **self.agent_info
        )
        self.state_features = np.array(
            [feature_representation[self.obs[i]] for i in range(self.num_obs)]
        ).reshape(self.num_obs, -1)

        self.rl_glue = RLGlue(self.env, self.agent)
        self.rl_glue.rl_init(self.agent_info, self.env_info)

    def run(self):
        self.init()
        self.learn()
        self.save(self.output_dir / f"{self.id}", self.msve_error)

    def learn(self):
        estimated_state_values = self.message("get approx value")
        self.msve_error[0] = self.objective.value(estimated_state_values)

        for episode in range(1, self.num_episodes + 1):
            self._learn(episode)
            if self.log_episodes:
                self.episodes[0].append(self.rl_glue.rl_env_message("get episode"))

    def _learn(self, episode):
        self.rl_glue.rl_episode(self.max_episode_steps)

        if episode % self.episode_eval_freq == 0:
            estimated_state_values = self.message("get approx value")
            self.msve_error[episode // self.episode_eval_freq] = self.objective.value(
                estimated_state_values
            )

    def save(self, path, data):
        np.save(path, data)

    def message(self, message):
        if message == "get approx value":
            current_theta = self.rl_glue.rl_agent_message("get weight vector")
            if self.agent_info.get("representations") == "TC":
                current_approx_v = np.sum(current_theta[self.state_features], axis=1)
            else:
                current_approx_v = np.dot(self.state_features, current_theta)
            return current_approx_v
        raise Exception("Unexpected message given.")
