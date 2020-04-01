import json
from pathlib import Path

import numpy as np

import agents.agents as agents
import environments.environments as envs
from experiments.base_experiment import BaseExperiment
from objectives.objectives import get_objective
from representations.representations import get_representation
from rl_glue.rl_glue import RLGlue
from utils.utils import get_simple_logger
from utils.utils import path_exists


class RandomWalkExp(BaseExperiment):
    def __init__(self, agent_info, env_info, experiment_info):
        super().__init__()
        self.agent_info = agent_info
        self.env_info = env_info
        self.experiment_info = experiment_info
        self.agent = agents.get_agent(agent_info.get("algorithm"))
        self.N = env_info["num_states"]
        self.env = envs.get_environment(env_info.get("env"))
        self.n_episodes = experiment_info.get("n_episodes")
        self.episode_eval_freq = experiment_info.get("episode_eval_freq")
        self.id = experiment_info.get("id")
        self.max_episode_steps = experiment_info.get("max_episode_steps")
        self.output_dir = Path(experiment_info.get("output_dir")).expanduser()
        self.initial_seed = experiment_info.get("seed")
        path_exists(self.output_dir)
        path_exists(self.output_dir / "logs")
        self.logger = get_simple_logger(
            __name__, self.output_dir / "logs" / f"{self.id}.txt"
        )
        self.logger.info(
            json.dumps([self.agent_info, self.env_info, self.experiment_info], indent=4)
        )
        path = self.output_dir.parents[
            0
        ] / f"true_v_{self.N}_{self.agent_info.get('discount_rate')}".replace(".", "-")
        self.true_values = np.load(f"{path}.npy")
        self.states = np.arange(self.N).reshape((-1, 1))
        self.state_distribution = np.ones_like(self.true_values) * 1 / len(self.states)
        self.msve_error = np.zeros((self.n_episodes // self.episode_eval_freq + 1,))

        self.error = get_objective(
            "RMSVE",
            self.true_values,
            self.state_distribution,
            np.ones(len(self.true_values)),
        )
        self.timesteps = []

    def init(self):
        FR = get_representation(
            name=self.agent_info.get("representations"), **self.agent_info
        )
        self.representations = np.array(
            [FR[self.states[i]] for i in range(len(self.states))]
        ).reshape(len(self.states), FR.num_features)
        if self.experiment_info.get("save_representations"):
            path = path_exists(self.output_dir / "representations")
            self.save(path / f"repr_{self.id}", self.representations)

        self.rl_glue = RLGlue(self.env, self.agent)
        self.rl_glue.rl_init(self.agent_info, self.env_info)

    def run(self):
        for i in range(self.experiment_info.get("runs")):
            self.agent_info["seed"] = i + self.initial_seed
            self.env_info["seed"] = i + self.initial_seed
            self.init()
            self.learn()
        self.save(self.output_dir / f"{self.id}", self.msve_error)

    def learn(self):
        estimated_state_values = self.message("get state value")
        self.msve_error[0] = self.error.value(estimated_state_values)

        for episode in range(1, self.n_episodes + 1):
            self._learn(episode)

    def _learn(self, episode):
        self.rl_glue.rl_episode(self.max_episode_steps)

        if episode % self.episode_eval_freq == 0:
            estimated_state_values = self.message("get state value")
            self.msve_error[episode // self.episode_eval_freq] = self.error.value(
                estimated_state_values
            )

    def save(self, path, data):
        np.save(path, data)

    def cleanup(self):
        pass

    def message(self, message):
        if message == "get state value":
            current_theta = self.rl_glue.rl_agent_message("get weight vector")
            current_approx_v = np.dot(self.representations, current_theta)
            return current_approx_v
        raise Exception("Unexpected message given.")
