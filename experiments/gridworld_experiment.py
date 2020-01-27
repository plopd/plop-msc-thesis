import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import agents.agents as agents
import environments.environments as envs
from experiments.base_experiment import BaseExperiment
from features.features import get_feature_representation
from rl_glue.rl_glue import RLGlue
from utils.objectives import MSVE
from utils.utils import get_simple_logger
from utils.utils import path_exists


class GridworldExp(BaseExperiment):
    def __init__(self, agent_info, env_info, experiment_info):
        super().__init__()
        self.agent_info = agent_info
        self.env_info = env_info
        self.experiment_info = experiment_info

        self.agent = agents.get_agent(agent_info["algorithm"])

        self.N = env_info["N"]
        self.env = envs.get_environment(env_info["env"])

        self.n_episodes = experiment_info["n_episodes"]
        self.episode_eval_freq = experiment_info["episode_eval_freq"]
        self.id = experiment_info["id"]
        self.max_episode_steps = experiment_info["max_episode_steps"]

        self.output_dir = Path(experiment_info["output_dir"]).expanduser()
        path_exists(self.output_dir)

        self.logger = get_simple_logger(
            __name__, self.output_dir / f"{self.id}_info.log"
        )

        self.logger.info(
            json.dumps([self.agent_info, self.env_info, self.experiment_info], indent=4)
        )

        self.states = np.load(
            self.output_dir.parents[0] / "states_puddle.npy", allow_pickle=True
        )

        # Load value function
        path = self.output_dir.parents[0] / f"true_v.npy"
        self.true_v = np.load(path, allow_pickle=True)

        self.states = np.load(
            self.output_dir.parents[0] / f"states_puddle.npy", allow_pickle=True
        )
        self.state_distribution = np.ones_like(self.true_v) * 1 / len(self.states)

        self.msve_error = np.zeros(self.n_episodes // self.episode_eval_freq + 1)
        # self.emphasis = np.zeros(self.n_episodes // self.episode_eval_freq + 1)

        # Load representations of S
        FR = get_feature_representation(name=agent_info.get("features"), **agent_info)

        self.representations = np.array(
            [FR[self.states[i]] for i in range(len(self.states))]
        ).reshape(len(self.states), -1)

    def init_experiment(self):
        self.rl_glue = RLGlue(self.env, self.agent)
        self.rl_glue.rl_init(self.agent_info, self.env_info)

    # from utils.decorators import timer

    # @timer
    def run_experiment(self):
        self.init_experiment()
        self.learn()
        self.save_experiment()

    def learn(self):
        # Log error prior to learning
        current_approx_v = self.message_experiment("get state value")
        self.msve_error[0] = MSVE(
            self.true_v, current_approx_v, self.state_distribution
        )

        # Learn for `self.n_episodes`.
        # Counting episodes starts from 1 because the 0-th episode is treated above.
        for episode in tqdm(range(1, self.n_episodes + 1)):
            self._learn(episode)
            # try:
            #     self.emphasis[
            #         episode // self.episode_eval_freq
            #     ] = self.rl_glue.rl_agent_message("get emphasis trace")
            #     # print(f"Emphasis stats - mean: {self.emphasis.mean()}\tstd:{self.emphasis.std()}\tmin:{self.emphasis.min()}\tmax:{self.emphasis.max()}")
            # except Exception:
            #     pass

    def _learn(self, episode):
        self.rl_glue.rl_episode(self.max_episode_steps)

        if episode % self.episode_eval_freq == 0:
            current_approx_v = self.message_experiment("get state value")
            self.msve_error[episode // self.episode_eval_freq] = MSVE(
                self.true_v, current_approx_v, self.state_distribution
            )

        # print(f"Episode: {episode}; timesteps: {self.rl_glue.rl_env_message('get length episode')}")

        if episode % 1000 == 0:
            precision = int(np.log10(self.n_episodes)) + 1
            self.logger.info(
                f"Episodes: "
                f"{episode:0{precision}d}/{self.n_episodes:0{precision}d},"
                f"\tMSVE: {self.msve_error[episode // self.episode_eval_freq]:.4f}"
            )

    def save_experiment(self):
        np.save(self.output_dir / f"{self.id}_msve", self.msve_error)

    def cleanup_experiment(self):
        pass

    def message_experiment(self, message):
        if message == "get state value":
            current_theta = self.rl_glue.rl_agent_message("get weight vector")
            current_approx_v = np.sum(current_theta[self.representations], axis=1)
            return current_approx_v
        raise Exception("Unexpected message given.")
