from pathlib import Path

import numpy as np
from tqdm import tqdm

import agents.agents as agents
import environments.environments as envs
from experiments.base_experiment import BaseExperiment
from rl_glue.rl_glue import RLGlue
from utils.calculate_state_distribution_chain import calculate_state_distribution
from utils.calculate_value_function_chain import calculate_v_chain
from utils.objectives import MSVE
from utils.utils import get_chain_states
from utils.utils import get_feature
from utils.utils import path_exists


class ChainExp(BaseExperiment):
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

        # Load value function
        path = self.output_dir.parents[0] / f"true_v_{self.N}.npy"
        if not path.is_file():
            true_v = calculate_v_chain(agent_info["N"])
            np.save(path, true_v)
        self.true_v = np.load(path)

        # Load set S of all nonterminal states
        path = self.output_dir.parents[0] / f"states_{self.N}.npy"
        if not path.is_file():
            states = get_chain_states(agent_info["N"])
            np.save(path, states)
        self.states = np.load(path)

        # Load state distribution
        path = self.output_dir.parents[0] / f"state_distribution_{self.N}.npy"
        if not path.is_file():
            state_dist = calculate_state_distribution(agent_info["N"])
            np.save(path, state_dist)
        self.state_distribution = np.load(path)

        self.msve_error = np.zeros(self.n_episodes // self.episode_eval_freq + 1)

        # Load representations of S
        self.representations = np.array(
            [
                get_feature(self.states[i], **self.agent_info)
                for i in range(self.states.shape[0])
            ]
        ).reshape(self.states.shape[0], -1)

    def init_experiment(self):
        self.rl_glue = RLGlue(self.env, self.agent)
        self.rl_glue.rl_init(self.agent_info, self.env_info)

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
            if (
                self.env_info.get("log_episodes") is not None
                and self.env_info.get("log_episodes") == 1
            ):
                print(
                    "Episode: {},\t{}".format(
                        episode,
                        np.array(self.rl_glue.rl_env_message("get episode"))
                        .squeeze()
                        .tolist(),
                    )
                )

    def _learn(self, episode):
        # Run one episode with `self.max_episode_steps`
        self.rl_glue.rl_episode(self.max_episode_steps)

        if episode % self.episode_eval_freq == 0:
            current_approx_v = self.message_experiment("get state value")
            self.msve_error[episode // self.episode_eval_freq] = MSVE(
                self.true_v, current_approx_v, self.state_distribution
            )

        if self.experiment_info.get("logging"):
            if episode % 1000 == 0:
                precision = int(np.log10(self.n_episodes)) + 1
                print(
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
            current_approx_v = np.dot(self.representations, current_theta)
            return current_approx_v
        raise Exception("Unexpected message given.")
