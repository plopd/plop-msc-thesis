import numpy as np

import utils.utils as utils
from agents.base_agent import BaseAgent

LEFT = 0
RIGHT = 1


class TD(BaseAgent):
    def agent_init(self, agent_info):
        self.alpha = agent_info.get("alpha")
        self.gamma = agent_info.get("gamma", 1.0)
        self.lmbda = agent_info.get("lmbda", 0.0)
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        states = np.arange(1, agent_info.get("N") + 1).reshape((-1, 1))
        self.phi = utils.get_features(states, agent_info["features"], **agent_info)
        self.theta = np.zeros(self.phi.shape[1])

        self.s_t = None
        self.a_t = None

    def agent_start(self, observation):

        self.z = np.zeros_like(self.theta)
        self.s_t = observation
        self.a_t = self.agent_policy(observation)

        return self.a_t

    def agent_step(self, reward, observation):
        """

        Args:
            reward:
            observation: (int) in [1, `self.num_states`]

        Returns:

        """
        current_state_feature = self.phi[observation - 1]
        last_state_feature = self.phi[self.s_t - 1]

        self._learn(reward, current_state_feature, last_state_feature)

        self.s_t = observation
        self.a_t = self.agent_policy(observation)

        return self.a_t

    def agent_end(self, reward):
        last_state_feature = self.phi[self.s_t - 1]

        self._learn(reward, 0, last_state_feature)

        return

    def agent_policy(self, observation):
        return self.rand_generator.choice([LEFT, RIGHT])

    def agent_message(self, message):
        if message == "get state value":
            approx_v = np.dot(self.phi, self.theta)
            return approx_v
        elif message == "get eligibility trace":
            return self.z
        elif message == "get feature matrix":
            return self.phi
        elif message == "get weight vector":
            return self.theta
        raise Exception("Unexpected message given.")

    def agent_cleanup(self):
        pass

    def _learn(self, reward, current_state_feature, last_state_feature):
        self._compute_traces(last_state_feature)
        td_error = self._td_error(reward, current_state_feature, last_state_feature)

        self.theta += self.alpha * td_error * self.z

    def _td_error(self, reward, current_state_feature, last_state_feature):
        td_error = (
            reward
            + self.gamma * np.dot(self.theta.T, current_state_feature)
            - np.dot(self.theta.T, last_state_feature)
        )
        return td_error

    def _compute_traces(self, last_state_feature):
        self.z = self.gamma * self.lmbda * self.z + last_state_feature
