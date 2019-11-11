import numpy as np

from agents.base_agent import BaseAgent
from utils.utils import get_feature

LEFT = 0
RIGHT = 1


class TD(BaseAgent):
    def agent_init(self, agent_info):
        self.agent_info = agent_info
        self.in_features = agent_info.get("in_features")
        self.alpha = agent_info.get("alpha")
        self.gamma = agent_info.get("gamma", 1.0)
        self.lmbda = agent_info.get("lmbda", 0.0)
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.theta = np.zeros(self.in_features)
        self.feature = agent_info["features"]

        self.s_t = None
        self.a_t = None

    def agent_start(self, observation):

        self.z = np.zeros_like(self.theta)
        self.s_t = observation
        self.a_t = self.agent_policy(observation)

        return self.a_t

    def agent_step(self, reward, observation):
        current_state_feature = get_feature(
            observation - 1, self.feature, **self.agent_info
        )
        last_state_feature = get_feature(self.s_t - 1, self.feature, **self.agent_info)

        self._learn(reward, current_state_feature, last_state_feature)

        self.s_t = observation
        self.a_t = self.agent_policy(observation)

        return self.a_t

    def agent_end(self, reward):
        last_state_feature = get_feature(self.s_t - 1, self.feature, **self.agent_info)

        self._learn(reward, 0, last_state_feature)

        return

    def agent_policy(self, observation):
        return self.rand_generator.choice([LEFT, RIGHT])

    def agent_message(self, message):
        if message == "get eligibility trace":
            return self.z
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
