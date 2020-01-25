import numpy as np

from agents.base_agent import BaseAgent
from agents.policies import get_action_from_policy
from features.features import get_feature_representation


class TD(BaseAgent):
    def agent_init(self, agent_info):
        self.agent_info = agent_info
        self.in_features = agent_info.get("in_features")
        self.alpha = agent_info.get("alpha")
        self.gamma = agent_info.get("gamma", 1.0)
        self.lmbda = agent_info.get("lmbda", 0.0)
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.theta = np.zeros(self.in_features)
        self.policy = agent_info.get("policy")
        self.FR = get_feature_representation(
            name=agent_info.get("features"), **agent_info
        )
        self.s_t = None
        self.a_t = None

    def agent_start(self, observation):
        self.agent_cleanup()
        self.s_t = observation
        self.a_t = self.agent_policy(observation)
        return self.a_t

    def agent_step(self, reward, observation):
        current_state_feature = self.FR[observation]
        last_state_feature = self.FR[self.s_t]
        self.learn(reward, current_state_feature, last_state_feature)

        self.s_t = observation
        self.a_t = self.agent_policy(observation)

        return self.a_t

    def agent_end(self, reward):
        last_state_feature = self.FR[self.s_t]
        self.learn(reward, 0.0, last_state_feature)

        return

    def agent_policy(self, observation):
        return get_action_from_policy(
            name=self.policy, rand_generator=self.rand_generator
        )

    def agent_message(self, message):
        if message == "get eligibility trace":
            return self.z
        elif message == "get weight vector":
            return self.theta
        raise Exception("Unexpected agent message given.")

    def agent_cleanup(self):
        self.z = np.zeros(self.in_features)

    def learn(self, reward, current_state_feature, last_state_feature):
        target = reward + self.gamma * np.dot(self.theta.T, current_state_feature)
        pred = np.dot(self.theta.T, last_state_feature)
        delta = target - pred
        self.z = self.gamma * self.lmbda * self.z + last_state_feature
        self.theta += self.alpha * delta * self.z
