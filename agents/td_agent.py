import numpy as np

from agents.base_agent import BaseAgent
from agents.policies import get_action_from_policy
from representations.representations import get_representation


class TD(BaseAgent):
    def agent_init(self, agent_info):
        self.agent_info = agent_info
        self.alpha = agent_info.get("step_size")
        self.gamma = agent_info.get("discount_rate")
        self.lmbda = agent_info.get("trace_decay")
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.policy = agent_info.get("policy")
        self.FR = get_representation(
            name=agent_info.get("representations"), **agent_info
        )
        self.weights = np.zeros(self.FR.num_features)
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

        if self.agent_info.get("representations") == "TC":
            self.learnTC(reward, current_state_feature, last_state_feature)
        else:
            self.learn(reward, current_state_feature, last_state_feature)

        self.s_t = observation
        self.a_t = self.agent_policy(observation)

        return self.a_t

    def agent_end(self, reward):
        last_state_feature = self.FR[self.s_t]

        if self.agent_info.get("representations") == "TC":
            self.learnTC(reward, None, last_state_feature)
        else:
            self.learn(reward, 0.0, last_state_feature)

        return

    def agent_policy(self, observation):
        return get_action_from_policy(
            name=self.policy, rand_generator=self.rand_generator
        )

    def agent_message(self, message):
        if message == "get eligibility trace":
            return self.eligibility
        elif message == "get weight vector":
            return self.weights
        raise Exception("Unexpected agent message given.")

    def agent_cleanup(self):
        self.eligibility = np.zeros_like(self.weights)

    def learn(self, reward, current_state_feature, last_state_feature):
        target = reward + self.gamma * np.dot(self.weights.T, current_state_feature)
        pred = np.dot(self.weights.T, last_state_feature)
        self.eligibility = (
            self.gamma * self.lmbda * self.eligibility + last_state_feature
        )
        self.weights += self.alpha * (target - pred) * self.eligibility

    def learnTC(self, reward, current_state_feature, last_state_feature):
        target = (
            reward
            if current_state_feature is None
            else reward + self.gamma * self.weights[current_state_feature].sum()
        )
        pred = self.weights[last_state_feature].sum()
        self.eligibility = self.gamma * self.lmbda * self.eligibility
        self.eligibility[last_state_feature] += 1
        self.weights += (
            (self.alpha / self.FR.tilings) * (target - pred) * self.eligibility
        )
