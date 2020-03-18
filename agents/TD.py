import numpy as np

from agents.Base import BaseAgent
from agents.policies import get_action_from_policy
from representations.representations import get_representation
from utils.utils import per_feature_step_size_fourier_KOT


class TD(BaseAgent):
    def __init__(self):
        super().__init__()
        self.agent_info = None
        self.step_size = None
        self.discount_rate = None
        self.trace_decay = None
        self.policy = None
        self.FR = None
        self.weights = None
        self.s_t = None
        self.a_t = None

    def agent_init(self, agent_info):
        self.agent_info = agent_info
        self.step_size = agent_info.get("step_size")
        self.discount_rate = agent_info.get("discount_rate")
        self.trace_decay = agent_info.get("trace_decay")
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.policy = agent_info.get("policy")
        self.FR = get_representation(
            name=agent_info.get("representations"), **agent_info
        )
        self.weights = np.zeros(self.FR.num_features)
        self.eligibility = np.zeros(self.FR.num_features)

        if agent_info.get("representations") == "F" and self.step_size is not None:
            self.step_size = per_feature_step_size_fourier_KOT(
                self.step_size, self.FR.num_features, self.FR.C
            )
        elif agent_info.get("representations") == "RB" and self.step_size is not None:
            self.step_size /= self.FR.num_ones

    def agent_start(self, observation):
        self.agent_cleanup()
        self.s_t = observation
        self.a_t = self.agent_policy(observation)

        return self.a_t

    def agent_step(self, reward, observation):
        current_state_feature = self.FR[observation]
        last_state_feature = self.FR[self.s_t]
        self.s_t = observation
        self.a_t = self.agent_policy(observation)

        self.learn(reward, current_state_feature, last_state_feature)

        return self.a_t

    def agent_end(self, reward):
        last_state_feature = self.FR[self.s_t]
        self.learn(reward, 0.0, last_state_feature)

        return last_state_feature

    def agent_policy(self, observation):
        return get_action_from_policy(
            name=self.policy, obs=observation, rand_generator=self.rand_generator
        )

    def agent_message(self, message):
        if message == "get eligibility trace":
            return self.eligibility
        elif message == "get weight vector":
            return self.weights
        raise Exception("Unexpected agent message given.")

    def agent_cleanup(self):
        self.eligibility = np.zeros(self.FR.num_features)

    def learn(self, reward, current_state_feature, last_state_feature):
        target = reward + self.discount_rate * np.dot(
            self.weights.T, current_state_feature
        )
        pred = np.dot(self.weights.T, last_state_feature)
        self.eligibility = (
            self.discount_rate * self.trace_decay * self.eligibility
            + last_state_feature
        )

        self.weights += self.step_size * (target - pred) * self.eligibility
