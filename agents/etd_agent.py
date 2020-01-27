import numpy as np

from agents.td_agent import TD


class ETD(TD):
    def __init__(self):
        super().__init__()
        self.i = None
        self.F = None
        self.M = None

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

    def agent_start(self, observation):
        self.a_t = super().agent_start(observation)
        self.agent_cleanup()

        return self.a_t

    def agent_message(self, message):
        if message == "get followon trace":
            return self.F
        if message == "get emphasis trace":
            return self.M
        response = super().agent_message(message)
        return response

    def learn(self, reward, current_state_feature, last_state_feature):
        target = reward + self.gamma * np.dot(self.weights.T, current_state_feature)
        pred = np.dot(self.weights.T, last_state_feature)
        delta = target - pred

        self.F = self.gamma * self.F + self.i
        self.M = self.lmbda * self.i + (1 - self.lmbda) * self.F
        self.eligibility = (
            self.gamma * self.lmbda * self.eligibility + self.M * last_state_feature
        )

        self.weights += self.alpha * delta * self.eligibility

    def learnTC(self, reward, current_state_feature, last_state_feature):
        target = (
            reward
            if current_state_feature is None
            else reward + self.gamma * self.weights[current_state_feature].sum()
        )
        pred = self.weights[last_state_feature].sum()
        delta = target - pred

        self.F = self.gamma * self.F + self.i
        self.M = self.lmbda * self.i + (1 - self.lmbda) * self.F
        self.eligibility = self.gamma * self.lmbda * self.eligibility
        self.eligibility[last_state_feature] += self.M

        self.weights += (self.alpha / self.FR.tilings) * delta * self.eligibility

    def agent_cleanup(self):
        super().agent_cleanup()
        self.i = 1.0
        self.F = 0.0
        self.M = 0.0
