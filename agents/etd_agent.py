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
        self.reset()

        return self.a_t

    def agent_message(self, message):
        if message == "get followon trace":
            return self.F
        if message == "get emphasis trace":
            return self.M
        response = super().agent_message(message)
        return response

    def update_traces(self, last_state_feature):
        self.F = self.gamma * self.F + self.i
        self.M = self.lmbda * self.i + (1 - self.lmbda) * self.F
        self.z = self.gamma * self.lmbda * self.z + self.M * last_state_feature

    def _learn(self, reward, current_state_feature, last_state_feature):
        self.update_traces(last_state_feature)

        td_error = (
            reward
            + self.gamma * np.dot(self.theta.T, current_state_feature)
            - np.dot(self.theta.T, last_state_feature)
        )

        self.theta += self.alpha * td_error * self.z

    def reset(self):
        super().reset()
        self.i = 1.0
        self.F = 0.0
        self.M = 0.0
