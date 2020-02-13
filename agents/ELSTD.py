import numpy as np

from agents.ETD import ETD
from agents.LSTD import LSTD


class ELSTD(LSTD, ETD):
    def __init__(self):
        super().__init__()

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

    def learn(self, reward, current_state_feature, last_state_feature):
        self.F = self.gamma * self.F + self.i
        self.M = self.lmbda * self.i + (1 - self.lmbda) * self.F
        self.eligibility = (
            self.gamma * self.lmbda * self.eligibility + self.M * last_state_feature
        )

        self.A += (
            1
            / self.timesteps
            * (
                np.outer(
                    self.eligibility,
                    last_state_feature - self.gamma * current_state_feature,
                )
                - self.A
            )
        )

        self.b += 1 / self.timesteps * (reward * self.eligibility - self.b)

    def agent_cleanup(self):
        super().agent_cleanup()

    def agent_message(self, message):
        response = super().agent_message(message)

        return response
