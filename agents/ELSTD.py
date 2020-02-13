import numpy as np

from agents.ETD import ETD
from agents.LSTD import LSTD


class ELSTD(LSTD, ETD):
    def __init__(self):
        super().__init__()

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

    def agent_start(self, observation):
        self.a_t = super().agent_start(observation)

        return self.a_t

    def learn(self, reward, current_state_feature, last_state_feature):
        self.F = self.gamma * self.F + self.i
        self.M = self.lmbda * self.i + (1 - self.lmbda) * self.F
        self.eligibility = (
            self.gamma * self.lmbda * self.eligibility + self.M * last_state_feature
        )

        self.A += (
            1
            / self.total_steps
            * (
                np.dot(
                    np.expand_dims(self.eligibility, axis=1),
                    np.expand_dims(
                        last_state_feature - self.gamma * current_state_feature, axis=1
                    ).T,
                )
                - self.A
            )
        )

        self.b += (
            1
            / self.total_steps
            * (reward * np.expand_dims(self.eligibility, axis=1) - self.b)
        )

    def agent_message(self, message):
        response = super().agent_message(message)

        return response
