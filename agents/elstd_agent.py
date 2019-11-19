import numpy as np

from agents.etd_agent import ETD


class ELSTD(ETD):
    def __init__(self):
        super().__init__()
        self.A = None
        self.b = None
        self.steps = None

    def agent_init(self, agent_info):
        super().agent_init(agent_info)
        self.steps = 0
        self.A = np.zeros((self.in_features, self.in_features))
        self.b = np.zeros((self.in_features, 1))

        return self.a_t

    def agent_start(self, observation):
        self.a_t = super().agent_start(observation)
        self.steps = 1
        return self.a_t

    def agent_step(self, reward, observation):
        self.steps += 1
        self.a_t = super().agent_step(reward, observation)

        return self.a_t

    def _learn(self, reward, current_state_feature, last_state_feature):
        self.F = self.gamma * self.F + self.i
        self.M = self.lmbda * self.i + (1 - self.lmbda) * self.F
        self.z = self.gamma * self.lmbda * self.z + self.M * last_state_feature

        self.A += (
            1
            / self.steps
            * (
                np.dot(
                    np.expand_dims(self.z, axis=1),
                    np.expand_dims(
                        last_state_feature - self.gamma * current_state_feature, axis=1
                    ).T,
                )
                - self.A
            )
        )

        self.b += 1 / self.steps * (reward * np.expand_dims(self.z, axis=1) - self.b)

    def agent_message(self, message):
        if message == "get A":
            return self.A
        elif message == "get b":
            return self.b
        elif message == "get weight vector":
            try:
                inv_A = np.linalg.inv(self.A)
                self.theta = np.dot(inv_A, self.b).squeeze()
            except np.linalg.LinAlgError:
                return self.theta
        response = super().agent_message(message)

        return response
