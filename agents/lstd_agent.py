import numpy as np

from agents.td_agent import TD


class LSTD(TD):
    def __init__(self):
        super().__init__()
        self.A = None
        self.b = None
        self.total_steps = None

    def agent_init(self, agent_info):
        super().agent_init(agent_info)
        self.total_steps = 0
        self.A = np.zeros((self.in_features, self.in_features))
        self.b = np.zeros((self.in_features, 1))

    def agent_start(self, observation):
        self.a_t = super().agent_start(observation)
        return self.a_t

    def agent_step(self, reward, observation):
        self.total_steps += 1
        self.a_t = super().agent_step(reward, observation)

        return self.a_t

    def _learn(self, reward, current_state_feature, last_state_feature):
        self.z = self.gamma * self.lmbda * self.z + last_state_feature

        self.A += self.get_A(last_state_feature, current_state_feature)

        self.b += self.get_b(reward)

    def get_A(self, last_state_feature, current_state_feature):
        A = (
            1
            / self.total_steps
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
        return A

    def get_b(self, reward):
        b = 1 / self.total_steps * (reward * np.expand_dims(self.z, axis=1) - self.b)
        return b

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
        elif message == "get steps":
            return self.total_steps
        response = super().agent_message(message)

        return response
