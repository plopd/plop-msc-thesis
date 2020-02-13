import numpy as np

from agents.TD import TD


class LSTD(TD):
    def __init__(self):
        super().__init__()
        self.A = None
        self.b = None
        self.total_steps = None

    def agent_init(self, agent_info):
        super().agent_init(agent_info)
        self.total_steps = 0
        self.A = np.zeros((self.FR.num_features, self.FR.num_features))
        self.b = np.zeros((self.FR.num_features, 1))

    def agent_start(self, observation):
        self.a_t = super().agent_start(observation)
        return self.a_t

    def agent_step(self, reward, observation):
        self.total_steps += 1
        self.a_t = super().agent_step(reward, observation)

        return self.a_t

    def learn(self, reward, current_state_feature, last_state_feature):
        self.eligibility = (
            self.gamma * self.lmbda * self.eligibility + last_state_feature
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
        if message == "get A":
            return self.A
        elif message == "get b":
            return self.b
        elif message == "get weight vector":
            try:
                inv_A = np.linalg.inv(self.A)
                self.weights = np.dot(inv_A, self.b).squeeze()
            except np.linalg.LinAlgError:
                return self.weights
        elif message == "get steps":
            return self.total_steps
        response = super().agent_message(message)

        return response
