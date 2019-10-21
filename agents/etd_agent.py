import numpy as np

from agents.td_agent import TD
from utils.utils import get_interest


class ETD(TD):
    def agent_init(self, agent_info):

        super(ETD, self).agent_init(agent_info)
        self.i = get_interest(self.N, agent_info["interest"])
        self.F = 0.0
        self.M = 0.0

    def agent_start(self, observation):

        self.a_t = super(ETD, self).agent_start(observation)
        self.F = 0.0
        self.M = 0.0

        return self.a_t

    def agent_step(self, reward, observation):

        current_state_feature = self.phi[observation - 1]
        last_state_feature = self.phi[self.s_t - 1]

        # cf. Eq. 20-17 http://www.jmlr.org/papers/volume17/14-488/14-488.pdf
        self.F = self.gamma * self.F + self.i[self.s_t - 1]
        self.M = self.lmbda * self.i[self.s_t - 1] + (1 - self.lmbda) * self.F
        self.z = self.gamma * self.lmbda * self.z + self.M * last_state_feature

        td_error = (
            reward
            + self.gamma * np.dot(self.theta.T, current_state_feature)
            - np.dot(self.theta.T, last_state_feature)
        )

        self.theta = self.theta + self.alpha * td_error * self.z

        self.s_t = observation
        self.a_t = self.agent_policy(observation)

        return self.a_t

    def agent_policy(self, observation):
        return super(ETD, self).agent_policy(observation)

    def agent_end(self, reward):
        last_state_feature = self.phi[self.s_t - 1]

        self.F = self.gamma * self.F + self.i[self.s_t - 1]
        self.M = self.lmbda * self.i[self.s_t - 1] + (1 - self.lmbda) * self.F

        self.z = self.gamma * self.lmbda * self.z + self.M * last_state_feature

        td_error = reward - np.dot(self.theta.T, last_state_feature)

        self.theta = self.theta + self.alpha * td_error * self.z

        return

    def agent_message(self, message):
        response = super(ETD, self).agent_message(message)
        if message == "get followon trace":
            return self.F
        if message == "get emphasis trace":
            return self.M
        return response

    def agent_cleanup(self):
        pass
