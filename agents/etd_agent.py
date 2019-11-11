from agents.td_agent import TD
from utils.utils import get_feature


class ETD(TD):
    def __init__(self):
        super().__init__()
        self.i = None
        self.F = None
        self.M = None

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

    def agent_start(self, observation):
        self.i = 1.0
        self.F = 0.0
        self.M = 0.0
        self.a_t = super().agent_start(observation)

        return self.a_t

    def agent_step(self, reward, observation):

        current_state_feature = get_feature(
            observation - 1, self.feature, **self.agent_info
        )
        last_state_feature = get_feature(self.s_t - 1, self.feature, **self.agent_info)

        self._learn(reward, current_state_feature, last_state_feature)

        self.s_t = observation
        self.a_t = self.agent_policy(observation)

        return self.a_t

    def agent_policy(self, observation):
        return super().agent_policy(observation)

    def agent_end(self, reward):
        last_state_feature = get_feature(self.s_t - 1, self.feature, **self.agent_info)

        self._learn(reward, 0.0, last_state_feature)

        return

    def agent_message(self, message):
        if message == "get followon trace":
            return self.F
        if message == "get emphasis trace":
            return self.M
        response = super().agent_message(message)
        return response

    def agent_cleanup(self):
        pass

    def _learn(self, reward, current_state_feature, last_state_feature):
        self._compute_traces(last_state_feature)
        td_error = self._td_error(reward, current_state_feature, last_state_feature)

        self.theta += self.alpha * td_error * self.z

    def _compute_traces(self, last_state_feature):
        self.F = self.gamma * self.F + self.i
        self.M = self.lmbda * self.i + (1 - self.lmbda) * self.F
        self.z = self.gamma * self.lmbda * self.z + self.M * last_state_feature
