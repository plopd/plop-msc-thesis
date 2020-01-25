from agents.etd_agent import ETD
from agents.lstd_agent import LSTD


class ELSTD(LSTD, ETD):
    def __init__(self):
        super().__init__()

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

    def agent_start(self, observation):
        self.a_t = super().agent_start(observation)

        return self.a_t

    def learn(self, reward, current_state_feature, last_state_feature):
        super().update_traces(last_state_feature)

        self.A += self.get_A(last_state_feature, current_state_feature)

        self.b += self.get_b(reward)

    def agent_message(self, message):
        response = super().agent_message(message)

        return response
