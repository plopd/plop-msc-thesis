from agents.TD import TD


class TDTileCoding(TD):
    def agent_end(self, reward):
        last_state_feature = super().agent_end(reward)
        self.learn(reward, [], last_state_feature)

        return last_state_feature

    def learn(self, reward, current_state_feature, last_state_feature):
        target = (
            reward
            if not current_state_feature
            else reward + self.gamma * self.weights[current_state_feature].sum()
        )
        pred = self.weights[last_state_feature].sum()
        self.eligibility = self.gamma * self.lmbda * self.eligibility
        self.eligibility[last_state_feature] += 1
        self.weights += (
            (self.alpha / self.FR.tilings) * (target - pred) * self.eligibility
        )
