import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_value_function(num_states, action_range, gamma):
    state_prob = 0.5 / float(action_range)
    theta = 0.000_001

    V = np.zeros(num_states + 1)

    delta = np.infty
    i = 0
    while delta > theta:
        i += 1
        delta = 0.0
        for s in range(1, num_states + 1):
            v = V[s]
            value_sum = 0.0
            for transition in range(1, action_range + 1):
                right = s + 1
                right_reward = 0
                if right > num_states:
                    right_reward = 1
                    right = 0

                left = s - 1
                left_reward = 0
                if left < 1:
                    left_reward = -1
                    left = 0

                value_sum += state_prob * (
                    (right_reward + gamma * V[right]) + (left_reward + gamma * V[left])
                )

            V[s] = value_sum
            delta = max(delta, np.abs(v - V[s]))

    return V[1:]


if __name__ == "__main__":
    num_states = int(sys.argv[1])
    action_range = int(sys.argv[2])
    gamma = float(sys.argv[3])
    V = compute_value_function(num_states, action_range, gamma)
    path = Path(
        f"~/scratch/RandomWalk/true_v_{num_states}_{gamma}".replace(".", "-")
    ).expanduser()
    np.save(path, V)
    states = range(0, num_states)
    plt.plot(states, V)
    # plt.xticks(states)
    # plt.yticks(V)
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.show()
