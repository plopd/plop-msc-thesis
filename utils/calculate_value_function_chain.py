import sys

import numpy as np


def calculate_v_chain(N):
    state_prob = 0.5
    gamma = 1
    theta = 0.000001

    V = np.zeros(N + 1)

    delta = np.infty
    i = 0
    while delta > theta:
        i += 1
        delta = 0.0
        for s in range(1, N + 1):
            v = V[s]
            value_sum = 0.0
            right = s + 1
            right_reward = 0
            if right > N:
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
    N = int(sys.argv[1])
    V = calculate_v_chain(N)
    print(V)
