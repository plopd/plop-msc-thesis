import numpy as np


def get_action_from_policy(name, obs, rand_generator=None):
    if name == "random-chain":
        left = 0
        right = 1
        return rand_generator.choice([left, right])
    elif name == "PW-north-east":
        north = 3
        east = 1
        return rand_generator.choice([north, east])
    elif name == "MC-fixed-policy":
        position, velocity = obs
        out = int(np.sign(velocity) + 1)
        # https://github.com/openai/gym/wiki/MountainCar-v0
        # Add '+1' to convert to action values required by gym
        return out

    raise Exception("Unexpected policy given.")
