import numpy as np


def get_action_from_policy(name, rand_generator=None, **kwargs):
    if name == "random-chain":
        left = 0
        right = 1
        return rand_generator.choice([left, right])
    elif name == "semi-random-puddle":
        north = 3
        east = 1
        return rand_generator.choice([north, east])
    elif name == "MC-fixed-policy":
        observation = kwargs.get("observation")
        position, velocity = observation
        # https://github.com/openai/gym/wiki/MountainCar-v0
        # Add '+1' to convert to action values required by gym
        return int(np.sign(velocity) + 1)

    raise Exception("Unexpected policy given.")
