from .chain_env import ChainEnv
from .mountain_car import MountainCarEnv
from .puddle_world import PuddleWorld
from .random_walk_env import RandomWalkEnv


def get_environment(name):
    if name == "RandomWalk":
        return RandomWalkEnv
    elif name == "Chain":
        return ChainEnv
    elif name == "PuddleWorld":
        return PuddleWorld
    elif name == "MountainCar":
        return MountainCarEnv

    raise Exception("Unexpected environment given.")
