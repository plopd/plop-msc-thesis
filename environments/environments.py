from .chain import Chain
from .deterministic_chain import DeterministicChain
from .puddle_world import PuddleWorld


def get_environment(name):
    if name == "Chain":
        return Chain
    elif name == "ChainDeterministic":
        return DeterministicChain
    elif name == "PuddleWorld":
        return PuddleWorld

    raise Exception("Unexpected environment given.")
