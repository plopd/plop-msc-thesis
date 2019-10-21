from .chain import Chain


def get_environment(name):
    if name == "chain":
        return Chain

    raise Exception("Unexpected environment given")
