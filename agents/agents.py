from .elstd_agent import ELSTD
from .etd_agent import ETD
from .lstd_agent import LSTD
from .mc_agent import MC
from .td_agent import TD


def get_agent(name):
    if name == "TD":
        return TD
    elif name == "ETD":
        return ETD
    elif name == "MC":
        return MC
    elif name == "LSTD":
        return LSTD
    elif name == "ELSTD":
        return ELSTD

    raise Exception("Unexpected agent given.")
