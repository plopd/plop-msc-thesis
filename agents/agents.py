from .ELSTD import ELSTD
from .ETD import ETD
from .ETDTileCoding import ETDTileCoding
from .LSTD import LSTD
from .MC import MC
from .MCTileCoding import MCTileCoding
from .TD import TD
from .TDTileCoding import TDTileCoding


def get_agent(name):
    if name == "TD":
        return TD
    elif name == "TDTileCoding":
        return TDTileCoding
    elif name == "ETDTileCoding":
        return ETDTileCoding
    elif name == "ETD":
        return ETD
    elif name == "MC":
        return MC
    elif name == "MCTileCoding":
        return MCTileCoding
    elif name == "LSTD":
        return LSTD
    elif name == "ELSTD":
        return ELSTD

    raise Exception("Unexpected agent given.")
