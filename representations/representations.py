from representations.dependent import DependentRepresentations
from representations.polynomial import BasisRepresentation
from representations.random import RandomRepresentations
from representations.random_cluster import RandomClusterRepresentation
from representations.tabular import TabularRepresentations
from representations.tile_coding import TileCoder


def get_representation(name, unit_norm=True, **kwargs):
    if name == "TC":
        num_dims = kwargs.get("num_dims")
        tiles_per_dim = [kwargs.get("tiles_per_dim")] * num_dims
        lims = [(kwargs.get("min_x"), (kwargs.get("max_x")))] * num_dims
        tilings = kwargs.get("tilings")
        T = TileCoder(tiles_per_dim, lims, tilings)

        return T
    elif name == "D":
        num_states = kwargs.get("num_states")
        DR = DependentRepresentations(num_states, unit_norm)

        return DR
    elif name == "RB" or name == "RNB":
        num_states = kwargs.get("num_states")
        num_features = kwargs.get("num_features")
        num_ones = kwargs.get("num_ones", 0)
        seed = kwargs.get("seed")
        RF = RandomRepresentations(
            name, num_states, num_features, num_ones, seed, unit_norm
        )

        return RF
    elif name == "P" or name == "F":
        order = kwargs.get("order")
        num_dims = kwargs.get("num_dims")
        min_x = kwargs.get("min_x")
        max_x = kwargs.get("max_x")
        a = kwargs.get("a")
        b = kwargs.get("b")
        BR = BasisRepresentation(name, num_dims, order, min_x, max_x, a, b, unit_norm)

        return BR
    elif name == "TA":
        num_states = kwargs.get("num_states")
        TR = TabularRepresentations(num_states)
        return TR
    elif name == "RC":
        num_dims = kwargs.get("num_dims")
        seed = kwargs.get("seed")
        RCR = RandomClusterRepresentation(num_dims, seed, unit_norm)

        return RCR

    raise Exception("Unexpected representations given.")
