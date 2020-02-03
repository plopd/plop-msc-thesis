from representations.dependent import DependentRepresentations
from representations.polynomial import PolynomialRepresentation
from representations.random import RandomRepresentations
from representations.random_cluster import RandomClusterRepresentation
from representations.tabular import TabularRepresentations
from representations.tile_coding import TileCoder


def get_representation(name, unit_norm=True, **kwargs):
    if name == "TC":
        num_dims = kwargs.get("num_dims")
        tiles_per_dim = [kwargs.get("tiles_per_dim")] * num_dims
        lims = [(kwargs.get("v_min"), (kwargs.get("v_max")))] * num_dims
        tilings = kwargs.get("tilings")
        T = TileCoder(tiles_per_dim, lims, tilings)

        return T
    elif name == "DF":
        num_states = kwargs.get("N")
        num_features = kwargs.get("num_dims")
        D = DependentRepresentations(num_states, num_features, unit_norm)

        return D
    elif name == "RB" or name == "RNB":
        num_states = kwargs.get("N")
        num_features = kwargs.get("num_dims")
        num_ones = kwargs.get("num_ones", 0)
        seed = kwargs.get("seed")
        RF = RandomRepresentations(name, num_states, num_features, num_ones, seed)

        return RF
    elif name == "P" or name == "F":
        order = kwargs.get("order")
        num_dims = kwargs.get("num_dims")
        v_min = kwargs.get("v_min")
        v_max = kwargs.get("v_max")
        BF = PolynomialRepresentation(name, num_dims, order, v_min, v_max)

        return BF
    elif name == "TF":
        num_states = kwargs.get("N")
        TF = TabularRepresentations(num_states)
        return TF
    elif name == "SA":
        num_dims = kwargs.get("num_dims")
        seed = kwargs.get("seed")
        SA = RandomClusterRepresentation(num_dims, seed, unit_norm)

        return SA

    raise Exception("Unexpected representations given.")
