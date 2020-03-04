from representations.dependent import DependentRepresentations
from representations.fourier import FourierRepresentation
from representations.inverted import InvertedRepresentations
from representations.polynomial import PolynomialRepresentation
from representations.random import RandomRepresentations
from representations.random_binary import RandomBinaryRepresentations
from representations.random_cluster import RandomClusterRepresentation
from representations.tabular import TabularRepresentations
from representations.tile_coding import TileCoder


def get_representation(name, unit_norm=True, **kwargs):
    if name == "TC":
        num_dims = kwargs.get("num_dims")
        tiles_per_dim = [kwargs.get("tiles_per_dim")] * num_dims
        lims = [(kwargs.get("min_x"), (kwargs.get("max_x")))] * num_dims
        tilings = kwargs.get("tilings")
        return TileCoder(tiles_per_dim, lims, tilings)
    elif name == "D":
        num_states = kwargs.get("num_states")
        return DependentRepresentations(num_states, unit_norm)
    elif name == "IN":
        num_states = kwargs.get("num_states")
        return InvertedRepresentations(num_states, unit_norm)
    elif name == "RB":
        num_states = kwargs.get("num_states")
        num_features = kwargs.get("num_features")
        num_ones = kwargs.get("num_ones")
        seed = kwargs.get("seed")
        return RandomBinaryRepresentations(num_states, num_features, num_ones, seed)
    elif name == "R":
        num_states = kwargs.get("num_states")
        num_features = kwargs.get("num_features")
        seed = kwargs.get("seed")
        return RandomRepresentations(num_states, num_features, seed, unit_norm)
    elif name == "P":
        order = kwargs.get("order")
        num_dims = kwargs.get("num_dims")
        min_x = kwargs.get("min_x")
        max_x = kwargs.get("max_x")
        a = kwargs.get("a")
        b = kwargs.get("b")
        return PolynomialRepresentation(num_dims, order, min_x, max_x, a, b, unit_norm)
    elif name == "F":
        order = kwargs.get("order")
        num_dims = kwargs.get("num_dims")
        min_x = kwargs.get("min_x")
        max_x = kwargs.get("max_x")
        a = kwargs.get("a")
        b = kwargs.get("b")
        return FourierRepresentation(num_dims, order, min_x, max_x, a, b, unit_norm)
    elif name == "TA":
        num_states = kwargs.get("num_states")
        return TabularRepresentations(num_states)
    elif name == "RC":
        num_dims = kwargs.get("num_dims")
        seed = kwargs.get("seed")
        return RandomClusterRepresentation(num_dims, seed, unit_norm)

    raise Exception("Unexpected representations given.")
