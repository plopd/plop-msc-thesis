from features.basis_features import BasisFeatures
from features.dependent_features import DependentFeatures
from features.random_features import RandomFeatures
from features.state_aggregation import StateAggregation
from features.tabular_features import TabularFeatures
from features.tilecoding import TileCoder


def get_feature_representation(name, unit_norm=True, **kwargs):
    if name == "TC":
        num_dims = kwargs.get("in_features")
        tiles_per_dim = [kwargs.get("tiles_per_dim")] * num_dims
        lims = [(kwargs.get("v_min"), (kwargs.get("v_max")))] * num_dims
        tilings = kwargs.get("tilings")
        T = TileCoder(tiles_per_dim, lims, tilings)

        return T
    elif name == "DF":
        num_states = kwargs.get("N")
        num_features = kwargs.get("in_features")
        D = DependentFeatures(num_states, num_features, unit_norm)

        return D
    elif name == "RB" or name == "RNB":
        num_states = kwargs.get("N")
        num_features = kwargs.get("in_features")
        num_ones = kwargs.get("num_ones", 0)
        seed = kwargs.get("seed")
        RF = RandomFeatures(name, num_states, num_features, num_ones, seed)

        return RF
    elif name == "P" or name == "F":
        order = kwargs.get("order")
        num_dims = kwargs.get("in_features")
        v_min = kwargs.get("v_min")
        v_max = kwargs.get("v_max")
        BF = BasisFeatures(name, num_dims, order, v_min, v_max)

        return BF
    elif name == "TF":
        num_states = kwargs.get("N")
        TF = TabularFeatures(num_states)
        return TF
    elif name == "SA":
        num_dims = kwargs.get("in_features")
        seed = kwargs.get("seed")
        SA = StateAggregation(num_dims, seed, unit_norm)

        return SA

    raise Exception("Unexpected features representation given.")
