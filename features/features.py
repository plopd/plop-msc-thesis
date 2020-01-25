from features.basis_features import BasisFeatures
from features.dependent_features import DependentFeatures
from features.random_features import RandomFeatures
from features.tabular_features import TabularFeatures

# from utils.tilecoding import TileCoder


def get_feature_representation(name, unit_norm=True, **kwargs):
    if name == "TC":
        # n_dim = 1
        # tiles_per_dim = 10
        # lims = [(0, 10)]
        # tilings = 8
        # T = TileCoder(tiles_per_dim, lims, tilings)

        return NotImplementedError
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

    raise Exception("Unexpected features representation given.")
