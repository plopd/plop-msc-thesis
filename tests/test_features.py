import numpy as np
import pytest

from utils.utils import get_dependent_features
from utils.utils import get_inverted_features
from utils.utils import get_random_features


@pytest.mark.parametrize("N", [5, 19])
def test_inverted_features_is_unit_norm(N):
    states = np.arange(N).reshape((-1, 1))
    inverted_features = get_inverted_features(states)
    norm_inv_features = np.linalg.norm(inverted_features, axis=1)
    assert np.allclose(norm_inv_features, np.ones(N))


@pytest.mark.parametrize("N", [5, 19])
def test_dependent_features_is_unit_norm(N):
    states = np.arange(N).reshape((-1, 1))
    dependent_features = get_dependent_features(states)
    norm_dep_features = np.linalg.norm(dependent_features, axis=1)
    assert np.allclose(norm_dep_features, np.ones(N))


@pytest.mark.parametrize("N", [5, 19])
def test_random_features_is_unit_norm(N):
    states = np.arange(N).reshape((-1, 1))
    dependent_features = get_random_features(
        states, N // 2, num_ones=N // 4, kind="random-binary"
    )
    norm_dep_features = np.linalg.norm(dependent_features, axis=1)
    assert np.allclose(norm_dep_features, np.ones(N))
