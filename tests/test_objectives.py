import numpy as np
import pytest

from utils.objectives import MSVE


@pytest.mark.parametrize("N, expected_res", [(5, 11.0), (19, 130.0)])
def test_MSVE_with_uniform_interest_and_state_distribution(N, expected_res):
    res = MSVE(np.arange(1, N + 1), np.zeros(N), np.ones(N) / N)
    assert np.isclose(res, np.sqrt(expected_res))
