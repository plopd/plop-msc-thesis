import numpy as np
import pytest

from objectives.objectives import get_objective


@pytest.mark.parametrize("N, expected_res", [(5, 11.0), (19, 130.0)])
def test_MSVE_with_uniform_interest_and_state_distribution(N, expected_res):
    error = get_objective("RMSVE", np.arange(1, N + 1), np.ones(N) / N, np.ones(N))
    res = error.value(np.zeros(N))
    assert np.isclose(res, np.sqrt(expected_res))
