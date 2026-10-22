import numpy as np
import pytest

import scikit_posthocs as sp


@pytest.mark.parametrize("alpha", [0.05, 1e-12])
def test_sign_array_threshold_and_zero_pvalue(alpha):
    values = np.array(
        [[1, alpha, 0], [np.nextafter(alpha, 0), 1, np.nextafter(alpha, 1)], [0, alpha, 1]]
    )
    original = values.copy()

    actual = sp.sign_array(values, alpha=alpha)

    np.testing.assert_array_equal(actual, [[-1, 0, 1], [1, -1, 0], [1, 0, -1]])
    np.testing.assert_array_equal(values, original)
