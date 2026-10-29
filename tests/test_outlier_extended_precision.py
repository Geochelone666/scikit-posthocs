"""Extended-precision observations must not be narrowed to find their scale."""
import numpy as np
import pytest
from scikit_posthocs import outliers_iqr, outliers_grubbs, outliers_gesd, outliers_tietjen


@pytest.mark.skipif(np.finfo(np.longdouble).max <= np.finfo(np.float64).max,
                    reason="Platform longdouble has no wider exponent range")
@pytest.mark.parametrize("scale_text", ["1e400", "1e-400"])
@pytest.mark.parametrize("method", ["iqr", "grubbs", "gesd", "tietjen"])
@pytest.mark.parametrize("decision", [False, True])
def test_extended_exponent_range_matches_unscaled_sample(scale_text, method, decision):
    original = np.array([1, 2, 3, 4, 5, 6, 7, 30], dtype=np.longdouble)
    scale = np.longdouble(scale_text)
    values = original * scale
    saved = values.copy()

    def run(data):
        if method == "iqr":
            return outliers_iqr(data, ret="outliers_indices" if decision else "filtered")
        if method == "grubbs":
            return outliers_grubbs(data, hypo=decision)
        if method == "gesd":
            return outliers_gesd(data, outliers=1, hypo=decision)
        return outliers_tietjen(data, k=1, hypo=decision)

    np.random.seed(713)
    expected = run(original)
    expected_rng = np.random.random(3)
    np.random.seed(713)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        actual = run(values)
    np.testing.assert_array_equal(np.random.random(3), expected_rng)
    if decision:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual / scale, expected)
        assert actual.dtype == values.dtype
    np.testing.assert_array_equal(values, saved)
