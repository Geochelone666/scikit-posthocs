"""Finite low-precision inputs need wider arithmetic, not wider outputs."""
import numpy as np
import pytest

from scikit_posthocs import (
    outliers_gesd,
    outliers_grubbs,
    outliers_iqr,
    outliers_tietjen,
)


@pytest.mark.parametrize("dtype,scale", [
    (np.float16, 1000), (np.float16, 1e-5),
    (np.float32, 1e20), (np.float32, 1e-30),
])
@pytest.mark.parametrize("test", ["grubbs", "gesd", "tietjen"])
def test_low_precision_matches_same_observations_in_float64(dtype, scale, test):
    values = np.array([1, 2, 3, 4, 5, 6, 7, 30], dtype=dtype) * dtype(scale)
    original = values.copy()
    reference = values.astype(np.float64)

    def run(data, hypo):
        if test == "grubbs":
            return outliers_grubbs(data, hypo=hypo)
        if test == "gesd":
            return outliers_gesd(data, outliers=1, hypo=hypo)
        return outliers_tietjen(data, k=1, hypo=hypo)

    # Seed both simulations equally; promotion must consume no extra randomness.
    np.random.seed(713)
    expected_hypo = run(reference, True)
    expected_next_random = np.random.random(3)
    np.random.seed(713)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual_hypo = run(values, True)
    np.testing.assert_array_equal(actual_hypo, expected_hypo)
    np.testing.assert_array_equal(np.random.random(3), expected_next_random)
    assert np.any(expected_hypo)
    np.random.seed(713)
    expected_filtered = run(reference, False)
    np.random.seed(713)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        filtered = run(values, False)
    np.testing.assert_array_equal(filtered, expected_filtered)
    assert filtered.dtype == values.dtype
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("ret", ["filtered", "outliers", "indices", "outliers_indices"])
def test_float16_iqr_wide_range_preserves_partition_and_dtype(ret):
    values = np.array([-60000, -60000, 0, 60000, 60000], dtype=np.float16)
    expected = outliers_iqr(values.astype(np.float64), coef=0, ret=ret)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = outliers_iqr(values, coef=0, ret=ret)
    np.testing.assert_array_equal(actual, expected)
    if ret in ("filtered", "outliers"):
        assert actual.dtype == values.dtype
