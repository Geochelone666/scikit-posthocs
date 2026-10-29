import warnings
import numpy as np
import pytest
from scikit_posthocs import outliers_iqr


@pytest.mark.parametrize('scale',[1e307,1e-307])
@pytest.mark.parametrize('coef',[0.,1.5])
@pytest.mark.parametrize('ret',['filtered','outliers','indices','outliers_indices'])
def test_finite_extreme_iqr_preserves_units_and_partition(scale,coef,ret):
    values=np.array([-10.,-10.,0.,10.,10.]);scaled=values*scale;original=scaled.copy()
    expected=outliers_iqr(values,coef=coef,ret=ret)
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        actual=outliers_iqr(scaled,coef=coef,ret=ret)
    if ret in ('filtered','outliers'):
        np.testing.assert_allclose(actual/scale,expected)
    else:
        np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(scaled,original)
