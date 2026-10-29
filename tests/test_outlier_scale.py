import warnings
import numpy as np
import pytest
from scikit_posthocs import outliers_grubbs,outliers_gesd


@pytest.mark.parametrize('scale',[1e200,1e-200,1e306])
@pytest.mark.parametrize('values',[[1.,2.,3.,4.,100.],[-100.,-4.,-3.,-2.,-1.],[-3.,-1.,0.,1.,3.,80.]])
def test_extreme_positive_scaling_preserves_test_and_original_outputs(scale,values):
    values=np.asarray(values);scaled=values*scale;original=scaled.copy()
    expected_grubbs=outliers_grubbs(values,hypo=True)
    expected_mask=outliers_gesd(values,outliers=2,hypo=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        assert outliers_grubbs(scaled,hypo=True)==expected_grubbs
        np.testing.assert_array_equal(outliers_gesd(scaled,outliers=2,hypo=True),expected_mask)
        np.testing.assert_array_equal(outliers_gesd(scaled,outliers=2),np.sort(scaled[~expected_mask]))
        np.testing.assert_allclose(outliers_grubbs(scaled)/scale,outliers_grubbs(values))
    np.testing.assert_array_equal(scaled,original)
