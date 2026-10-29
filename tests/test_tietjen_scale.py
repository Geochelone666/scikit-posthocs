import warnings
import numpy as np
import pytest
from scikit_posthocs import outliers_tietjen


@pytest.mark.parametrize('scale',[1e200,1e-200,1e306])
@pytest.mark.parametrize('values,k', [([1.,2.,3.,4.,100.],1),([-100.,-4.,-3.,-2.,-1.],1),
                                     ([-3.,-1.,0.,1.,3.,80.,100.],2)])
def test_tietjen_extreme_scaling_preserves_seeded_decisions_outputs_and_rng(scale,values,k):
    values=np.asarray(values);scaled=values*scale;original=scaled.copy()
    state=np.random.get_state()
    try:
        np.random.seed(91);expected=outliers_tietjen(values,k,hypo=True);next_draw=np.random.random(4)
        np.random.seed(91);expected_filtered=outliers_tietjen(values,k)
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            np.random.seed(91);actual=outliers_tietjen(scaled,k,hypo=True)
            assert actual==expected
            np.testing.assert_array_equal(np.random.random(4),next_draw)
            np.random.seed(91);filtered=outliers_tietjen(scaled,k)
            np.testing.assert_allclose(filtered/scale,expected_filtered)
    finally:np.random.set_state(state)
    np.testing.assert_array_equal(scaled,original)
