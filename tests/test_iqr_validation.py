import numpy as np
import pytest
from scikit_posthocs import outliers_iqr


@pytest.mark.parametrize('data', [[], [[1,2],[3,4]], [1,np.nan,3], [1,np.inf],
                                  [1,2+3j], ['one','two']])
@pytest.mark.parametrize('ret', ['filtered','outliers','indices','outliers_indices'])
def test_invalid_observations_never_silently_partition(data,ret):
    with pytest.raises(ValueError): outliers_iqr(data, ret=ret)


@pytest.mark.parametrize('coef', [-1, np.nan, np.inf, True, '1.5'])
def test_invalid_fence_coefficient(coef):
    with pytest.raises(ValueError): outliers_iqr([1,2,3,4],coef=coef)


def test_invalid_return_mode():
    with pytest.raises(ValueError): outliers_iqr([1,2,3],ret='outlier')


def test_zero_coefficient_and_constant_or_single_observation():
    np.testing.assert_array_equal(outliers_iqr([1,2,3,4],coef=0),[2,3])
    for data in [[4], [4,4,4]]:
        np.testing.assert_array_equal(outliers_iqr(data),data)
        assert len(outliers_iqr(data,ret='outliers')) == 0


def test_boundary_values_remain_and_input_is_unchanged():
    data=np.array([-3.,0.,1.,2.,5.])
    original=data.copy()
    # Q1=0, Q3=2, fences[-3,5], equality is not an outlier.
    np.testing.assert_array_equal(outliers_iqr(data),original)
    np.testing.assert_array_equal(data,original)
