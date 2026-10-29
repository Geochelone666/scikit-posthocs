import numpy as np
import pytest
from scikit_posthocs import outliers_grubbs, outliers_gesd


@pytest.mark.parametrize('test', [outliers_grubbs, lambda x, **kw: outliers_gesd(x, outliers=1, **kw)])
@pytest.mark.parametrize('data', [[], [1,2], [[1,2],[3,4]], [1,2,np.nan], [1,2,np.inf],
                                  [1,2,3j], ['a','b','c'], [4,4,4]])
@pytest.mark.parametrize('hypo', [False,True])
def test_undefined_statistics_rejected(test,data,hypo):
    with pytest.raises(ValueError): test(data,hypo=hypo)


@pytest.mark.parametrize('test', [outliers_grubbs, lambda x, **kw: outliers_gesd(x,outliers=1,**kw)])
@pytest.mark.parametrize('alpha', [0,1,-.1,np.nan,np.inf,True,'0.05'])
def test_invalid_significance_level(test,alpha):
    with pytest.raises(ValueError): test([1,2,3,4],alpha=alpha)


@pytest.mark.parametrize('count', [0,-1,3,4,1.5,True,'1'])
def test_gesd_requires_positive_count_leaving_three_for_last_test(count):
    with pytest.raises(ValueError): outliers_gesd([1,2,3,4],outliers=count)


def test_smallest_valid_sample_and_count_and_no_mutation():
    data=np.array([1.,2.,9.]);original=data.copy()
    assert isinstance(bool(outliers_grubbs(data,hypo=True)),bool)
    mask=outliers_gesd(data,outliers=1,hypo=True)
    assert mask.shape==(3,) and mask.dtype==bool
    np.testing.assert_array_equal(data,original)


def test_maximum_valid_removals_does_not_create_undefined_final_step():
    result=outliers_gesd([1.,2.,3.,4.,20.],outliers=3,hypo=True)
    assert result.shape==(5,)


def test_gesd_constant_remainder_preserves_detected_candidates_without_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        mask=outliers_gesd([1.,1.,1.,1.,100.],outliers=3,hypo=True)
    np.testing.assert_array_equal(mask,[False,False,False,False,True])
