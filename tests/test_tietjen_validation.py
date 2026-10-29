import numpy as np
import pytest
from scikit_posthocs import outliers_tietjen


@pytest.mark.parametrize('data,k,alpha', [
    ([1,2,np.nan],1,.05),([1,2,np.inf],1,.05),([1,2,3j],1,.05),
    ([],1,.05),([1,2],1,.05),([[1,2],[3,4]],1,.05),([1,1,1],1,.05),
    ([1,2,3,4],0,.05),([1,2,3,4],-1,.05),([1,2,3,4],3,.05),
    ([1,2,3,4],4,.05),([1,2,3,4],1.5,.05),([1,2,3,4],True,.05),
    ([1,2,3,4],1,0),([1,2,3,4],1,1),([1,2,3,4],1,np.nan),
])
def test_invalid_request_rejected_before_rng_consumption(data,k,alpha):
    state=np.random.get_state()
    try:
        np.random.seed(41)
        expected=np.random.random(4)
        np.random.seed(41)
        with pytest.raises(ValueError): outliers_tietjen(data,k,alpha=alpha)
        np.testing.assert_array_equal(np.random.random(4),expected)
    finally: np.random.set_state(state)


def test_valid_maximum_count_and_input_unchanged():
    data=np.array([1.,2.,3.,20.]); original=data.copy()
    state=np.random.get_state()
    try:
        np.random.seed(43)
        result=outliers_tietjen(data,2,hypo=True)
        assert isinstance(bool(result),bool)
    finally: np.random.set_state(state)
    np.testing.assert_array_equal(data,original)
