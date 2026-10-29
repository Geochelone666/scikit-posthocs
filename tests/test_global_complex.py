import warnings
import numpy as np
import pytest
from scikit_posthocs import global_f_test, global_simes_test


@pytest.mark.parametrize('function',[global_f_test,global_simes_test])
@pytest.mark.parametrize('values',[np.array([.01+5j,.2+3j]),np.array([.01,.2],dtype=complex),
                                   [.01+5j,.2+3j],np.array([.01+5j,.2+3j],dtype=object)])
def test_nonreal_pvalues_are_rejected_without_lossy_cast_warning(function,values):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError,match='p_vals'):
            function(values)
