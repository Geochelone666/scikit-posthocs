import numpy as np
import pandas as pd
import pytest
from scikit_posthocs import compact_letter_display


@pytest.mark.parametrize('matrix', [[], [1, .1], np.ones((2, 3)), np.ones((2, 2, 1)),
                                    [[1, np.nan], [np.nan, 1]], [[1, np.inf], [np.inf, 1]],
                                    [[1, -.1], [-.1, 1]], [[1, 1.1], [1.1, 1]],
                                    [[1, .5], [.001, 1]], [[1, 1j], [1j, 1]]])
def test_invalid_matrix_is_not_a_significance_statement(matrix):
    with pytest.raises(ValueError):
        compact_letter_display(matrix)


@pytest.mark.parametrize('alpha', [np.nan, np.inf, -0.1, 0, 1, 1.1, True, '0.05'])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        compact_letter_display(np.eye(2), alpha=alpha)


@pytest.mark.parametrize('maxiter', [-1, 1.5, True, '2'])
def test_invalid_iteration_budget(maxiter):
    with pytest.raises(ValueError):
        compact_letter_display(np.eye(2), maxiter=maxiter)


def test_dataframe_labels_must_agree_and_be_unique():
    for data in [pd.DataFrame(np.eye(2), index=['A','B'], columns=['B','A']),
                 pd.DataFrame(np.eye(2), index=['A','A'], columns=['A','A'])]:
        with pytest.raises(ValueError): compact_letter_display(data)
    with pytest.raises(ValueError): compact_letter_display(np.eye(2), names=['A','A'])
    with pytest.raises(ValueError): compact_letter_display(np.eye(2), names=['A'])


def test_diagonal_ignored_singleton_and_alpha_boundary():
    matrix = np.array([[np.nan, .05], [.05, -1.]])
    original = matrix.copy()
    result = compact_letter_display(matrix)
    assert result.tolist() == ['a', 'a']
    np.testing.assert_equal(matrix, original)
    assert compact_letter_display([[np.inf]], names=['only']).to_dict() == {'only': 'a'}
