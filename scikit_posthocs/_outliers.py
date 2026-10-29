from typing import Union, List
from numbers import Real, Integral
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import t


def outliers_iqr(
    x: Union[List, np.ndarray], ret: str = "filtered", coef: float = 1.5
) -> np.ndarray:
    """Simple detection of potential outliers based on interquartile range
    (IQR). Data that lie within the lower and upper limits are considered
    non-outliers. The lower limit is the number that lies 1.5 IQRs below
    (coefficient may be changed with an argument, see Parameters)
    the first quartile; the upper limit is the number that lies 1.5 IQRs
    above the third quartile.

    Parameters
    ----------
    x : Union[List, np.ndarray]
        An array, any object exposing the array interface, containing
        observations to check for outliers.

    ret : str = 'filtered'
        Specifies object to be returned. Available options are:

        - ``filtered``: return a filtered array (default)
        - ``outliers``: return outliers
        - ``indices``: return indices of non-outliers
        - ``outliers_indices``: return indices of outliers

    coef : float = 1.5
        Coefficient by which IQR is multiplied.

    Returns
    -------
    numpy.ndarray
        One of the following objects:

        - Filtered array (default) if ``ret`` is set to ``filtered``.
        - Array with indices of elements lying within the specified limits
          if ``ret`` is set to ``indices``.
        - Array with outliers if ``ret`` is set to ``outliers``.
        - Array with indices of outlier elements
          if ``ret`` is set to ``outliers_indices``.

    Raises
    ------
    ValueError
        If observations are empty, nonfinite, nonnumeric, complex or not
        one-dimensional; if coef is negative/nonfinite or ret is unknown.
        Missing values are not silently removed or classified as outliers.

    Examples
    --------
    >>> import numpy as np
    >>> import scikit_posthocs as ph
    >>> x = np.array([4, 5, 6, 10, 12, 4, 3, 1, 2, 3, 23, 5, 3])
    >>> ph.outliers_iqr(x, ret = 'outliers')
    array([12, 23])
    """
    if ret not in ("filtered", "outliers", "indices", "outliers_indices"):
        raise ValueError("Unknown IQR return mode.")
    if not isinstance(coef, Real) or isinstance(coef, (bool, np.bool_)) or not np.isfinite(coef) or coef < 0:
        raise ValueError("coef must be a finite nonnegative real number.")
    arr = np.asarray(x)
    if arr.ndim != 1 or arr.size == 0 or not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
        raise ValueError("x must be a nonempty one-dimensional real numeric array.")
    if not np.isfinite(arr).all():
        raise ValueError("x must contain only finite observations; remove missing values explicitly.")

    standardized = _stable_studentized_values(arr)
    q1, q3 = np.percentile(standardized, [25, 75])
    iqr = q3 - q1
    ll = q1 - iqr * coef
    ul = q3 + iqr * coef

    non_outlier_mask = (standardized >= ll) & (standardized <= ul)

    if ret == "indices":
        return np.where(non_outlier_mask)[0]
    elif ret == "outliers":
        return arr[~non_outlier_mask]
    elif ret == "outliers_indices":
        return np.where(~non_outlier_mask)[0]
    else:
        return arr[non_outlier_mask]


def _validate_outlier_sample(x, alpha):
    """Reject undefined tests instead of reporting nonrejection from NaN."""
    if not isinstance(alpha, Real) or isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite real number strictly between 0 and 1.")
    arr = np.asarray(x)
    if arr.ndim != 1 or arr.size < 3 or not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
        raise ValueError("x must contain at least three one-dimensional real numeric observations.")
    if not np.isfinite(arr).all():
        raise ValueError("Observations must be finite; remove missing values explicitly.")
    if np.all(arr == arr[0]):
        raise ValueError("The test statistic is undefined for constant observations.")
    return arr.copy()


def _stable_studentized_values(values):
    """Rescale extreme magnitudes only; studentization is scale invariant.

    Keep ordinary inputs bit-for-bit on the existing arithmetic path. Divide
    before centering/squaring to avoid finite-input overflow and underflow.
    Original observations remain separate for returned values and indices.
    """
    # NumPy retains float16/float32 precision in means, deviations and squares.
    # Promote only the working array: finite inputs can otherwise overflow or
    # underflow well below the float64 rescaling thresholds used below.
    if np.issubdtype(values.dtype, np.floating) and values.dtype.itemsize < 8:
        values = values.astype(np.float64)
    magnitude = np.max(np.abs(values.astype(float)))
    if magnitude > 1e100 or 0 < magnitude < 1e-100:
        return values / magnitude
    return values


def outliers_grubbs(
    x: Union[List, np.ndarray], hypo: bool = False, alpha: float = 0.05
) -> Union[np.ndarray, bool]:
    """Grubbs' Test for Outliers [1]_. This is the two-sided version
    of the test. The null hypothesis implies that there are no outliers
    in the data set.

    Parameters
    ----------
    x : Union[List, np.ndarray]
        An array, any object exposing the array interface, containing
        data to test for an outlier in.

    hypo : bool = False
        Specifies whether to return a bool value of a hypothesis test result.
        Returns ``True`` when we can reject the null hypothesis.
        Otherwise, ``False``. Available options are:

        - ``True``: return a hypothesis test result
        - ``False``: return a filtered array without an outlier (default)

    alpha : float = 0.05
        Significance level for a hypothesis test.

    Returns
    -------
    Union[np.ndarray, bool]
        Returns a filtered array if alternative hypothesis is true, otherwise
        an unfiltered array. Returns null hypothesis test result instead of an
        array if ``hypo`` argument is set to ``True``.

    Notes
    -----
    .. [1] http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm

    Examples
    --------
    >>> import numpy as np
    >>> import scikit_posthocs as ph
    >>> x = np.array([199.31,199.53,200.19,200.82,201.92,201.95,202.18,245.57])
    >>> ph.outliers_grubbs(x).tolist()
    [199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 202.18]
    """
    arr = _validate_outlier_sample(x, alpha)
    standardized = _stable_studentized_values(arr)
    deviations = np.abs(standardized - np.mean(standardized))
    ind = np.argmax(deviations)
    val = deviations[ind]
    G = val / np.std(standardized, ddof=1)
    N = len(arr)
    t_value = t.ppf(1 - alpha / (2 * N), N - 2)
    result = G > (N - 1) / np.sqrt(N) * np.sqrt(t_value**2 / (N - 2 + t_value**2))

    if hypo:
        return result
    else:
        if result:
            return np.delete(arr, ind)
        else:
            return arr


def outliers_tietjen(
    x: Union[List, np.ndarray], k: int, hypo: bool = False, alpha: float = 0.05
) -> Union[np.ndarray, bool]:
    """Tietjen-Moore test [1]_ to detect multiple outliers in a univariate
    data set that follows an approximately normal distribution.
    The Tietjen-Moore test [2]_ is a generalization of the Grubbs' test to
    the case of multiple outliers. If testing for a single outlier,
    the Tietjen-Moore test is equivalent to the Grubbs' test.

    The null hypothesis implies that there are no outliers in the data set.

    Parameters
    ----------
    x : Union[List, np.ndarray]
        An array, any object exposing the array interface, containing
        data to test for an outlier in.

    k : int
        Integer number of potential outliers to test for, from 1 to len(x)-2.
        Function tests for outliers in both tails.

    hypo : bool = False
        Specifies whether to return a bool value of a hypothesis test result.
        Returns ``True`` when we can reject the null hypothesis.
        Otherwise, ``False``. Available options are:

        - ``True``: return a hypothesis test result
        - ``False``: return a filtered array without outliers (default).

    alpha : float = 0.05
        Significance level for a hypothesis test.

    Returns
    -------
    Union[numpy.ndarray, bool]
        Returns a filtered array if alternative hypothesis is true, otherwise
        an unfiltered array. Returns null hypothesis test result instead of an
        array if ``hypo`` argument is set to True.

    Notes
    -----
    .. [1] Tietjen and Moore (August 1972), Some Grubbs-Type Statistics
        for the Detection of Outliers, Technometrics, 14(3), pp. 583-597.
    .. [2] http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h2.htm

    Examples
    --------
    >>> import numpy as np
    >>> import scikit_posthocs as ph
    >>> x = np.array([-1.40, -0.44, -0.30, -0.24, -0.22, -0.13, -0.05, 0.06,
    ...                0.10, 0.18, 0.20, 0.39, 0.48, 0.63, 1.01])
    >>> ph.outliers_tietjen(x, 2)
    array([-0.44, -0.3 , -0.24, -0.22, -0.13, -0.05,  0.06,  0.1 ,  0.18,
    0.2 ,  0.39,  0.48,  0.63])
    """
    arr = _validate_outlier_sample(x, alpha)
    n = arr.size
    if not isinstance(k, Integral) or isinstance(k, (bool, np.bool_)) or not 1 <= k <= n - 2:
        raise ValueError("k must be an integer between 1 and len(x)-2.")

    def tietjen(x_, k_):
        x_mean = x_.mean()
        r = np.abs(x_ - x_mean)
        z = x_[r.argsort()]
        E = np.sum((z[:-k_] - z[:-k_].mean()) ** 2) / np.sum((z - x_mean) ** 2)
        return E

    standardized = _stable_studentized_values(arr)
    e_x = tietjen(standardized, k)
    simulations = 10000
    e_norm = np.empty(simulations)

    retained = n - k
    batch_size = 256
    for start in range(0, simulations, batch_size):
        stop = min(start + batch_size, simulations)
        norm = np.random.normal(size=(stop - start, n))
        means = norm.mean(axis=1, keepdims=True)
        deviations = np.abs(norm - means)
        indices = np.argpartition(deviations, retained - 1, axis=1)[:, :retained]
        kept = np.take_along_axis(norm, indices, axis=1)
        numerator = np.sum((kept - kept.mean(axis=1, keepdims=True)) ** 2, axis=1)
        denominator = np.sum((norm - means) ** 2, axis=1)
        e_norm[start:stop] = numerator / denominator

    CV = np.percentile(e_norm, alpha * 100)
    result = e_x < CV

    if hypo:
        return result
    else:
        if result:
            ind = np.argpartition(np.abs(standardized - standardized.mean()), -k)[-k:]
            return np.delete(arr, ind)
        else:
            return arr


def outliers_gesd(
    x: ArrayLike,
    outliers: int = 5,
    hypo: bool = False,
    report: bool = False,
    alpha: float = 0.05,
) -> np.ndarray:
    """The generalized (Extreme Studentized Deviate) ESD test is used
    to detect one or more outliers in a univariate data set that follows
    an approximately normal distribution [1]_.

    Parameters
    ----------
    x : Union[List, np.ndarray]
        An array, any object exposing the array interface, containing
        data to test for outliers.

    outliers : int = 5
        Number of potential outliers to test for. Test is two-tailed, i.e.
        maximum and minimum values are checked for potential outliers.

    hypo : bool = False
        If True, return a boolean mask identifying outliers in the original
        input order. Otherwise, return the filtered array (default).

    report : bool = False
        Specifies whether to print a summary table of the test.

    alpha : float = 0.05
        Significance level for a hypothesis test.

    Returns
    -------
    np.ndarray
        If hypo is True, returns a boolean array where True indicates an outlier.
        If hypo is False, returns the filtered array with outliers removed.

    Notes
    -----
    .. [1] Rosner, Bernard (May 1983), Percentage Points for a Generalized
        ESD Many-Outlier Procedure,Technometrics, 25(2), pp. 165-172.

    Examples
    --------
    >>> import numpy as np
    >>> import scikit_posthocs as ph
    >>> data = np.array([-0.25, 0.68, 0.94, 1.15, 1.2, 1.26, 1.26, 1.34,
    ...                  1.38, 1.43, 1.49, 1.49, 1.55, 1.56, 1.58, 1.65, 1.69, 1.7, 1.76,
    ...                  1.77, 1.81, 1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.1, 2.14, 2.15,
    ...                  2.23, 2.24, 2.26, 2.35, 2.37, 2.4, 2.47, 2.54, 2.62, 2.64, 2.9,
    ...                  2.92, 2.92, 2.93, 3.21, 3.26, 3.3, 3.59, 3.68, 4.3, 4.64, 5.34,
    ...                  5.42, 6.01])
    >>> ph.outliers_gesd(data, 5)
    array([-0.25,  0.68,  0.94,  1.15,  1.2 ,  1.26,  1.26,  1.34,  1.38,
            1.43,  1.49,  1.49,  1.55,  1.56,  1.58,  1.65,  1.69,  1.7 ,
            1.76,  1.77,  1.81,  1.91,  1.94,  1.96,  1.99,  2.06,  2.09,
            2.1 ,  2.14,  2.15,  2.23,  2.24,  2.26,  2.35,  2.37,  2.4 ,
            2.47,  2.54,  2.62,  2.64,  2.9 ,  2.92,  2.92,  2.93,  3.21,
            3.26,  3.3 ,  3.59,  3.68,  4.3 ,  4.64])
    >>> filtered = ph.outliers_gesd(data, outliers=5, report=True)
    H0: no outliers in the data
    Ha: up to 5 outliers in the data
    Significance level:  α = 0.05
    Reject H0 if Ri > Critical Value (λi)
    <BLANKLINE>
    Summary Table for Two-Tailed Test
    ---------------------------------------
          Exact           Test     Critical
      Number of      Statistic    Value, λi
    Outliers, i      Value, Ri          5 %
    ---------------------------------------
              1          3.119        3.159
              2          2.943        3.151
              3          3.179        3.144 *
              4           2.81        3.136
              5          2.816        3.128
    """
    data_proc = _validate_outlier_sample(x, alpha)
    if not isinstance(outliers, Integral) or isinstance(outliers, (bool, np.bool_)) or not 1 <= outliers <= data_proc.size - 2:
        raise ValueError("outliers must be an integer between 1 and len(x)-2.")
    rs, ls = np.zeros(outliers, dtype=float), np.zeros(outliers, dtype=float)
    argsort_index = np.argsort(data_proc)
    data = data_proc[argsort_index]
    sorted_positions = np.empty(data_proc.size, dtype=np.intp)
    sorted_positions[argsort_index] = np.arange(data_proc.size)
    n = data_proc.size
    active = np.ones(n, dtype=bool)
    # Removals are tracked in two index spaces: positions in the sorted array,
    # used to filter it, and positions in the input array, used for the mask.
    removed = np.empty(outliers, dtype=np.intp)
    removed_input = np.empty(outliers, dtype=np.intp)

    # Lambda values (critical values): do not depend on the outliers.
    nol = np.arange(outliers)  # the number of outliers
    df = n - nol - 2  # degrees of freedom
    t_ppr = t.ppf(1 - alpha / (2 * (n - nol)), df)
    ls = ((n - nol - 1) * t_ppr) / np.sqrt((df + t_ppr**2) * (n - nol))

    for i in np.arange(outliers):
        active_indices = np.flatnonzero(active)
        active_data = _stable_studentized_values(data_proc[active_indices])
        abs_d = np.abs(active_data - np.mean(active_data))

        # R-value calculation
        # A constant remainder has no further candidate; retain prior rejections.
        sd = np.std(active_data, ddof=1)
        R = np.max(abs_d) / sd if sd > 0 else 0.0
        rs[i] = R

        # Remove the observation that maximizes |xi − xmean|
        selected = active_indices[np.argmax(abs_d)]
        removed[i] = sorted_positions[selected]
        removed_input[i] = selected
        active[selected] = False

    if report:
        report_str = [
            "H0: no outliers in the data",
            "Ha: up to " + str(outliers) + " outliers in the data",
            "Significance level:  α = " + str(alpha),
            "Reject H0 if Ri > Critical Value (λi)",
            "",
            "Summary Table for Two-Tailed Test",
            "---------------------------------------",
            "      Exact           Test     Critical",
            "  Number of      Statistic    Value, λi",
            "Outliers, i      Value, Ri      {:5.3g} %".format(100 * alpha),
            "---------------------------------------",
        ]

        for i, (stat, crit_val) in enumerate(zip(rs, ls)):
            report_str.append(
                "{: >11s}".format(str(i + 1))
                + "{: >15s}".format(str(np.round(stat, 3)))
                + "{: >13s}".format(str(np.round(crit_val, 3)))
                + (" *" if stat > crit_val else "")
            )

        print("\n".join(report_str))

    # Remove masked values
    # for which the test statistic is greater
    # than the critical value and return the result
    if hypo:
        mask = np.zeros(n, dtype=bool)
        if any(rs > ls):
            mask[removed_input[: np.max(np.where(rs > ls)) + 1]] = True
        return mask
    else:
        if any(rs > ls):
            return np.delete(data, removed[: np.max(np.where(rs > ls)) + 1])
        return data
