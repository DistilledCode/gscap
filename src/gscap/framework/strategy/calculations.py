import numpy as np
import pandas as pd
from numba import jit

import gscap


@jit(forceobj=True)
def buffer(
    raw_position_series: pd.Series,
    fraction: float = 0.10,
    return_df=False,
    integrity_check=True,
):
    rps = raw_position_series.to_numpy()  # Convert to NumPy for fast operations

    lower = (np.sign(rps) * (np.abs(rps) - rps * fraction)).round()
    upper = (np.sign(rps) * (np.abs(rps) + rps * fraction)).round()

    buffer = np.zeros_like(rps)  # Initialize buffer array
    buffer[0] = 0.0  # Start at lower bound

    # Vectorized loop instead of `apply()`
    for i in range(1, len(rps)):
        buffer[i] = buffer[i - 1]
        if buffer[i] < lower[i]:
            buffer[i] = lower[i]
        elif buffer[i] > upper[i]:
            buffer[i] = upper[i]

    # Integrity check
    if integrity_check:
        if (buffer < lower).any():
            raise AssertionError(f"buffer values below lower bound")
        if (buffer > upper).any():
            raise AssertionError(f"buffer values above upper bound")

    # Return DataFrame if required
    if return_df:
        return pd.DataFrame(
            {"raw": rps, "lower": lower, "upper": upper, "buffer": buffer},
            index=raw_position_series.index,
        )
    return pd.Series(buffer.astype(int), index=raw_position_series.index)


@jit(nopython=True)
def nan_corr_coef_pair(x, y):
    """Calculate correlation coefficient between two columns, ignoring NaN values"""
    # Find indices where both values are not NaN
    mask = ~(np.isnan(x) | np.isnan(y))

    # Extract valid values
    x_valid = x[mask]
    y_valid = y[mask]

    # If not enough data points, return NaN
    if len(x_valid) < 2:
        return np.nan

    # Calculate means
    mean_x = np.mean(x_valid)
    mean_y = np.mean(y_valid)

    # Calculate numerator and denominator for correlation
    numer = np.sum((x_valid - mean_x) * (y_valid - mean_y))
    denom = np.sqrt(np.sum((x_valid - mean_x) ** 2) * np.sum((y_valid - mean_y) ** 2))

    # Return correlation or NaN if denominator is zero
    return numer / denom if denom != 0 else np.nan


@jit(nopython=True)
def numba_nan_corrcoef(x):
    """Calculate correlation matrix between columns (variables) ignoring NaNs"""
    n_cols = x.shape[1]
    corr_mat = np.ones((n_cols, n_cols))

    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            corr = nan_corr_coef_pair(x[:, i], x[:, j])
            _val = corr if corr > 0.0 else 0.0
            corr_mat[i, j] = _val
            corr_mat[j, i] = _val  # Matrix is symmetric

    return corr_mat


@jit(nopython=True)
def fill_nan(arr, fill_value):
    result = arr.copy()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isnan(result[i, j]):
                result[i, j] = fill_value
    return result


@jit(nopython=True)
def compute_idm_value(w, corr_mat):
    """Calculate a single IDM value"""
    # Matrix multiplication w @ corr_mat @ w.T
    # Implementing manually for numba compatibility
    wt_corr = np.zeros(len(w))
    for i in range(len(w)):
        for j in range(len(w)):
            wt_corr[i] += w[j] * corr_mat[j, i]

    result = 0.0
    for i in range(len(w)):
        result += wt_corr[i] * w[i]

    # Check if result is effectively zero
    if abs(result) < 1e-10:
        return np.nan
    else:
        return 1.0 / np.sqrt(result)


@jit(nopython=True)
def idm_optimized(returns, weights):
    """Optimized IDM calculation"""
    n_returns = len(returns)
    idm = np.zeros(n_returns)
    n_weights = len(weights)

    for i in range(n_returns):
        # Get slice of returns up to i+1
        slice_end = i + 1
        r_slice = returns[:slice_end]

        # Get appropriate weight
        w_idx = min(i, n_weights - 1)
        w = weights[w_idx]

        # Calculate correlation matrix
        corr_mat = numba_nan_corrcoef(r_slice)
        corr_mat = fill_nan(corr_mat, 1.0)

        # Calculate IDM value
        idm[i] = compute_idm_value(w, corr_mat)

    return idm


def calculate_idm(indv_rtr_df, inst_wgt_df):
    if len(indv_rtr_df.columns) == 1:
        return pd.Series(1.0, index=indv_rtr_df.index, name="idm")
    assert all(indv_rtr_df.index == inst_wgt_df.index)
    _index = indv_rtr_df.resample(gscap.IDM_RESAMPLE).last().index
    _rs_rtr = indv_rtr_df.resample(gscap.IDM_RESAMPLE).last().values
    _rs_wgt = inst_wgt_df.resample(gscap.IDM_RESAMPLE).last().values
    _idm = pd.DataFrame(
        {
            "idm": idm_optimized(_rs_rtr, _rs_wgt),
        },
        index=_index,
    )
    _idm = _idm.reindex(index=indv_rtr_df.index.union(_index))
    _idm = _idm.bfill().ffill()
    _idm = _idm.reindex(index=indv_rtr_df.index)
    return _idm.squeeze()
