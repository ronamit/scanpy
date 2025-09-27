"""
Optimized Mean and Variance Computation
=======================================

This module provides optimized implementations of mean and variance computation
for sparse matrices, designed to replace bottlenecks in preprocessing functions
like highly_variable_genes and normalize_total.

Key optimizations:
- Direct sparse matrix operations without densification
- Numba JIT compilation for performance
- Memory-efficient processing
- Identical results to fast_array_utils.stats.mean_var
"""

from __future__ import annotations

import numba
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix


@numba.njit(cache=True)
def _compute_mean_var_sparse_numba(data, indices, indptr, n_rows, n_cols):
    """Numba-accelerated mean and variance computation for CSR sparse matrix.

    Uses sample variance (n-1 denominator) to match fast_array_utils behavior.
    """
    # Use double precision for intermediate calculations
    sums = np.zeros(n_cols, dtype=np.float64)
    sq_sums = np.zeros(n_cols, dtype=np.float64)

    # Process each row (observation)
    for row in range(n_rows):
        start = indptr[row]
        end = indptr[row + 1]

        # Process non-zero elements in this row
        for idx in range(start, end):
            col = indices[idx]
            val = data[idx]
            sums[col] += val
            sq_sums[col] += val * val

    # Compute means and variances
    means = sums / n_rows

    # Use sample variance (n-1 denominator) to match correction=1
    if n_rows > 1:
        variances = (sq_sums / n_rows - means * means) * n_rows / (n_rows - 1)
    else:
        # For single row, variance is 0 - match master branch behavior
        variances = np.zeros(n_cols, dtype=np.float64)

    # Ensure non-negative variances (numerical stability)
    variances = np.maximum(variances, 0.0)

    return means, variances


@numba.njit(cache=True)
def _compute_mean_var_dense_numba(X):
    """Numba-accelerated mean and variance computation for dense matrix."""
    n_rows, n_cols = X.shape
    means = np.zeros(n_cols, dtype=np.float64)
    variances = np.zeros(n_cols, dtype=np.float64)

    # Compute means
    for col in range(n_cols):
        sum_val = 0.0
        for row in range(n_rows):
            sum_val += X[row, col]
        means[col] = sum_val / n_rows

    # Compute variances (sample variance with n-1 denominator)
    for col in range(n_cols):
        if n_rows > 1:
            mean_val = means[col]
            var_sum = 0.0
            for row in range(n_rows):
                diff = X[row, col] - mean_val
                var_sum += diff * diff
            variances[col] = var_sum / (n_rows - 1)
        else:
            # For single row, variance is 0 - match master branch behavior
            variances[col] = 0.0

    return means, variances


def optimized_mean_var(X, *, axis=0, correction=1):
    """
    Optimized replacement for fast_array_utils.stats.mean_var.

    Provides identical interface and results as the original function,
    but with optimized sparse matrix handling.

    Parameters
    ----------
    X : array-like
        Input data matrix
    axis : int, default 0
        Axis along which to compute statistics (only 0 supported)
    correction : int, default 1
        Degrees of freedom correction (only 1 supported)

    Returns
    -------
    means : ndarray
        Mean values along specified axis
    variances : ndarray
        Variance values along specified axis
    """
    if axis != 0:
        raise NotImplementedError("Only axis=0 is currently supported")

    if correction != 1:
        raise NotImplementedError("Only correction=1 is currently supported")

    if sparse.issparse(X):
        # Use our optimized sparse implementation
        if not isinstance(X, csr_matrix):
            X = X.tocsr()

        means, variances = _compute_mean_var_sparse_numba(
            X.data, X.indices, X.indptr, X.shape[0], X.shape[1]
        )
        return means, variances
    else:
        # Use our optimized dense implementation
        X = np.asarray(X, dtype=np.float64)
        means, variances = _compute_mean_var_dense_numba(X)
        return means, variances


def patch_mean_var():
    """
    Patch fast_array_utils.stats.mean_var with our optimized version.

    This enables automatic optimization of HVG and other preprocessing
    functions without changing their interfaces.
    """
    try:
        from fast_array_utils import stats

        # Store original function for restoration if needed
        if not hasattr(stats, '_original_mean_var'):
            stats._original_mean_var = stats.mean_var

        # Replace with our optimized version
        stats.mean_var = optimized_mean_var

        return True
    except ImportError:
        return False


def restore_mean_var():
    """
    Restore original fast_array_utils.stats.mean_var function.
    """
    try:
        from fast_array_utils import stats

        if hasattr(stats, '_original_mean_var'):
            stats.mean_var = stats._original_mean_var
            delattr(stats, '_original_mean_var')

        return True
    except ImportError:
        return False
