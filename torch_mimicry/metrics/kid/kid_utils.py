"""
Helper functions for computing FID, as based on:
https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py
"""
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel


def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1):
    """
    Compute MMD between two sets of features.

    Polynomial kernel given by:
    K(X, Y) = (gamma <X, Y> + coef0)^degree

    Args:
        codes_g (ndarray): Set of features from 1st distribution.
        codes_r (ndarray): Set of features from 2nd distribution.
        degree (int): Power of the kernel.
        gamma (float): Scaling factor of dot product.
        coeff0 (float): Constant factor of kernel.

    Returns:
        np.float64: Scalar MMD score between features of 2 distributions.
    """
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _compute_mmd2(K_XX, K_XY, K_YY)


def polynomial_mmd_averages(codes_g,
                            codes_r,
                            n_subsets=50,
                            subset_size=1000,
                            **kernel_args):
    """
    Computes average MMD between two set of features using n_subsets,
    each of which is of subset_size.

    Args:
        codes_g (ndarray): Set of features from 1st distribution.
        codes_r (ndarray): Set of features from 2nd distribution.
        n_subsets (int): Number of subsets to compute averages.
        subset_size (int): Size of each subset of features to choose.

    Returns:
        list: List of n_subsets MMD scores.
    """
    m = min(codes_g.shape[0], codes_r.shape[0])
    mmds = np.zeros(n_subsets)

    # Account for inordinately small subset sizes
    n_subsets = min(m, n_subsets)
    subset_size = min(subset_size, m // n_subsets)

    for i in range(n_subsets):
        g = codes_g[np.random.choice(len(codes_g), subset_size, replace=False)]
        r = codes_r[np.random.choice(len(codes_r), subset_size, replace=False)]
        o = polynomial_mmd(g, r, **kernel_args)
        mmds[i] = o

    return mmds


def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)


def _compute_mmd2(K_XX, K_XY, K_YY, unit_diagonal=False, mmd_est='unbiased'):
    """
    Based on https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    but changed to not compute the full kernel matrix at once.
    """
    if mmd_est not in ['unbiased', 'u-statistic']:
        raise ValueError(
            "mmd_est should be one of [unbiased', 'u-statistic] but got {}.".
            format(mmd_est))

    m = K_XX.shape[0]
    if K_XX.shape != (m, m):
        raise ValueError("K_XX shape should be {} but got {} instead.".format(
            (m, m), K_XX.shape))

    if K_XY.shape != (m, m):
        raise ValueError("K_XX shape should be {} but got {} instead.".format(
            (m, m), K_XY.shape))

    if K_YY.shape != (m, m):
        raise ValueError("K_XX shape should be {} but got {} instead.".format(
            (m, m), K_YY.shape))

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m) + (Kt_YY_sum + sum_diag_Y) /
                (m * m) - 2 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))

        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m - 1))

    return mmd2
