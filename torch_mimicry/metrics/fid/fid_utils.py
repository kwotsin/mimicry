"""
Helper functions for calculating FID as adopted from the official FID code:
https://github.com/bioinf-jku/TTUR/blob/master/fid.py
"""
import numpy as np
from scipy import linalg

from torch_mimicry.metrics.inception_model import inception_utils


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.

    Args:
        mu1 : Numpy array containing the activations of the pool_3 layer of the
            inception net ( like returned by the function 'get_predictions')
            for generated samples.
        mu2: The sample mean over activations of the pool_3 layer, precalcualted
            on an representive data set.
        sigma1 (ndarray): The covariance matrix over activations of the pool_3 layer for
            generated samples.
        sigma2: The covariance matrix over activations of the pool_3 layer,
            precalcualted on an representive data set.

    Returns:
        np.float64: The Frechet Distance.
    """
    if mu1.shape != mu2.shape or sigma1.shape != sigma2.shape:
        raise ValueError(
            "(mu1, sigma1) should have exactly the same shape as (mu2, sigma2)."
        )

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(
            "WARNING: fid calculation produces singular product; adding {} to diagonal of cov estimates"
            .format(eps))

        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(
        sigma2) - 2 * tr_covmean


def calculate_activation_statistics(images, sess, batch_size=50, verbose=True):
    """
    Calculation of the statistics used by the FID.

    Args:
        images (ndarray): Numpy array of shape (N, H, W, 3) and values in
            the range [0, 255].
        sess (Session): TensorFlow session object.
        batch_size (int): Batch size for inference.
        verbose (bool): If True, prints out logging information.

    Returns:
        ndarray: Mean of inception features from samples.
        ndarray: Covariance of inception features from samples.
    """
    act = inception_utils.get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    return mu, sigma
