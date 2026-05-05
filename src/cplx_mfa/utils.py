"""Utility functions for complex-valued random variables and representations."""

import numpy as np


def real2cplx(vec: np.ndarray, axis: int = 0) -> np.ndarray:
    """Convert concatenated real and imaginary parts to a complex array."""
    re, im = np.split(vec, 2, axis=axis)
    return re + 1j * im


def cplx2real(vec: np.ndarray, axis: int = 0) -> np.ndarray:
    """Concatenate real and imaginary parts of a complex array."""
    return np.concatenate([vec.real, vec.imag], axis=axis)


def multivariate_normal_cplx(
    mean: np.ndarray,
    covariance: np.ndarray,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw samples from a circularly symmetric complex Gaussian distribution."""
    if rng is None:
        rng = np.random.default_rng()

    cov_sqrt = np.linalg.cholesky(covariance)
    samples = np.squeeze(cov_sqrt @ crandn(n_samples, covariance.shape[0], 1, rng=rng))

    if n_samples > 1:
        samples += np.expand_dims(mean, 0)

    return samples


def crandn(
    *shape: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw circularly symmetric complex standard normal samples."""
    if rng is None:
        rng = np.random.default_rng()

    return np.sqrt(0.5) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
