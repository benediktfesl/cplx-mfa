import numpy as np


def real2cplx(vec: np.ndarray, axis=0):
    """
    Assume vec consists of concatenated real and imaginary parts. Return the
    corresponding complex vector. Split along axis=axis.
    """
    re, im = np.split(vec, 2, axis=axis)
    return re + 1j * im


def cplx2real(vec: np.ndarray, axis=0):
    """
    Concatenate real and imaginary parts of vec along axis=axis.
    """
    return np.concatenate([vec.real, vec.imag], axis=axis)


def multivariate_normal_cplx(mean, covariance, n_samples):
    cov_sqrt = np.linalg.cholesky(covariance)
    h = np.squeeze(cov_sqrt @ crandn(n_samples, covariance.shape[0], 1))
    if n_samples > 1:
        h += np.expand_dims(mean, 0)
    return h


def crandn(*arg, rng=np.random.default_rng()):
    return np.sqrt(0.5) * (rng.standard_normal(arg) + 1j * rng.standard_normal(arg))