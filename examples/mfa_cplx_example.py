# Author: Benedikt Fesl <benedikt.fesl@tum.de>
# License: BSD 3 clause

import time
import numpy as np
from cplx_mfa import ComplexMFA


if __name__ == '__main__':
    """
    Test script for complex-valued MFA implementation.
    """
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_val = 100
    n_dim = 32

    # create toy data
    h_train = (rng.standard_normal((n_train, n_dim)) + 1j * rng.standard_normal((n_train, n_dim))) / np.sqrt(2)
    h_val = (rng.standard_normal((n_val, n_dim)) + 1j * rng.standard_normal((n_val, n_dim))) / np.sqrt(2)

    # use a scaled identity for the diagonal covariance
    PPCA = False
    # enforce the same diagonal matrix for all components
    lock_psis = False

    #
    # MFA training
    #
    tic = time.time()
    mfa_est = ComplexMFA(
        n_components=16,
        latent_dim=12,
        PPCA=PPCA,
        lock_psis=lock_psis,
        rs_clip=1e-6,
        max_condition_number=1.0e6,
        maxiter=100,
        verbose=False,
    )
    mfa_est.fit(h_train)
    toc = time.time()
    print(f'training done: {toc-tic} sec.')

    # Covariances & means & mixing weights
    means = mfa_est.means
    covs = mfa_est.covs
    weights = mfa_est.amps
    print(f'Sum of weights: {np.real(np.sum(weights))}')

    #
    # Responsibility evaluation
    #
    # soft responsibilities for all components
    proba_soft = mfa_est.predict_proba(h_val)
    # components with max responsibilities
    proba_max = mfa_est.predict_proba_max(h_val)

    #
    # Generate new samples
    #
    samples, comps = mfa_est.sample(n_samples=100)
    # check generated samples by computing max responsibility
    proba_max_samples = mfa_est.predict_proba_max(samples)
    print('Test completed.')