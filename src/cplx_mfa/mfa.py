# Original code base:
# Authors: Ross Fadely, Daniel Foreman-Mackey, David W. Hogg, and contributors.
# License: MIT

# Contributions with considerable memory savings, complexity speed-ups, and complex-valued extension:
# Author: Benedikt Fesl <benedikt.fesl@tum.de>
# License: BSD 3 clause

import numpy as np
from scipy.linalg import inv
from sklearn import cluster
from . import utils as ut


class ComplexMFA:
    """
    Complex-valued implementation of the EM algorithm for fitting Mixture of Factor Analyzers.

    internal variables:
    `K`:           Number of components
    `M`:           Latent dimensionality
    `D`:           Data dimensionality
    `N`:           Number of data points
    `data`:        (N,D) array of observations
    `latents`:     (K,M,N) array of latent variables
    `latent_covs`: (K,M,M,N) array of latent covariances
    `lambdas`:     (K,M,D) array of loadings
    `psis`:        (K,D) array of diagonal variance values
    `rs`:          (K,N) array of responsibilities
    `amps`:        (K) array of component amplitudes
    maxiter:
        The maximum number of iterations to try.
    tol:
        The tolerance on the relative change in the loss function that
        controls convergence.
    verbose:
        Print all the messages?
    """

    def __init__(self,
                 n_components,
                 latent_dim,
                 PPCA=False,
                 lock_psis=False,
                 rs_clip=0.0,
                 max_condition_number=1.e6,
                 maxiter=100,
                 tol=1e-4,
                 verbose=True,
                 ):

        # required
        self.n_components = n_components
        self.M = latent_dim

        # options
        self.PPCA = PPCA
        self.lock_psis = lock_psis
        self.rs_clip = rs_clip
        self.L_all = list()
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = verbose
        self.max_condition_number = float(max_condition_number)
        assert rs_clip >= 0.0

        self.N = None
        self.D = None
        self.betas = None
        self.latents = None
        self.latent_covs = None
        self.kmeans_rs = None
        self.rs = None
        self.logLs = None
        self.batch_size = None

        # member variables used for calculating, e.g., responsibilities
        self._means = None
        self._lambdas = None
        self._covs = None
        self._inv_covs = None
        self._psis = None
        # fixed variables after training. Not used for calculating, e.g., responsibilities
        self.means = None
        self.lambdas = None
        self.covs = None
        self.inv_covs = None
        self.psis = None


    def fit(self, data):
        # covs = low-rank + diagonal cov
        # empty arrays to be filled
        self.N = data.shape[0]
        self.D = data.shape[1]
        self.rs = np.zeros((self.n_components, self.N))
        self._covs = np.zeros((self.n_components, self.D, self.D), dtype=complex)
        self._inv_covs = np.zeros_like(self._covs)

        # initialize
        self._initialize(data)
        # run em algorithm
        self.run_em(data)
        # delete unnecessary memory
        del self.latents, self.latent_covs, self.rs, self.kmeans_rs, self.betas, self.logLs
        # store fixed parameters
        self.means = self._means.copy()
        self.covs = self._covs.copy()
        self.inv_covs = self._inv_covs.copy()
        self.psis = self._psis.copy()
        self.lambdas = self._lambdas.copy()


    def _initialize(self, data):
        # Run K-means
        Kmeans = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                                ).fit(ut.cplx2real(data, axis=1))
        self._means = ut.real2cplx(Kmeans.cluster_centers_, axis=1)
        del Kmeans

        # Randomly assign factor loadings
        self._lambdas = (np.random.randn(self.n_components, self.D, self.M) +
                         1j * np.random.randn(self.n_components, self.D, self.M)) / np.sqrt(
            self.max_condition_number) / np.sqrt(2)

        # Set (high rank) variance to variance of all data, along a dimension
        self._psis = np.tile(np.var(data, axis=0)[None, :], (self.n_components, 1))

        # Set initial covs
        self._update_covs()

        # Randomly assign the amplitudes.
        self.amps = np.random.rand(self.n_components)
        self.amps /= np.sum(self.amps)


    def run_em(self, data):
        """
        Run the EM algorithm.
        """
        L = -np.inf
        for i in range(self.maxiter):
            self._EM_per_component(data, self.PPCA)
            newL = self.logLs.sum()
            self.L_all.append(newL)
            if self.verbose:
                print(f'Iteration {i} | lower bound: {newL:.5f}', end='\r')
            dL = np.abs((newL - L) / newL)
            if i > 5 and dL < self.tol:
                break
            L = newL

        if i < self.maxiter - 1:
            if self.verbose:
                print("EM converged after {0} iterations".format(i))
                print("Final NLL = {0}".format(-newL))
        else:
            print("\nWarning: EM didn't converge after {0} iterations"
                  .format(i))


    def _EM_per_component(self, data, PPCA):
        # resposibilities and likelihoods
        self.logLs, self.rs = self._calc_probs(data)
        sumrs = np.sum(self.rs, axis=1)

        #pre-compute betas
        betas = np.transpose(self._lambdas.conj(), [0, 2, 1]) @ self._inv_covs

        for k in range(self.n_components):
            #E-step: Calculation of latents per component
            # latent values
            latents = betas[k] @ (data.T - self._means[k, :, None])

            # latent empirical covariance
            step1 = latents[:, None, :] * latents[None, :, :].conj()
            step2 = betas[k] @ self._lambdas[k]
            latent_covs = np.eye(self.M)[:, :, None] - step2[:, :, None] + step1

            #M-step: Calculation of new parameters per component
            lambdalatents = self._lambdas[k] @ latents
            self._means[k] = np.sum(self.rs[k] * (data.T - lambdalatents), axis=1) / sumrs[k]
            zeroed = data.T - self._means[k, :, None]
            self._lambdas[k] = np.dot(np.dot(zeroed[:, None, :] * latents[None, :, :].conj(), self.rs[k]),
                                      inv(np.dot(latent_covs, self.rs[k])))
            psis = np.real(np.dot((zeroed - lambdalatents) * zeroed.conj(), self.rs[k]) / sumrs[k])
            self._psis[k] = np.clip(psis, 1e-6, np.inf)
            if PPCA:
                self._psis[k] = np.mean(self._psis[k]) * np.ones(self.D)
            self.amps[k] = sumrs[k] / data.shape[0]

        if self.lock_psis:
            psi = np.dot(sumrs, self._psis) / np.sum(sumrs)
            self._psis = np.full_like(self._psis, psi)
        self._update_covs()


    def _update_covs(self):
        """
        Update self.cov for responsibility, logL calc
        """
        self._covs = self._lambdas @ np.transpose(self._lambdas.conj(), [0,2,1])
        for k in range(self.n_components):
            self._covs[k] += np.diag(self._psis[k])
        self._inv_covs = self._invert_cov_all()


    def _calc_probs(self, data):
        """
        Calculate log likelihoods, responsibilites for each datum
        under each component.
        """
        logrs = np.zeros((self.n_components, self.N))
        #pre-compute logdets
        sgn, logdet = np.linalg.slogdet(self._covs)
        for k in range(self.n_components):
            logrs[k] = np.log(self.amps[k]) + self._log_multi_gauss_nodet(k, data) - logdet[k]

        # here lies some ghetto log-sum-exp...
        # nothing like a little bit of overflow to make your day better!
        L = self._log_sum(logrs)
        logrs -= L[None, :]
        if self.rs_clip > 0.0:
            logrs = np.clip(logrs, np.log(self.rs_clip), np.inf)
        return L, np.exp(logrs)


    def predict_proba(self, data):
        """
        Calculate responsibilites.
        """
        logrs = np.zeros((self.n_components, data.shape[0]))
        for k in range(self.n_components):
            logrs[k] = np.log(self.amps[k]) + self._log_multi_gauss(k, data)

        # here lies some ghetto log-sum-exp...
        # nothing like a little bit of overflow to make your day better!
        L = self._log_sum(logrs)
        logrs -= L[None, :]
        # if self.rs_clip > 0.0:
        #    logrs = np.clip(logrs, np.log(self.rs_clip), np.inf)
        return np.exp(logrs).T


    def predict_proba_max(self, data):
        """
        Calculate label with highest responsibility (argmax).
        """
        logrs = np.zeros((self.n_components, data.shape[0]))
        for k in range(self.n_components):
            logrs[k] = np.log(self.amps[k]) + self._log_multi_gauss(k, data)
        return np.exp(logrs).argmax(axis=0)


    def _log_multi_gauss(self, k, data):
        """
        Gaussian log likelihood of the data for component k.
        """
        sgn, logdet = np.linalg.slogdet(self._covs[k])
        assert sgn > 0
        X1 = (data - self._means[k]).T
        X2 = self._inv_covs[k] @ X1
        p = np.sum(X1.conj() * X2, axis=0)
        return np.real(- np.log(np.pi) * data.shape[1] - logdet - p)


    def _log_multi_gauss_nodet(self, k, data):
        """
        Gaussian log likelihood of the data for component k without determinant computation.
        """
        X1 = (data - self._means[k]).T
        X2 = self._inv_covs[k] @ X1
        p = np.sum(X1.conj() * X2, axis=0)
        return np.real(- np.log(np.pi) * data.shape[1] - p)


    def _log_sum(self, loglikes):
        """
        Calculate sum of log likelihoods
        """
        loglikes = np.atleast_2d(loglikes)
        a = np.max(loglikes, axis=0)
        return a + np.log(np.sum(np.exp(loglikes - a[None, :]), axis=0))


    def _invert_cov_all(self):
        """
        Calculate inverse covariance of mofa or ppca model,
        using inversion lemma of all components at once.
        """
        psiI = 1 / self._psis
        inv_inner = np.linalg.pinv(np.eye(self.M)[None, :, :] + (np.transpose(self._lambdas.conj(), [0, 2, 1]) * psiI[:, None, :]) @ self._lambdas)
        step = psiI[:, :, None] * (self._lambdas @ inv_inner @ np.transpose(self._lambdas.conj(), [0,2,1])) * psiI[:, None, :]
        for k in range(self.n_components):
            step[k] -= np.diag(psiI[k])
        return -step


    def sample(self, n_samples=1, rng=np.random.default_rng()):
        """Generate random samples from the fitted Gaussian distribution.
        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
        rng: np.random.RandomState instance.
        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.
        y : array, shape (nsamples,)
            Component labels.
        """

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % n_samples
            )

        _, n_features = self.means.shape
        if rng is None:
            rng = np.random.RandomState(12531616843613)
        n_samples_comp = rng.multinomial(n_samples, self.amps)

        X = np.vstack(
            [
                #rng.multivariate_normal(mean, covariance, int(sample))
                ut.multivariate_normal_cplx(mean, covariances, int(sample))
                for (mean, covariances, sample) in zip(
                    self.means, self.covs, n_samples_comp
                )
            ]
        )


        y = np.concatenate(
            [np.full(sample, j, dtype=int) for j, sample in enumerate(n_samples_comp)]
        )

        return (X, y)