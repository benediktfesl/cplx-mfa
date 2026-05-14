"""Complex-valued mixture of factor analyzers."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import inv
from sklearn import cluster

from . import utils as ut


class ComplexMFA:
    """Complex-valued mixture of factor analyzers.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    latent_dim : int
        Latent dimensionality of each factor analyzer.
    ppca : bool, default=False
        If True, use an isotropic diagonal covariance per component.
    lock_psis : bool, default=False
        If True, use a shared diagonal covariance across components.
    zero_mean : bool, default=False
        If True, enforce zero component means during initialization and EM.
    rs_clip : float, default=0.0
        Lower clipping value for responsibilities during EM.
    max_condition_number : float, default=1e6
        Scaling factor used for random loading initialization.
    max_iter : int, default=100
        Maximum number of EM iterations.
    tol : float, default=1e-4
        Relative convergence tolerance for the EM lower bound.
    random_state : int, numpy.random.Generator, or None, default=None
        Random seed or random number generator used for initialization.
    verbose : bool, default=True
        If True, print EM progress.

    Attributes
    ----------
    means_ : ndarray of shape (n_components, n_features)
        Fitted component means.
    loadings_ : ndarray of shape (n_components, n_features, latent_dim)
        Fitted factor loading matrices.
    covariances_ : ndarray of shape (n_components, n_features, n_features)
        Fitted component covariance matrices.
    precisions_ : ndarray of shape (n_components, n_features, n_features)
        Inverse component covariance matrices.
    noise_variances_ : ndarray of shape (n_components, n_features)
        Fitted diagonal noise variances.
    weights_ : ndarray of shape (n_components,)
        Fitted mixture weights.
    lower_bound_history_ : list of float
        EM lower-bound values collected during fitting.
    """

    def __init__(
        self,
        n_components: int,
        latent_dim: int,
        ppca: bool = False,
        lock_psis: bool = False,
        zero_mean: bool = False,
        rs_clip: float = 0.0,
        max_condition_number: float = 1.0e6,
        max_iter: int = 100,
        tol: float = 1.0e-4,
        random_state: int | np.random.Generator | None = None,
        verbose: bool = True,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        if latent_dim < 1:
            raise ValueError("latent_dim must be at least 1.")
        if rs_clip < 0.0:
            raise ValueError("rs_clip must be non-negative.")
        if max_condition_number <= 0.0:
            raise ValueError("max_condition_number must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if tol <= 0.0:
            raise ValueError("tol must be positive.")

        self.n_components = n_components
        self.latent_dim = latent_dim
        self.ppca = ppca
        self.lock_psis = lock_psis
        self.zero_mean = zero_mean
        self.rs_clip = rs_clip
        self.max_condition_number = float(max_condition_number)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        self._rng = self._make_rng(random_state)
        self._sklearn_random_state = self._make_sklearn_random_state(random_state)

        self.lower_bound_history_: list[float] = []

        self.means_: np.ndarray | None = None
        self.loadings_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None
        self.precisions_: np.ndarray | None = None
        self.noise_variances_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

        self._n_samples: int | None = None
        self._n_features: int | None = None
        self._responsibilities: np.ndarray | None = None
        self._log_likelihoods: np.ndarray | None = None

    @staticmethod
    def _make_rng(
        random_state: int | np.random.Generator | None,
    ) -> np.random.Generator:
        """Create a NumPy random generator from a seed or generator."""
        if isinstance(random_state, np.random.Generator):
            return random_state

        return np.random.default_rng(random_state)

    @staticmethod
    def _make_sklearn_random_state(
        random_state: int | np.random.Generator | None,
    ) -> int | None:
        """Create a scikit-learn-compatible random state.

        scikit-learn estimators accept integer seeds but not NumPy Generator
        instances. If a Generator is provided, draw one deterministic integer
        seed from it and use that for scikit-learn initialization.
        """
        if isinstance(random_state, int):
            return random_state

        if isinstance(random_state, np.random.Generator):
            return int(random_state.integers(0, np.iinfo(np.int32).max))

        return None

    @staticmethod
    def _validate_input_data(data: np.ndarray) -> np.ndarray:
        """Validate and normalize input data."""
        data = np.asarray(data)

        if data.ndim != 2:
            raise ValueError(
                "data must be a 2D array of shape (n_samples, n_features)."
            )
        if data.shape[0] < 1:
            raise ValueError("data must contain at least one sample.")
        if data.shape[1] < 1:
            raise ValueError("data must contain at least one feature.")

        if not np.iscomplexobj(data):
            data = data.astype(complex)

        if not np.all(np.isfinite(data)):
            raise ValueError("data must not contain NaN or infinite values.")

        return data

    def _validate_prediction_data(self, data: np.ndarray) -> np.ndarray:
        """Validate input data for prediction methods."""
        self._check_is_fitted()
        data = self._validate_input_data(data)

        if data.shape[1] != self.means_.shape[1]:
            raise ValueError(
                "data has incompatible number of features. "
                f"Expected {self.means_.shape[1]}, got {data.shape[1]}."
            )

        return data

    def _check_is_fitted(self) -> None:
        """Raise an error if the model has not been fitted."""
        if (
            self.means_ is None
            or self.loadings_ is None
            or self.covariances_ is None
            or self.precisions_ is None
            or self.noise_variances_ is None
            or self.weights_ is None
        ):
            raise RuntimeError("The model must be fitted before this method is called.")

    def fit(self, data: np.ndarray) -> ComplexMFA:
        """Fit the complex-valued mixture of factor analyzers.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            Complex-valued training data.

        Returns
        -------
        self : ComplexMFA
            Fitted estimator.
        """
        data = self._validate_input_data(data)

        self._n_samples, self._n_features = data.shape
        self._responsibilities = np.zeros((self.n_components, self._n_samples))
        self.lower_bound_history_ = []

        self._initialize(data)
        self._run_em(data)

        self._responsibilities = None
        self._log_likelihoods = None

        return self

    def _initialize(self, data: np.ndarray) -> None:
        """Initialize mixture parameters."""
        if self.zero_mean:
            self.means_ = np.zeros((self.n_components, self._n_features), dtype=complex)
        else:
            kmeans = cluster.KMeans(
                n_clusters=self.n_components,
                n_init=1,
                random_state=self._sklearn_random_state,
            ).fit(ut.cplx2real(data, axis=1))

            self.means_ = ut.real2cplx(kmeans.cluster_centers_, axis=1)

        self.loadings_ = (
            self._rng.standard_normal(
                (self.n_components, self._n_features, self.latent_dim)
            )
            + 1j
            * self._rng.standard_normal(
                (self.n_components, self._n_features, self.latent_dim)
            )
        ) / np.sqrt(2.0 * self.max_condition_number)

        initial_noise_variances = np.clip(np.var(data, axis=0), 1.0e-6, np.inf)

        self.noise_variances_ = np.tile(
            initial_noise_variances[None, :],
            (self.n_components, 1),
        )

        self.covariances_ = np.zeros(
            (self.n_components, self._n_features, self._n_features),
            dtype=complex,
        )
        self.precisions_ = np.zeros_like(self.covariances_)

        self.weights_ = self._rng.random(self.n_components)
        self.weights_ /= np.sum(self.weights_)

        self._update_covariances()

    def _run_em(self, data: np.ndarray) -> None:
        """Run the expectation-maximization algorithm."""
        lower_bound = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            self._em_step(data)

            new_lower_bound = float(np.sum(self._log_likelihoods))
            self.lower_bound_history_.append(new_lower_bound)

            if self.verbose:
                print(
                    f"Iteration {iteration} | lower bound: {new_lower_bound:.5f}",
                    end="\r",
                )

            denominator = max(abs(new_lower_bound), np.finfo(float).eps)
            relative_change = abs((new_lower_bound - lower_bound) / denominator)

            if iteration > 5 and relative_change < self.tol:
                converged = True
                break

            lower_bound = new_lower_bound

        if converged:
            if self.verbose:
                print(f"EM converged after {iteration} iterations")
                print(f"Final NLL = {-new_lower_bound}")
        else:
            warnings.warn(
                f"EM did not converge after {self.max_iter} iterations.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _em_step(self, data: np.ndarray) -> None:
        """Run one EM update over all mixture components."""
        self._check_is_fitted()

        self._log_likelihoods, self._responsibilities = self._calculate_probabilities(
            data
        )
        responsibility_sums = np.sum(self._responsibilities, axis=1)
        safe_responsibility_sums = np.maximum(
            responsibility_sums,
            np.finfo(float).eps,
        )

        betas = np.transpose(self.loadings_.conj(), [0, 2, 1]) @ self.precisions_

        for component in range(self.n_components):
            centered_data = data.T - self.means_[component, :, None]

            latents = betas[component] @ centered_data
            latent_outer_products = latents[:, None, :] * latents[None, :, :].conj()
            beta_loadings = betas[component] @ self.loadings_[component]

            latent_covariances = (
                np.eye(self.latent_dim)[:, :, None]
                - beta_loadings[:, :, None]
                + latent_outer_products
            )

            loadings_latents = self.loadings_[component] @ latents

            if self.zero_mean:
                self.means_[component] = np.zeros(self._n_features, dtype=complex)
            else:
                self.means_[component] = (
                    np.sum(
                        self._responsibilities[component] * (data.T - loadings_latents),
                        axis=1,
                    )
                    / safe_responsibility_sums[component]
                )

            zeroed_data = data.T - self.means_[component, :, None]

            weighted_cross_covariance = np.dot(
                zeroed_data[:, None, :] * latents[None, :, :].conj(),
                self._responsibilities[component],
            )
            weighted_latent_covariance = np.dot(
                latent_covariances,
                self._responsibilities[component],
            )

            self.loadings_[component] = weighted_cross_covariance @ inv(
                weighted_latent_covariance
            )

            residual = zeroed_data - loadings_latents
            noise_variance = np.real(
                np.dot(
                    residual * zeroed_data.conj(),
                    self._responsibilities[component],
                )
                / safe_responsibility_sums[component]
            )

            self.noise_variances_[component] = np.clip(noise_variance, 1.0e-6, np.inf)

            if self.ppca:
                self.noise_variances_[component] = np.mean(
                    self.noise_variances_[component]
                ) * np.ones(self._n_features)

            self.weights_[component] = responsibility_sums[component] / data.shape[0]

        if self.lock_psis:
            shared_noise_variance = (
                responsibility_sums @ self.noise_variances_
            ) / np.sum(responsibility_sums)

            self.noise_variances_ = np.tile(
                shared_noise_variance[None, :],
                (self.n_components, 1),
            )

        self._update_covariances()

    def _update_covariances(self) -> None:
        """Update component covariances and precisions."""
        self._check_is_fitted()

        self.covariances_ = self.loadings_ @ np.transpose(
            self.loadings_.conj(),
            [0, 2, 1],
        )

        for component in range(self.n_components):
            self.covariances_[component] += np.diag(self.noise_variances_[component])

        self.precisions_ = self._invert_covariances()

    def _calculate_probabilities(
        self,
        data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate log likelihoods and component responsibilities."""
        self._check_is_fitted()

        n_samples = data.shape[0]
        log_responsibilities = np.zeros((self.n_components, n_samples))

        _, log_determinants = np.linalg.slogdet(self.covariances_)

        for component in range(self.n_components):
            log_responsibilities[component] = (
                np.log(self.weights_[component])
                + self._log_complex_normal_without_logdet(component, data)
                - np.real(log_determinants[component])
            )

        log_likelihoods = self._log_sum_exp(log_responsibilities)
        log_responsibilities -= log_likelihoods[None, :]

        responsibilities = np.exp(log_responsibilities)

        if self.rs_clip > 0.0:
            responsibilities = np.maximum(responsibilities, self.rs_clip)
            responsibilities /= np.sum(responsibilities, axis=0, keepdims=True)

        return log_likelihoods, responsibilities

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Calculate component responsibilities for each sample.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            Complex-valued input data.

        Returns
        -------
        responsibilities : ndarray of shape (n_samples, n_components)
            Posterior component probabilities.
        """
        data = self._validate_prediction_data(data)

        log_responsibilities = np.zeros((self.n_components, data.shape[0]))

        for component in range(self.n_components):
            log_responsibilities[component] = np.log(
                self.weights_[component]
            ) + self._log_complex_normal(component, data)

        log_likelihoods = self._log_sum_exp(log_responsibilities)
        log_responsibilities -= log_likelihoods[None, :]

        return np.exp(log_responsibilities).T

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict the most likely component label for each sample.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            Complex-valued input data.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Most likely component labels.
        """
        return np.argmax(self.predict_proba(data), axis=1)

    def _log_complex_normal(
        self,
        component: int,
        data: np.ndarray,
    ) -> np.ndarray:
        """Calculate complex Gaussian log likelihoods for one component."""
        _, log_determinant = np.linalg.slogdet(self.covariances_[component])

        centered_data = (data - self.means_[component]).T
        transformed_data = self.precisions_[component] @ centered_data
        quadratic_form = np.sum(centered_data.conj() * transformed_data, axis=0)

        return np.real(
            -np.log(np.pi) * data.shape[1] - log_determinant - quadratic_form
        )

    def _log_complex_normal_without_logdet(
        self,
        component: int,
        data: np.ndarray,
    ) -> np.ndarray:
        """Calculate complex Gaussian log likelihoods without log determinant."""
        centered_data = (data - self.means_[component]).T
        transformed_data = self.precisions_[component] @ centered_data
        quadratic_form = np.sum(centered_data.conj() * transformed_data, axis=0)

        return np.real(-np.log(np.pi) * data.shape[1] - quadratic_form)

    @staticmethod
    def _log_sum_exp(log_likelihoods: np.ndarray) -> np.ndarray:
        """Calculate log-sum-exp over components in a numerically stable way."""
        log_likelihoods = np.atleast_2d(log_likelihoods)
        max_log_likelihood = np.max(log_likelihoods, axis=0)

        return max_log_likelihood + np.log(
            np.sum(
                np.exp(log_likelihoods - max_log_likelihood[None, :]),
                axis=0,
            )
        )

    def _invert_covariances(self) -> np.ndarray:
        """Calculate inverse covariances using the Woodbury identity."""
        self._check_is_fitted()

        noise_precisions = 1.0 / self.noise_variances_

        inner = np.linalg.pinv(
            np.eye(self.latent_dim)[None, :, :]
            + (
                np.transpose(self.loadings_.conj(), [0, 2, 1])
                * noise_precisions[:, None, :]
            )
            @ self.loadings_
        )

        correction = (
            noise_precisions[:, :, None]
            * (self.loadings_ @ inner @ np.transpose(self.loadings_.conj(), [0, 2, 1]))
            * noise_precisions[:, None, :]
        )

        for component in range(self.n_components):
            correction[component] -= np.diag(noise_precisions[component])

        return -correction

    def sample(
        self,
        n_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate random samples from the fitted complex Gaussian mixture.

        Samples are returned grouped by mixture component. The corresponding
        component labels are returned in the same order.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
        rng : numpy.random.Generator, optional
            Random number generator used for sampling.

        Returns
        -------
        samples : ndarray of shape (n_samples, n_features)
            Randomly generated samples.
        labels : ndarray of shape (n_samples,)
            Component labels. Labels are grouped by component because samples
            are generated component-wise.
        """
        self._check_is_fitted()

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': "
                f"{n_samples}. Sampling requires at least one sample."
            )

        if rng is None:
            rng = np.random.default_rng()

        n_samples_per_component = rng.multinomial(n_samples, self.weights_)

        samples = np.vstack(
            [
                ut.multivariate_normal_cplx(
                    mean,
                    covariance,
                    int(component_samples),
                    rng=rng,
                )
                for mean, covariance, component_samples in zip(
                    self.means_,
                    self.covariances_,
                    n_samples_per_component,
                    strict=True,
                )
                if component_samples > 0
            ]
        )

        labels = np.concatenate(
            [
                np.full(component_samples, component, dtype=int)
                for component, component_samples in enumerate(n_samples_per_component)
                if component_samples > 0
            ]
        )

        return samples, labels
