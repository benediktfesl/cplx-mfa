import warnings

import numpy as np

from cplx_mfa import ComplexMFA


def _complex_data(
    n_samples: int = 100,
    n_features: int = 4,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((n_samples, n_features))
        + 1j * rng.standard_normal((n_samples, n_features))
    ) / np.sqrt(2.0)


def test_fit_with_same_integer_random_state_is_reproducible() -> None:
    data = _complex_data()

    model_1 = ComplexMFA(
        n_components=2,
        latent_dim=2,
        max_iter=20,
        tol=1.0e-3,
        random_state=0,
        verbose=False,
    )
    model_2 = ComplexMFA(
        n_components=2,
        latent_dim=2,
        max_iter=20,
        tol=1.0e-3,
        random_state=0,
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model_1.fit(data)
        model_2.fit(data)

    assert np.allclose(model_1.means_, model_2.means_)
    assert np.allclose(model_1.loadings_, model_2.loadings_)
    assert np.allclose(model_1.weights_, model_2.weights_)
    assert np.allclose(model_1.predict_proba(data), model_2.predict_proba(data))


def test_fit_with_same_generator_seed_is_reproducible() -> None:
    data = _complex_data()

    model_1 = ComplexMFA(
        n_components=2,
        latent_dim=2,
        max_iter=20,
        tol=1.0e-3,
        random_state=np.random.default_rng(0),
        verbose=False,
    )
    model_2 = ComplexMFA(
        n_components=2,
        latent_dim=2,
        max_iter=20,
        tol=1.0e-3,
        random_state=np.random.default_rng(0),
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model_1.fit(data)
        model_2.fit(data)

    assert np.allclose(model_1.means_, model_2.means_)
    assert np.allclose(model_1.loadings_, model_2.loadings_)
    assert np.allclose(model_1.weights_, model_2.weights_)
