import warnings

import numpy as np

from cplx_mfa import ComplexMFA


def _complex_data(
    n_samples: int = 80,
    n_features: int = 4,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((n_samples, n_features))
        + 1j * rng.standard_normal((n_samples, n_features))
    ) / np.sqrt(2.0)


def test_fit_predict_proba_shapes() -> None:
    data = _complex_data()

    model = ComplexMFA(
        n_components=2,
        latent_dim=2,
        max_iter=30,
        tol=1.0e-3,
        random_state=0,
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        returned = model.fit(data)

    labels = model.predict(data)
    responsibilities = model.predict_proba(data)

    assert returned is model
    assert labels.shape == (80,)
    assert responsibilities.shape == (80, 2)
    assert np.all(np.isfinite(responsibilities))
    assert np.allclose(responsibilities.sum(axis=1), 1.0)


def test_fit_sets_expected_fitted_attributes() -> None:
    data = _complex_data(n_samples=60, n_features=5)

    model = ComplexMFA(
        n_components=3,
        latent_dim=2,
        max_iter=20,
        tol=1.0e-3,
        random_state=0,
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(data)

    assert model.means_.shape == (3, 5)
    assert model.loadings_.shape == (3, 5, 2)
    assert model.covariances_.shape == (3, 5, 5)
    assert model.precisions_.shape == (3, 5, 5)
    assert model.noise_variances_.shape == (3, 5)
    assert model.weights_.shape == (3,)

    assert np.iscomplexobj(model.means_)
    assert np.iscomplexobj(model.loadings_)
    assert np.iscomplexobj(model.covariances_)
    assert np.iscomplexobj(model.precisions_)
