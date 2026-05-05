import warnings

import numpy as np

from cplx_mfa import ComplexMFA


def _complex_data(
    n_samples: int = 120,
    n_features: int = 4,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((n_samples, n_features))
        + 1j * rng.standard_normal((n_samples, n_features))
    ) / np.sqrt(2.0)


def test_lower_bound_history_is_finite() -> None:
    data = _complex_data()

    model = ComplexMFA(
        n_components=2,
        latent_dim=2,
        max_iter=20,
        tol=1.0e-3,
        random_state=0,
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(data)

    lower_bounds = np.asarray(model.lower_bound_history_)

    assert lower_bounds.ndim == 1
    assert lower_bounds.size >= 1
    assert np.all(np.isfinite(lower_bounds))


def test_lower_bound_history_has_at_most_max_iter_entries() -> None:
    data = _complex_data()

    model = ComplexMFA(
        n_components=2,
        latent_dim=2,
        max_iter=10,
        tol=1.0e-3,
        random_state=0,
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(data)

    assert 1 <= len(model.lower_bound_history_) <= model.max_iter
