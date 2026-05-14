import warnings

import numpy as np

from cplx_mfa import ComplexMFA


def _offset_complex_data(
    n_samples: int = 120,
    n_features: int = 3,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    offset = np.array([2.0 + 1.0j, -1.0 + 0.5j, 0.25 - 1.5j])
    noise = (
        rng.standard_normal((n_samples, n_features))
        + 1j * rng.standard_normal((n_samples, n_features))
    ) / np.sqrt(2.0)

    return offset[None, :] + noise


def test_zero_mean_enforces_zero_component_means_after_fit() -> None:
    data = _offset_complex_data()

    model = ComplexMFA(
        n_components=2,
        latent_dim=1,
        zero_mean=True,
        max_iter=10,
        random_state=0,
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(data)

    assert model.zero_mean is True
    assert model.means_.shape == (2, 3)
    assert np.allclose(model.means_, 0.0)


def test_default_fit_keeps_learned_component_means() -> None:
    data = _offset_complex_data()

    model = ComplexMFA(
        n_components=1,
        latent_dim=1,
        max_iter=10,
        random_state=0,
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(data)

    assert model.zero_mean is False
    assert not np.allclose(model.means_, 0.0)
