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


def test_weights_sum_to_one_after_fit() -> None:
    data = _complex_data()

    model = ComplexMFA(
        n_components=3,
        latent_dim=2,
        max_iter=30,
        tol=1.0e-3,
        random_state=0,
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(data)

    assert model.weights_.shape == (3,)
    assert np.all(model.weights_ >= 0.0)
    assert np.allclose(model.weights_.sum(), 1.0)
