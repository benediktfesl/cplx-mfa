import warnings

import numpy as np
import pytest

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


def _fit_model() -> ComplexMFA:
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

    return model


def test_sample_shapes() -> None:
    model = _fit_model()

    samples, labels = model.sample(n_samples=20, rng=np.random.default_rng(1))

    assert samples.shape == (20, 4)
    assert labels.shape == (20,)
    assert np.iscomplexobj(samples)
    assert np.all(np.isfinite(samples))
    assert np.all(labels >= 0)
    assert np.all(labels < model.n_components)


def test_sample_labels_are_grouped_by_component() -> None:
    model = _fit_model()

    _, labels = model.sample(n_samples=50, rng=np.random.default_rng(1))

    if labels.size > 1:
        transitions = np.diff(labels)
        assert np.all(transitions >= 0)


def test_sample_with_same_rng_seed_is_reproducible() -> None:
    model = _fit_model()

    samples_1, labels_1 = model.sample(n_samples=20, rng=np.random.default_rng(1))
    samples_2, labels_2 = model.sample(n_samples=20, rng=np.random.default_rng(1))

    assert np.allclose(samples_1, samples_2)
    assert np.array_equal(labels_1, labels_2)


@pytest.mark.parametrize("n_samples", [0, -1])
def test_sample_invalid_n_samples_raises(n_samples: int) -> None:
    model = _fit_model()

    with pytest.raises(ValueError, match="Sampling requires at least one sample"):
        model.sample(n_samples=n_samples)
