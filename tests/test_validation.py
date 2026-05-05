import warnings

import numpy as np
import pytest

from cplx_mfa import ComplexMFA


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_components": 0, "latent_dim": 1}, "n_components must be at least 1"),
        ({"n_components": 1, "latent_dim": 0}, "latent_dim must be at least 1"),
        (
            {"n_components": 1, "latent_dim": 1, "rs_clip": -1.0},
            "rs_clip must be non-negative",
        ),
        (
            {"n_components": 1, "latent_dim": 1, "max_condition_number": 0.0},
            "max_condition_number must be positive",
        ),
        (
            {"n_components": 1, "latent_dim": 1, "max_iter": 0},
            "max_iter must be at least 1",
        ),
        (
            {"n_components": 1, "latent_dim": 1, "tol": 0.0},
            "tol must be positive",
        ),
    ],
)
def test_invalid_hyperparameters_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ComplexMFA(**kwargs)


def test_one_dimensional_input_raises() -> None:
    data = np.ones(10, dtype=complex)
    model = ComplexMFA(n_components=1, latent_dim=1, verbose=False)

    with pytest.raises(ValueError, match="2D array"):
        model.fit(data)


def test_empty_input_raises() -> None:
    data = np.ones((0, 2), dtype=complex)
    model = ComplexMFA(n_components=1, latent_dim=1, verbose=False)

    with pytest.raises(ValueError, match="at least one sample"):
        model.fit(data)


def test_nan_input_raises() -> None:
    data = np.ones((10, 2), dtype=complex)
    data[0, 0] = np.nan
    model = ComplexMFA(n_components=1, latent_dim=1, verbose=False)

    with pytest.raises(ValueError, match="NaN or infinite"):
        model.fit(data)


def test_real_input_is_accepted_and_cast_to_complex() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((20, 2))

    model = ComplexMFA(
        n_components=1,
        latent_dim=1,
        max_iter=5,
        random_state=0,
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(data)

    assert np.iscomplexobj(model.means_)
    assert model.means_.shape == (1, 2)


def test_prediction_with_wrong_number_of_features_raises() -> None:
    rng = np.random.default_rng(0)
    train_data = (
        rng.standard_normal((20, 2)) + 1j * rng.standard_normal((20, 2))
    ) / np.sqrt(2.0)
    test_data = np.ones((5, 3), dtype=complex)

    model = ComplexMFA(
        n_components=1,
        latent_dim=1,
        max_iter=5,
        random_state=0,
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(train_data)

    with pytest.raises(ValueError, match="incompatible number of features"):
        model.predict_proba(test_data)


def test_too_few_samples_for_components_raises_from_initialization() -> None:
    data = np.ones((2, 3), dtype=complex)
    model = ComplexMFA(n_components=3, latent_dim=1, verbose=False)

    with pytest.raises(ValueError):
        model.fit(data)


def test_constant_input_does_not_create_zero_noise_variances() -> None:
    data = np.ones((20, 2), dtype=complex)

    model = ComplexMFA(
        n_components=1,
        latent_dim=1,
        max_iter=2,
        random_state=0,
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(data)

    assert np.all(model.noise_variances_ > 0.0)
    assert np.all(np.isfinite(model.noise_variances_))
