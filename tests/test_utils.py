import numpy as np

from cplx_mfa import utils


def test_complex_real_roundtrip_axis_0() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))

    real_data = utils.cplx2real(data, axis=0)
    reconstructed = utils.real2cplx(real_data, axis=0)

    assert reconstructed.shape == data.shape
    assert np.allclose(reconstructed, data)


def test_complex_real_roundtrip_axis_1() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))

    real_data = utils.cplx2real(data, axis=1)
    reconstructed = utils.real2cplx(real_data, axis=1)

    assert reconstructed.shape == data.shape
    assert np.allclose(reconstructed, data)


def test_crandn_shape_and_complex_dtype() -> None:
    samples = utils.crandn(5, 3, rng=np.random.default_rng(0))

    assert samples.shape == (5, 3)
    assert np.iscomplexobj(samples)
    assert np.all(np.isfinite(samples))


def test_multivariate_normal_cplx_shape_and_complex_dtype() -> None:
    mean = np.zeros(3, dtype=complex)
    covariance = np.eye(3, dtype=complex)

    samples = utils.multivariate_normal_cplx(
        mean,
        covariance,
        n_samples=10,
        rng=np.random.default_rng(0),
    )

    assert samples.shape == (10, 3)
    assert np.iscomplexobj(samples)
    assert np.all(np.isfinite(samples))
