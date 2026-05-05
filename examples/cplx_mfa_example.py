"""Example usage of the complex-valued MFA estimator."""

from __future__ import annotations

import time

import numpy as np

from cplx_mfa import ComplexMFA


def make_complex_normal_data(
    rng: np.random.Generator,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    """Generate circularly symmetric complex Gaussian toy data."""
    return (
        rng.standard_normal((n_samples, n_features))
        + 1j * rng.standard_normal((n_samples, n_features))
    ) / np.sqrt(2.0)


def main() -> None:
    """Fit ComplexMFA on toy data and evaluate basic model operations."""
    rng = np.random.default_rng(12345)

    n_train = 300
    n_val = 50
    n_features = 8

    train_data = make_complex_normal_data(rng, n_train, n_features)
    val_data = make_complex_normal_data(rng, n_val, n_features)

    model = ComplexMFA(
        n_components=4,
        latent_dim=2,
        ppca=False,
        lock_psis=False,
        rs_clip=1.0e-6,
        max_condition_number=1.0e6,
        max_iter=100,
        random_state=0,
        verbose=False,
    )

    start_time = time.perf_counter()
    model.fit(train_data)
    elapsed_time = time.perf_counter() - start_time

    responsibilities = model.predict_proba(val_data)
    labels = model.predict(val_data)

    samples, sample_labels = model.sample(
        n_samples=20,
        rng=np.random.default_rng(1),
    )
    sample_predictions = model.predict(samples)

    print(f"Training completed in {elapsed_time:.3f} s.")
    print(f"Mixture weight sum: {np.sum(model.weights_):.6f}")
    print(f"Validation responsibilities shape: {responsibilities.shape}")
    print(f"Validation labels shape: {labels.shape}")
    print(f"Generated samples shape: {samples.shape}")
    print(f"Generated sample labels shape: {sample_labels.shape}")
    print(f"Predicted labels for generated samples shape: {sample_predictions.shape}")


if __name__ == "__main__":
    main()
