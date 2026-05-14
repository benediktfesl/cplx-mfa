# cplx-mfa

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Package](https://img.shields.io/badge/package-PyPI-informational.svg)](https://pypi.org/project/cplx-mfa/)

Complex-valued mixture of factor analyzers with a scikit-learn-like API.

`cplx-mfa` provides an estimator for fitting mixture of factor analyzers (MFA) models to complex-valued data using expectation-maximization (EM). The model uses circularly symmetric complex Gaussian components with low-rank covariance structure, making it useful for high-dimensional signal processing problems where full covariance Gaussian mixtures can be too expensive or statistically inefficient.

Parts of the implementation are derived from the original [`mofa`](https://pypi.org/project/mofa/) package and extended with complex-valued modeling support, modern packaging, improved naming, validation, reproducible initialization, and functional tests.

## ✨ Highlights

- Complex-valued mixture of factor analyzers for data in complex vector spaces
- Low-rank covariance structure via component-wise factor loading matrices
- Circularly symmetric complex Gaussian components
- scikit-learn-like estimator API with `fit`, `predict`, `predict_proba`, and `sample`
- Optional isotropic PPCA-style noise model
- Optional shared diagonal noise variances across components
- Optional zero-mean component constraint
- Sampling from fitted complex-valued MFA models
- Fitted parameters exposed with trailing-underscore names
- Modern Python packaging with `pyproject.toml`, `uv`, `pytest`, and `ruff`

## 📌 Citation

If you use `cplx-mfa` in academic work, please cite the package directly:

```bibtex
@software{fesl_cplx_mfa,
  author = {Fesl, Benedikt},
  title = {{cplx-mfa}: Complex-valued mixture of factor analyzers},
  year = {2026},
  url = {https://github.com/benediktfesl/cplx-mfa},
  version = {0.1.1}
}
```

Plain-text citation:

> B. Fesl, `cplx-mfa`: Complex-valued mixture of factor analyzers, version 0.1.1. Available: https://github.com/benediktfesl/cplx-mfa

## 📦 Installation

Install from PyPI:

```bash
pip install cplx-mfa
```

or with `uv`:

```bash
uv add cplx-mfa
```

For development, clone the repository and install the development environment:

```bash
git clone https://github.com/benediktfesl/cplx-mfa.git
cd cplx-mfa
uv sync --group dev
```

## 🚀 Quick Start

```python
import numpy as np

from cplx_mfa import ComplexMFA

rng = np.random.default_rng(0)

X = (
    rng.normal(size=(1_000, 8))
    + 1j * rng.normal(size=(1_000, 8))
) / np.sqrt(2.0)

model = ComplexMFA(
    n_components=4,
    latent_dim=2,
    random_state=0,
    max_iter=100,
    verbose=False,
)

model.fit(X)

labels = model.predict(X)
responsibilities = model.predict_proba(X)

samples, component_labels = model.sample(
    n_samples=100,
    rng=np.random.default_rng(1),
)
```

The estimator follows the usual pattern: model configuration is passed to the constructor, and `fit(X)` receives the data.

## 🧩 Model Structure

A mixture of factor analyzers represents each mixture component with a low-rank covariance structure:

```text
covariance = loadings @ loadingsᴴ + diagonal_noise
```

This is useful when the feature dimension is large but the dominant component-wise variation is approximately low-dimensional.

The fitted attributes are:

| Attribute | Description |
|---|---|
| `weights_` | Mixture weights of shape `(n_components,)`. |
| `means_` | Component means of shape `(n_components, n_features)`. |
| `loadings_` | Factor loading matrices of shape `(n_components, n_features, latent_dim)`. |
| `covariances_` | Full implied covariance matrices of shape `(n_components, n_features, n_features)`. |
| `precisions_` | Inverse covariance matrices of shape `(n_components, n_features, n_features)`. |
| `noise_variances_` | Diagonal noise variances of shape `(n_components, n_features)`. |
| `lower_bound_history_` | EM lower-bound values collected during fitting. |

## 🧠 Estimator API

The main class is:

```python
from cplx_mfa import ComplexMFA
```

Core methods:

| Method | Description |
|---|---|
| `fit(X)` | Fit the complex-valued MFA model. |
| `predict(X)` | Predict the most likely component for each sample. |
| `predict_proba(X)` | Return posterior component probabilities. |
| `sample(n_samples=1, rng=None)` | Draw samples from the fitted mixture model. |

Constructor parameters:

| Parameter | Description |
|---|---|
| `n_components` | Number of mixture components. |
| `latent_dim` | Latent dimensionality of each factor analyzer. |
| `ppca` | If `True`, use one isotropic noise variance per component. |
| `lock_psis` | If `True`, use shared diagonal noise variances across components. |
| `zero_mean` | If `True`, enforce zero component means during initialization and EM. |
| `rs_clip` | Lower clipping value for responsibilities during EM. |
| `max_condition_number` | Scaling factor used for random loading initialization. |
| `max_iter` | Maximum number of EM iterations. |
| `tol` | Relative convergence tolerance. |
| `random_state` | Integer seed or NumPy random generator used for initialization. |
| `verbose` | If `True`, print EM progress. |

## 🔁 Sampling

Samples are generated component-wise and returned grouped by component. The returned labels follow the same order.

```python
samples, labels = model.sample(
    n_samples=100,
    rng=np.random.default_rng(1),
)
```

This means `labels` is sorted by component group rather than shuffled randomly. This behavior is intentional and documented so that generated samples can be inspected component by component.

## 🔬 PPCA-Style Components

Set `ppca=True` to constrain each component to use an isotropic diagonal noise variance:

```python
model = ComplexMFA(
    n_components=4,
    latent_dim=2,
    ppca=True,
    random_state=0,
)

model.fit(X)
```

This gives a probabilistic PCA-style covariance structure per mixture component.

## 🔒 Shared Noise Variances

Set `lock_psis=True` to enforce shared diagonal noise variances across all mixture components:

```python
model = ComplexMFA(
    n_components=4,
    latent_dim=2,
    lock_psis=True,
    random_state=0,
)

model.fit(X)
```

This can be useful when all mixture components are expected to share the same residual noise floor.

## 🎯 Zero-Mean Components

Set `zero_mean=True` to enforce zero means for all mixture components:

```python
model = ComplexMFA(
    n_components=4,
    latent_dim=2,
    zero_mean=True,
    random_state=0,
)

model.fit(X)
```

This is useful when the data is known or assumed to be zero-mean and only the component covariance structure should be learned.

## 📚 Research Background

This implementation was developed in the context of complex-valued generative modeling for wireless channel estimation and related signal processing applications.

The results of the following work are, in parts, based on the complex-valued MFA implementation:

- B. Fesl, N. Turan, and W. Utschick, “Low-Rank Structured MMSE Channel Estimation with Mixtures of Factor Analyzers,” *57th Asilomar Conference on Signals, Systems, and Computers*, 2023.  
  [[IEEE](https://ieeexplore.ieee.org/document/10477088)] [[arXiv](https://arxiv.org/abs/2304.14809)]

## 🧪 Development

Install the development environment with `uv`:

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

Format code:

```bash
uv run ruff format .
```

Run the example:

```bash
uv run python examples/cplx_mfa_example.py
```

Build the package:

```bash
uv run python -m build
```

## ✅ Test Coverage

The test suite covers:

- package imports
- fitting and fitted attribute shapes
- prediction and responsibility normalization
- sampling behavior
- grouped sample labels
- validation behavior
- unfitted-estimator behavior
- reproducibility with fixed `random_state`
- EM lower-bound history
- utility functions for complex-valued data handling
- example execution

## 📄 License

This project is licensed under the [BSD 3-Clause License](LICENSE).

The implementation contains code derived from the original `mofa` package, which is MIT-licensed. Attribution and provenance are retained in [NOTICE](NOTICE).
