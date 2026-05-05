import numpy as np
import pytest

from cplx_mfa import ComplexMFA


@pytest.mark.parametrize(
    "method_name",
    ["predict", "predict_proba", "sample"],
)
def test_public_methods_raise_before_fit(method_name: str) -> None:
    data = np.ones((10, 2), dtype=complex)
    model = ComplexMFA(n_components=2, latent_dim=1, verbose=False)

    method = getattr(model, method_name)

    with pytest.raises(RuntimeError, match="must be fitted"):
        if method_name == "sample":
            method()
        else:
            method(data)
