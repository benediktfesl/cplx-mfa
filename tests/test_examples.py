import subprocess
import sys
from pathlib import Path


def test_complex_mfa_example_runs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_path = repo_root / "examples" / "cplx_mfa_example.py"

    result = subprocess.run(
        [sys.executable, str(example_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Training completed" in result.stdout
    assert "Mixture weight sum" in result.stdout
