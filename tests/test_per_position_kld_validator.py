from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from safetensors.torch import load_file, save_file
import torch

from evaluation.validate_per_position_kld import validate_per_position_kld


POSITIONS = 2047
METADATA = {
    "schema": "glm52-paired-position-kld-v1",
    "direction": "KL(ref||model)",
    "positions": str(POSITIONS),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(
    path: Path,
    values: torch.Tensor | None = None,
    *,
    metadata: dict[str, str] | None = None,
    tensors: dict[str, torch.Tensor] | None = None,
) -> tuple[str, float]:
    if values is None:
        values = torch.linspace(0.0, 0.02, POSITIONS, dtype=torch.float32)
    save_file(
        tensors if tensors is not None else {"kld_ref_to_model": values},
        path,
        metadata=METADATA if metadata is None else metadata,
    )
    return _sha256(path), float(values.to(torch.float64).mean().item())


def test_valid_result_and_cli_emit_machine_checkable_json(tmp_path: Path) -> None:
    path = tmp_path / "positions.safetensors"
    digest, mean = _write(path)

    report = validate_per_position_kld(
        path,
        expected_sha256=digest,
        expected_positions=POSITIONS,
        expected_mean_kld=mean + 1e-8,
    )
    assert report["valid"] is True
    assert report["sha256"] == digest
    assert report["positions"] == POSITIONS
    assert report["tensor"] == "kld_ref_to_model"
    assert report["tensor_mean_kld"] == pytest.approx(mean, abs=0.0)
    assert report["expected_mean_kld"] == pytest.approx(mean + 1e-8, abs=0.0)
    assert report["mean_absolute_error"] == pytest.approx(1e-8)

    script = Path(__file__).parents[1] / "evaluation/validate_per_position_kld.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(path),
            "--expected-sha256",
            digest,
            "--expected-positions",
            str(POSITIONS),
            "--expected-mean-kld",
            repr(mean),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cli_report = json.loads(completed.stdout)
    assert cli_report["valid"] is True
    assert cli_report["sha256"] == digest
    assert cli_report["tensor_mean_kld"] == mean


def test_rejects_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "positions.safetensors"
    _, mean = _write(path)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_per_position_kld(
            path,
            expected_sha256="0" * 64,
            expected_mean_kld=mean,
        )


def test_rejects_wrong_shape(tmp_path: Path) -> None:
    path = tmp_path / "positions.safetensors"
    values = torch.zeros(POSITIONS - 1, dtype=torch.float32)
    digest, mean = _write(path, values)

    with pytest.raises(ValueError, match=r"shape \(2047,\)"):
        validate_per_position_kld(
            path,
            expected_sha256=digest,
            expected_mean_kld=mean,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {**METADATA, "schema": "wrong-schema"},
        {**METADATA, "direction": "KL(model||ref)"},
        {**METADATA, "positions": "2046"},
        {**METADATA, "unexpected": "not-allowed"},
    ],
)
def test_rejects_nonexact_metadata(
    tmp_path: Path, metadata: dict[str, str]
) -> None:
    path = tmp_path / "positions.safetensors"
    digest, mean = _write(path, metadata=metadata)

    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_per_position_kld(
            path,
            expected_sha256=digest,
            expected_mean_kld=mean,
        )


@pytest.mark.parametrize(
    ("bad_value", "expected_error"),
    [
        (float("nan"), "non-finite"),
        (float("inf"), "non-finite"),
        (-1e-6, "negative KLD"),
    ],
)
def test_rejects_nonfinite_or_negative_values(
    tmp_path: Path, bad_value: float, expected_error: str
) -> None:
    path = tmp_path / "positions.safetensors"
    values = torch.zeros(POSITIONS, dtype=torch.float32)
    values[37] = bad_value
    digest, _ = _write(path, values)

    with pytest.raises(ValueError, match=expected_error):
        validate_per_position_kld(
            path,
            expected_sha256=digest,
            expected_mean_kld=0.0,
        )


def test_accepts_tiny_float32_negative_roundoff_without_clamping(
    tmp_path: Path,
) -> None:
    path = tmp_path / "positions.safetensors"
    values = torch.full((POSITIONS,), 0.0617, dtype=torch.float32)
    values[3] = -4.7e-8
    values[29] = -1e-9
    digest, mean = _write(path, values)

    report = validate_per_position_kld(
        path,
        expected_sha256=digest,
        expected_mean_kld=mean,
    )
    assert report["valid"] is True
    assert report["minimum_kld"] == pytest.approx(-4.7e-8)
    assert report["negative_roundoff_count"] == 2
    assert report["negative_roundoff_tolerance"] == 1e-7
    assert torch.equal(values, load_file(path)["kld_ref_to_model"])


def test_rejects_mean_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "positions.safetensors"
    digest, mean = _write(path)

    with pytest.raises(ValueError, match="mean KLD mismatch"):
        validate_per_position_kld(
            path,
            expected_sha256=digest,
            expected_mean_kld=mean + 1e-4,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_rejects_non_float32_tensor(tmp_path: Path, dtype: torch.dtype) -> None:
    path = tmp_path / "positions.safetensors"
    values = torch.zeros(POSITIONS, dtype=dtype)
    digest, _ = _write(path, values)

    with pytest.raises(ValueError, match="dtype float32"):
        validate_per_position_kld(
            path,
            expected_sha256=digest,
            expected_mean_kld=0.0,
        )


def test_rejects_extra_tensor(tmp_path: Path) -> None:
    path = tmp_path / "positions.safetensors"
    values = torch.zeros(POSITIONS, dtype=torch.float32)
    digest, mean = _write(
        path,
        values,
        tensors={"kld_ref_to_model": values, "extra": torch.zeros(1)},
    )

    with pytest.raises(ValueError, match="expected exactly tensor"):
        validate_per_position_kld(
            path,
            expected_sha256=digest,
            expected_mean_kld=mean,
        )
