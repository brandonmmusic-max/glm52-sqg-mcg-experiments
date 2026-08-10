#!/usr/bin/env python3
"""Validate a sealed per-position ``KL(ref||model)`` safetensors result.

The scalar KLD printed by the paired evaluator is accumulated from float32
chunk reductions, while this validator recomputes a float64 mean from the
persisted float32 positions.  The acceptance tolerance therefore permits only
the small reduction-order discrepancy expected from that representation:
``max(1e-8, 5e-7 * max(1, abs(expected_mean)))``.

Per-position KL is also persisted as float32 after a large-vocabulary
reduction. Values down to ``-1e-7`` are accepted as cancellation roundoff and
reported without modifying the tensor; any value below that remains fatal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys
from typing import Any

from safetensors import safe_open
import torch


TENSOR_NAME = "kld_ref_to_model"
EXPECTED_SCHEMA = "glm52-paired-position-kld-v1"
EXPECTED_DIRECTION = "KL(ref||model)"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MEAN_ABSOLUTE_TOLERANCE = 1e-8
MEAN_RELATIVE_TOLERANCE = 5e-7
NEGATIVE_ROUNDOFF_TOLERANCE = 1e-7


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_tolerance(expected_mean_kld: float) -> float:
    return max(
        MEAN_ABSOLUTE_TOLERANCE,
        MEAN_RELATIVE_TOLERANCE * max(1.0, abs(expected_mean_kld)),
    )


def validate_per_position_kld(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_positions: int = 2047,
    expected_mean_kld: float,
) -> dict[str, Any]:
    """Return a validation report, or raise ``ValueError`` fail closed."""

    result_path = Path(path)
    if not SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("expected SHA-256 must be exactly 64 lowercase hex digits")
    if (
        isinstance(expected_positions, bool)
        or not isinstance(expected_positions, int)
        or expected_positions <= 0
    ):
        raise ValueError("expected positions must be a positive integer")
    if not math.isfinite(expected_mean_kld) or expected_mean_kld < 0.0:
        raise ValueError("expected mean KLD must be finite and nonnegative")
    if result_path.is_symlink():
        raise ValueError(f"per-position result must not be a symlink: {result_path}")
    try:
        before = result_path.stat()
    except OSError as exc:
        raise ValueError(f"cannot stat per-position result: {result_path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"per-position result is not a regular file: {result_path}")

    actual_sha256 = _sha256_file(result_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "per-position SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    expected_metadata = {
        "schema": EXPECTED_SCHEMA,
        "direction": EXPECTED_DIRECTION,
        "positions": str(expected_positions),
    }
    try:
        with safe_open(result_path, framework="pt", device="cpu") as handle:
            tensor_names = list(handle.keys())
            if tensor_names != [TENSOR_NAME]:
                raise ValueError(
                    f"expected exactly tensor {TENSOR_NAME!r}, got {tensor_names!r}"
                )
            metadata = handle.metadata()
            if metadata != expected_metadata:
                raise ValueError(
                    "per-position metadata mismatch: "
                    f"expected {expected_metadata!r}, got {metadata!r}"
                )
            values = handle.get_tensor(TENSOR_NAME)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"invalid safetensors result: {result_path}") from exc

    if values.dtype != torch.float32:
        raise ValueError(
            f"{TENSOR_NAME} must have dtype float32, got {values.dtype}"
        )
    if tuple(values.shape) != (expected_positions,):
        raise ValueError(
            f"{TENSOR_NAME} must have shape ({expected_positions},), "
            f"got {tuple(values.shape)}"
        )
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError(f"{TENSOR_NAME} contains a non-finite value")
    minimum_kld = float(values.min().item())
    negative_count = int((values < 0).sum().item())
    if minimum_kld < -NEGATIVE_ROUNDOFF_TOLERANCE:
        raise ValueError(
            f"{TENSOR_NAME} contains a negative KLD value below roundoff "
            f"tolerance: {minimum_kld:.17g} < "
            f"{-NEGATIVE_ROUNDOFF_TOLERANCE:.17g}"
        )

    tensor_mean_kld = float(values.to(dtype=torch.float64).mean().item())
    mean_absolute_error = abs(tensor_mean_kld - expected_mean_kld)
    mean_tolerance = _mean_tolerance(expected_mean_kld)
    if mean_absolute_error > mean_tolerance:
        raise ValueError(
            "per-position mean KLD mismatch: "
            f"expected {expected_mean_kld:.17g}, got {tensor_mean_kld:.17g}, "
            f"absolute error {mean_absolute_error:.17g} exceeds "
            f"tolerance {mean_tolerance:.17g}"
        )

    after = result_path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("per-position result changed during validation")
    closing_sha256 = _sha256_file(result_path)
    if closing_sha256 != actual_sha256:
        raise ValueError("per-position result changed during validation")

    return {
        "valid": True,
        "path": str(result_path.resolve()),
        "sha256": actual_sha256,
        "tensor": TENSOR_NAME,
        "positions": expected_positions,
        "tensor_mean_kld": tensor_mean_kld,
        "expected_mean_kld": expected_mean_kld,
        "mean_absolute_error": mean_absolute_error,
        "mean_tolerance": mean_tolerance,
        "minimum_kld": minimum_kld,
        "negative_roundoff_count": negative_count,
        "negative_roundoff_tolerance": NEGATIVE_ROUNDOFF_TOLERANCE,
        "metadata": expected_metadata,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-positions", type=int, default=2047)
    parser.add_argument("--expected-mean-kld", type=float, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = validate_per_position_kld(
        args.path,
        expected_sha256=args.expected_sha256,
        expected_positions=args.expected_positions,
        expected_mean_kld=args.expected_mean_kld,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
