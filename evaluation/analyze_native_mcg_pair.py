#!/usr/bin/env python3
"""Compare SQG and native-MCG KLD as paired per-position vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch


POSITIONS = 2047
TENSOR_NAME = "kld_ref_to_model"
EXPECTED_METADATA = {
    "schema": "glm52-paired-position-kld-v1",
    "direction": "KL(ref||model)",
    "positions": str(POSITIONS),
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"summary must not be a symlink: {path}")
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise ValueError(f"cannot stat summary: {path}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"summary must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read summary JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"summary root must be an object: {path}")
    return value


def _load_position_tensor(
    entry: dict[str, Any], expected_mean: float
) -> tuple[torch.Tensor, dict[str, Any]]:
    raw_path = entry.get("host_path")
    expected_sha256 = entry.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("per-position entry lacks host_path")
    if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(
        expected_sha256
    ):
        raise ValueError("per-position entry lacks an exact SHA-256")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError("per-position host_path must be absolute")
    if path.is_symlink():
        raise ValueError(f"per-position input must not be a symlink: {path}")
    try:
        before = path.stat()
    except OSError as exc:
        raise ValueError(f"cannot stat per-position input: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"per-position input must be a regular file: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"per-position SHA-256 differs: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if list(handle.keys()) != [TENSOR_NAME]:
            raise ValueError(f"per-position tensor inventory differs: {path}")
        if handle.metadata() != EXPECTED_METADATA:
            raise ValueError(f"per-position metadata differs: {path}")
        values = handle.get_tensor(TENSOR_NAME)
    if values.dtype != torch.float32 or tuple(values.shape) != (POSITIONS,):
        raise ValueError(f"per-position dtype or shape differs: {path}")
    if not bool(torch.isfinite(values).all().item()) or bool((values < 0).any().item()):
        raise ValueError(f"per-position KLD contains an invalid value: {path}")
    observed_mean = float(values.to(torch.float64).mean().item())
    tolerance = max(1e-8, 5e-7 * max(1.0, abs(expected_mean)))
    if not math.isclose(observed_mean, expected_mean, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"per-position mean differs from summary: {path}")
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"per-position input changed during validation: {path}")
    if _sha256_file(path) != actual_sha256:
        raise ValueError(f"per-position input changed during validation: {path}")
    return values.to(torch.float64), {
        "path": str(path.resolve()),
        "sha256": actual_sha256,
        "mean_kld": observed_mean,
    }


def _extract_arm(
    summary: dict[str, Any], *, arm: str, expected_runs: int
) -> tuple[torch.Tensor, list[dict[str, Any]], list[float]]:
    if arm == "candidate":
        if summary.get("schema") != "glm52-fresh-sqg-candidate-kld-result-v2":
            raise ValueError("candidate summary schema differs")
        result = summary.get("candidate_result")
    elif arm == "native_mcg":
        if summary.get("schema") != "glm52-native-mcg-control-kld-result-v1":
            raise ValueError("native-MCG summary schema differs")
        result = summary.get("control_result")
        if summary.get("selected_layers") != [6, 28, 52, 77]:
            raise ValueError("native-MCG selected layers differ")
        proof = summary.get("runtime_dispatch")
        if not isinstance(proof, dict) or proof.get("native_mcg_dispatch_proved") is not True:
            raise ValueError("native-MCG dispatch proof is absent")
    else:  # pragma: no cover - internal call contract
        raise AssertionError(arm)
    if not isinstance(result, dict):
        raise ValueError(f"{arm} result object is absent")
    if result.get("runs") != expected_runs:
        raise ValueError(f"{arm} run count differs")
    raw_values = result.get("values")
    entries = result.get("paired_per_position_outputs")
    if (
        not isinstance(raw_values, list)
        or len(raw_values) != expected_runs
        or not isinstance(entries, list)
        or len(entries) != expected_runs
    ):
        raise ValueError(f"{arm} run evidence census differs")
    means: list[float] = []
    tensors: list[torch.Tensor] = []
    evidence: list[dict[str, Any]] = []
    for raw_mean, entry in zip(raw_values, entries, strict=True):
        if isinstance(raw_mean, bool) or not isinstance(raw_mean, (int, float)):
            raise ValueError(f"{arm} mean KLD is not numeric")
        mean = float(raw_mean)
        if not math.isfinite(mean) or mean < 0.0 or not isinstance(entry, dict):
            raise ValueError(f"{arm} run evidence is invalid")
        tensor, record = _load_position_tensor(entry, mean)
        means.append(mean)
        tensors.append(tensor)
        evidence.append(record)
    return torch.stack(tensors), evidence, means


def _runtime_image_id(summary: dict[str, Any], arm: str) -> str | None:
    runtime = summary.get("runtime")
    if not isinstance(runtime, dict):
        return None
    if arm == "candidate":
        return runtime.get("candidate_image_id")
    return runtime.get("image_id")


def _overlay_sha256(summary: dict[str, Any]) -> str | None:
    runtime = summary.get("runtime")
    if not isinstance(runtime, dict):
        return None
    overlay = runtime.get("runtime_overlay")
    return overlay.get("manifest_sha256") if isinstance(overlay, dict) else None


def _circular_block_bootstrap(
    values: torch.Tensor,
    *,
    iterations: int,
    block_size: int,
    seed: int,
) -> tuple[float, float]:
    if iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")
    if block_size < 1 or block_size > values.numel():
        raise ValueError("bootstrap block size is invalid")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    positions = int(values.numel())
    blocks = math.ceil(positions / block_size)
    offsets = torch.arange(block_size, dtype=torch.int64)
    samples: list[torch.Tensor] = []
    remaining = iterations
    while remaining:
        batch = min(256, remaining)
        starts = torch.randint(
            positions, (batch, blocks), generator=generator, dtype=torch.int64
        )
        indices = (starts.unsqueeze(-1) + offsets) % positions
        indices = indices.reshape(batch, -1)[:, :positions]
        samples.append(values[indices].mean(dim=1))
        remaining -= batch
    bootstrap = torch.cat(samples)
    quantiles = torch.quantile(
        bootstrap, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    return float(quantiles[0].item()), float(quantiles[1].item())


def analyze_pair(
    candidate_summary_path: str | Path,
    control_summary_path: str | Path,
    *,
    tensor_output: str | Path,
    expected_runs: int = 5,
    bootstrap_iterations: int = 10_000,
    bootstrap_block_size: int = 32,
    bootstrap_seed: int = 20260810,
) -> dict[str, Any]:
    """Validate both arms and return a position-paired SQG-minus-MCG report."""

    if expected_runs != 5:
        raise ValueError("the preregistered comparison requires exactly five runs")
    candidate_path = Path(candidate_summary_path).resolve(strict=True)
    control_path = Path(control_summary_path).resolve(strict=True)
    if candidate_path == control_path:
        raise ValueError("candidate and control summaries must be distinct")
    candidate_summary_sha256 = _sha256_file(candidate_path)
    control_summary_sha256 = _sha256_file(control_path)
    candidate = _load_json(candidate_path)
    control = _load_json(control_path)
    if (
        _sha256_file(candidate_path) != candidate_summary_sha256
        or _sha256_file(control_path) != control_summary_sha256
    ):
        raise ValueError("candidate or control summary changed while being read")

    for key in ("reference_sha256", "reference_token_ids_u32le_sha256", "regime"):
        if candidate.get(key) != control.get(key):
            raise ValueError(f"candidate/control {key} differs")
    if _runtime_image_id(candidate, "candidate") != _runtime_image_id(
        control, "native_mcg"
    ):
        raise ValueError("candidate/control image ID differs")
    if _overlay_sha256(candidate) != _overlay_sha256(control):
        raise ValueError("candidate/control runtime overlay differs")

    candidate_tensor, candidate_evidence, candidate_means = _extract_arm(
        candidate, arm="candidate", expected_runs=expected_runs
    )
    control_tensor, control_evidence, control_means = _extract_arm(
        control, arm="native_mcg", expected_runs=expected_runs
    )
    by_run = candidate_tensor - control_tensor
    by_position = candidate_tensor.mean(dim=0) - control_tensor.mean(dim=0)
    run_mean_deltas = by_run.mean(dim=1)
    overall_delta = float(by_position.mean().item())
    scalar_delta = float(sum(candidate_means) / expected_runs) - float(
        sum(control_means) / expected_runs
    )
    if not math.isclose(overall_delta, scalar_delta, rel_tol=0.0, abs_tol=5e-8):
        raise ValueError("paired vectors do not close to the reported scalar means")

    lower, upper = _circular_block_bootstrap(
        by_position,
        iterations=bootstrap_iterations,
        block_size=bootstrap_block_size,
        seed=bootstrap_seed,
    )
    quantile_levels = torch.tensor(
        [0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64
    )
    quantiles = torch.quantile(by_position, quantile_levels)

    tensor_path = Path(tensor_output)
    if tensor_path.suffix != ".safetensors":
        raise ValueError("paired tensor output must end in .safetensors")
    if tensor_path.exists() or tensor_path.is_symlink():
        raise ValueError(f"paired tensor output already exists: {tensor_path}")
    parent = tensor_path.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("paired tensor output parent must be a real directory")
    tensor_path = parent / tensor_path.name
    partial = Path(f"{tensor_path}.partial")
    if partial.exists() or partial.is_symlink():
        raise ValueError(f"paired tensor partial already exists: {partial}")
    tensor_metadata = {
        "schema": "glm52-sqg-native-mcg-position-pair-v1",
        "direction": "candidate_sqg_minus_native_mcg; lower_is_better",
        "runs": str(expected_runs),
        "positions": str(POSITIONS),
        "candidate_summary_sha256": candidate_summary_sha256,
        "control_summary_sha256": control_summary_sha256,
    }
    save_file(
        {
            "candidate_minus_native_mcg_by_run": by_run.to(torch.float32),
            "candidate_minus_native_mcg_mean_by_position": by_position.to(
                torch.float32
            ),
        },
        partial,
        metadata=tensor_metadata,
    )
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(partial, tensor_path)
    tensor_sha256 = _sha256_file(tensor_path)
    if (
        _sha256_file(candidate_path) != candidate_summary_sha256
        or _sha256_file(control_path) != control_summary_sha256
    ):
        raise ValueError("candidate or control summary changed during analysis")

    direction = "lower" if overall_delta < 0 else "higher" if overall_delta > 0 else "equal"
    report: dict[str, Any] = {
        "schema": "glm52-sqg-native-mcg-paired-kld-analysis-v1",
        "candidate_summary": str(candidate_path),
        "candidate_summary_sha256": tensor_metadata["candidate_summary_sha256"],
        "control_summary": str(control_path),
        "control_summary_sha256": tensor_metadata["control_summary_sha256"],
        "reference_sha256": candidate["reference_sha256"],
        "reference_token_ids_u32le_sha256": candidate[
            "reference_token_ids_u32le_sha256"
        ],
        "regime": candidate["regime"],
        "selected_layers": [6, 28, 52, 77],
        "runs_per_arm": expected_runs,
        "positions": POSITIONS,
        "candidate_inputs": candidate_evidence,
        "native_mcg_inputs": control_evidence,
        "paired_tensor": {
            "path": str(tensor_path),
            "sha256": tensor_sha256,
            "metadata": tensor_metadata,
            "tensors": {
                "candidate_minus_native_mcg_by_run": [expected_runs, POSITIONS],
                "candidate_minus_native_mcg_mean_by_position": [POSITIONS],
            },
        },
        "paired_analysis": {
            "difference": "candidate_sqg_minus_native_mcg",
            "lower_is_better": True,
            "mean_delta": overall_delta,
            "mean_direction": direction,
            "run_pair_mean_deltas": [float(value) for value in run_mean_deltas],
            "candidate_lower_fraction": float((by_position < 0).double().mean().item()),
            "candidate_higher_fraction": float((by_position > 0).double().mean().item()),
            "position_delta_quantiles": {
                "p05": float(quantiles[0].item()),
                "p25": float(quantiles[1].item()),
                "p50": float(quantiles[2].item()),
                "p75": float(quantiles[3].item()),
                "p95": float(quantiles[4].item()),
            },
            "circular_block_bootstrap": {
                "iterations": bootstrap_iterations,
                "block_size": bootstrap_block_size,
                "seed": bootstrap_seed,
                "within_prompt_mean_delta_95pct_interval": [lower, upper],
            },
            "scalar_mean_closure_delta": scalar_delta,
        },
        "inference_limit": (
            "The paired vector comparison covers one fixed 2047-position prompt. "
            "Its block-bootstrap interval describes within-prompt position structure "
            "and runtime repeats, not generalization across texts or tasks."
        ),
    }
    report["analysis_id"] = _canonical_sha256(report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_summary", type=Path)
    parser.add_argument("control_summary", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-block-size", type=int, default=32)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    json_path = args.json_output
    if json_path.exists() or json_path.is_symlink():
        raise ValueError(f"paired JSON output already exists: {json_path}")
    parent = json_path.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("paired JSON output parent must be a real directory")
    json_path = parent / json_path.name
    report = analyze_pair(
        args.candidate_summary,
        args.control_summary,
        tensor_output=args.tensor_output,
        expected_runs=args.expected_runs,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_block_size=args.bootstrap_block_size,
        bootstrap_seed=args.bootstrap_seed,
    )
    partial = Path(f"{json_path}.partial")
    if partial.exists() or partial.is_symlink():
        raise ValueError(f"paired JSON partial already exists: {partial}")
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    with partial.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, json_path)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
