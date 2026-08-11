#!/usr/bin/env python3
"""Compare expert-local-H13 SQG with the prior shared-H13 SQG KLD arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch


RUNS = 5
POSITIONS = 2047
TENSOR_NAME = "kld_ref_to_model"
MIN_ROUNDOFF = -2.0 * float(torch.finfo(torch.float32).eps)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_METADATA = {
    "schema": "glm52-paired-position-kld-v1",
    "direction": "KL(ref||model)",
    "positions": str(POSITIONS),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def load_summary(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"summary must be a real regular file: {path}")
    digest = sha256_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "glm52-fresh-sqg-candidate-kld-result-v2":
        raise ValueError(f"candidate summary schema differs: {path}")
    result = value.get("candidate_result")
    if not isinstance(result, dict) or result.get("runs") != RUNS:
        raise ValueError(f"candidate summary does not contain five runs: {path}")
    return value, digest


def load_arm(summary: dict[str, Any]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    result = summary["candidate_result"]
    means = result.get("values")
    entries = result.get("paired_per_position_outputs")
    if not isinstance(means, list) or not isinstance(entries, list):
        raise ValueError("candidate run evidence is absent")
    if len(means) != RUNS or len(entries) != RUNS:
        raise ValueError("candidate run evidence census differs")
    tensors: list[torch.Tensor] = []
    evidence: list[dict[str, Any]] = []
    for mean, entry in zip(means, entries, strict=True):
        if not isinstance(entry, dict) or not isinstance(mean, (int, float)):
            raise ValueError("candidate run evidence is malformed")
        path = Path(entry.get("host_path", ""))
        expected_sha256 = entry.get("sha256")
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError(f"per-position tensor is unsafe or absent: {path}")
        if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(
            expected_sha256
        ):
            raise ValueError("per-position tensor lacks a valid SHA-256")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"per-position tensor SHA-256 differs: {path}")
        before = path.stat()
        with safe_open(path, framework="pt", device="cpu") as handle:
            if list(handle.keys()) != [TENSOR_NAME]:
                raise ValueError(f"per-position tensor inventory differs: {path}")
            if handle.metadata() != EXPECTED_METADATA:
                raise ValueError(f"per-position tensor metadata differs: {path}")
            tensor = handle.get_tensor(TENSOR_NAME)
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != (POSITIONS,):
            raise ValueError(f"per-position tensor dtype/shape differs: {path}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"per-position tensor contains a non-finite value: {path}")
        if float(tensor.min().item()) < MIN_ROUNDOFF:
            raise ValueError(f"per-position KLD exceeds roundoff bound: {path}")
        observed_mean = float(tensor.to(torch.float64).mean().item())
        if not math.isclose(observed_mean, float(mean), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"per-position mean does not close: {path}")
        after = path.stat()
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after or sha256_file(path) != actual_sha256:
            raise ValueError(f"per-position tensor changed while read: {path}")
        tensors.append(tensor.to(torch.float64))
        evidence.append(
            {
                "path": str(path.resolve()),
                "sha256": actual_sha256,
                "mean_kld": observed_mean,
                "minimum_kld": float(tensor.min().item()),
                "bounded_negative_roundoff_count": int((tensor < 0).sum().item()),
            }
        )
    return torch.stack(tensors), evidence


def circular_block_bootstrap(
    values: torch.Tensor, *, iterations: int, block_size: int, seed: int
) -> tuple[float, float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    offsets = torch.arange(block_size, dtype=torch.int64)
    blocks = math.ceil(POSITIONS / block_size)
    samples: list[torch.Tensor] = []
    for start in range(0, iterations, 256):
        batch = min(256, iterations - start)
        origins = torch.randint(
            POSITIONS, (batch, blocks), generator=generator, dtype=torch.int64
        )
        indices = (origins.unsqueeze(-1) + offsets) % POSITIONS
        indices = indices.reshape(batch, -1)[:, :POSITIONS]
        samples.append(values[indices].mean(dim=1))
    quantiles = torch.quantile(
        torch.cat(samples), torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    return float(quantiles[0].item()), float(quantiles[1].item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current_summary", type=Path)
    parser.add_argument("h13e_summary", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-block-size", type=int, default=32)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    args = parser.parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")
    if not 1 <= args.bootstrap_block_size <= POSITIONS:
        raise ValueError("bootstrap block size is invalid")

    current_path = args.current_summary.resolve(strict=True)
    h13e_path = args.h13e_summary.resolve(strict=True)
    if current_path == h13e_path:
        raise ValueError("the two summaries must be distinct")
    current, current_sha256 = load_summary(current_path)
    h13e, h13e_sha256 = load_summary(h13e_path)
    for key in ("reference_sha256", "reference_token_ids_u32le_sha256", "regime"):
        if current.get(key) != h13e.get(key):
            raise ValueError(f"summary {key} differs")
    if current.get("runtime", {}).get("candidate_image_id") != h13e.get(
        "runtime", {}
    ).get("candidate_image_id"):
        raise ValueError("runtime image differs")
    if current.get("runtime", {}).get("runtime_overlay", {}).get(
        "manifest_sha256"
    ) != h13e.get("runtime", {}).get("runtime_overlay", {}).get("manifest_sha256"):
        raise ValueError("runtime overlay differs")

    current_tensor, current_evidence = load_arm(current)
    h13e_tensor, h13e_evidence = load_arm(h13e)
    by_position = h13e_tensor.mean(dim=0) - current_tensor.mean(dim=0)
    scalar_current = float(current["candidate_result"]["mean_kld"])
    scalar_h13e = float(h13e["candidate_result"]["mean_kld"])
    scalar_delta = scalar_h13e - scalar_current
    if not math.isclose(
        float(by_position.mean().item()), scalar_delta, rel_tol=0.0, abs_tol=5e-8
    ):
        raise ValueError("per-position and scalar deltas do not close")

    current_sd = float(current["candidate_result"]["sample_sd_kld"])
    h13e_sd = float(h13e["candidate_result"]["sample_sd_kld"])
    repeat_se = math.sqrt((current_sd**2 + h13e_sd**2) / RUNS)
    welch_t = scalar_delta / repeat_se
    term_current = current_sd**2 / RUNS
    term_h13e = h13e_sd**2 / RUNS
    welch_df = (term_current + term_h13e) ** 2 / (
        term_current**2 / (RUNS - 1) + term_h13e**2 / (RUNS - 1)
    )
    block_lower, block_upper = circular_block_bootstrap(
        by_position,
        iterations=args.bootstrap_iterations,
        block_size=args.bootstrap_block_size,
        seed=args.bootstrap_seed,
    )
    quantiles = torch.quantile(
        by_position,
        torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64),
    )

    tensor_path = args.tensor_output.resolve()
    json_path = args.json_output.resolve()
    for path in (tensor_path, json_path):
        if path.exists() or path.is_symlink():
            raise ValueError(f"output already exists: {path}")
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise ValueError(f"output parent is unsafe or absent: {path.parent}")
    tensor_partial = Path(f"{tensor_path}.partial")
    save_file(
        {
            "current_sqg_kld_by_run": current_tensor.to(torch.float32),
            "h13e_sqg_kld_by_run": h13e_tensor.to(torch.float32),
            "h13e_minus_current_mean_by_position": by_position.to(torch.float32),
        },
        tensor_partial,
        metadata={
            "schema": "glm52-h13e-current-sqg-position-pair-v1",
            "direction": "h13e_minus_current_sqg; lower_is_better",
            "runs_per_arm": str(RUNS),
            "positions": str(POSITIONS),
            "current_summary_sha256": current_sha256,
            "h13e_summary_sha256": h13e_sha256,
        },
    )
    with tensor_partial.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tensor_partial, tensor_path)

    report: dict[str, Any] = {
        "schema": "glm52-h13e-current-sqg-paired-kld-analysis-v1",
        "direction": "h13e_minus_current_sqg; lower_is_better",
        "current_summary": str(current_path),
        "current_summary_sha256": current_sha256,
        "h13e_summary": str(h13e_path),
        "h13e_summary_sha256": h13e_sha256,
        "reference_sha256": current["reference_sha256"],
        "reference_token_ids_u32le_sha256": current[
            "reference_token_ids_u32le_sha256"
        ],
        "regime": current["regime"],
        "runs_per_arm": RUNS,
        "positions": POSITIONS,
        "current_inputs": current_evidence,
        "h13e_inputs": h13e_evidence,
        "paired_tensor": {
            "path": str(tensor_path),
            "sha256": sha256_file(tensor_path),
        },
        "scalar": {
            "current_mean_kld": scalar_current,
            "current_sample_sd_kld": current_sd,
            "h13e_mean_kld": scalar_h13e,
            "h13e_sample_sd_kld": h13e_sd,
            "h13e_minus_current": scalar_delta,
            "relative_delta": scalar_delta / scalar_current,
            "repeat_noise_standard_error": repeat_se,
            "welch_t": welch_t,
            "welch_degrees_freedom": welch_df,
            "repeat_noise_direction": "inconclusive"
            if abs(welch_t) < 2.776
            else ("higher" if scalar_delta > 0 else "lower"),
        },
        "per_position": {
            "h13e_lower_fraction": float((by_position < 0).double().mean().item()),
            "h13e_higher_fraction": float((by_position > 0).double().mean().item()),
            "delta_quantiles": {
                "p05": float(quantiles[0].item()),
                "p25": float(quantiles[1].item()),
                "p50": float(quantiles[2].item()),
                "p75": float(quantiles[3].item()),
                "p95": float(quantiles[4].item()),
            },
            "circular_block_bootstrap": {
                "iterations": args.bootstrap_iterations,
                "block_size": args.bootstrap_block_size,
                "seed": args.bootstrap_seed,
                "within_prompt_mean_delta_95pct_interval": [
                    block_lower,
                    block_upper,
                ],
            },
        },
        "inference_limit": (
            "Both arms use five fresh boots of one fixed 2047-position prompt. "
            "Welch statistics describe repeat/runtime variation; the circular-block "
            "interval describes within-prompt position structure. Neither estimates "
            "generalization across texts or tasks."
        ),
    }
    report["analysis_id"] = canonical_sha256(report)
    json_partial = Path(f"{json_path}.partial")
    with json_partial.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(json_partial, json_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
