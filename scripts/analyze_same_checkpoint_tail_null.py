#!/usr/bin/env python3
"""Calibrate KLD tail statistics from two five-boot sets of one checkpoint.

The input summaries must identify the same candidate, selected layers, runtime
image, prompt/reference, and five independently validated 2,047-position KLD
vectors.  The ten boots are pooled and every unique balanced 5-vs-5 partition
is enumerated.  For directional tail metrics both orientations are retained,
because either arbitrary group can appear worse under the same-checkpoint null.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


SUMMARY_SCHEMA = "glm52-fresh-sqg-candidate-kld-result-v2"
KLD_SCHEMA = "glm52-paired-position-kld-v1"
KLD_TENSOR = "kld_ref_to_model"
EXPECTED_RUNS = 5
EXPECTED_POSITIONS = 2047
MIN_ROUNDOFF = -2.0 * float(np.finfo(np.float32).eps)


def _float32_mean_closure_atol(values: np.ndarray, scalar: float) -> float:
    """Bound scalar-versus-stored-float32 mean closure without hiding drift.

    The emitted scalar is computed before the per-position evidence is stored as
    float32.  A bound scaled by the observed mean magnitude admits that endpoint
    conversion while remaining orders of magnitude below any experiment effect.
    """

    scale = max(abs(scalar), float(np.abs(values).mean(dtype=np.float64)))
    return max(5.0e-10, 2.0 * float(np.finfo(np.float32).eps) * scale)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a real regular file: {path}")
    return path.resolve()


def _load_vector(path: Path, expected_sha256: str) -> tuple[np.ndarray, dict[str, Any]]:
    path = _regular_file(path, "KLD tensor")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"KLD SHA256 differs: {path}")
    before = path.stat()
    with safe_open(path, framework="np") as handle:
        metadata = handle.metadata() or {}
        if set(handle.keys()) != {KLD_TENSOR}:
            raise ValueError(f"KLD tensor inventory differs: {path}")
        expected_metadata = {
            "schema": KLD_SCHEMA,
            "direction": "KL(ref||model)",
            "positions": str(EXPECTED_POSITIONS),
        }
        if metadata != expected_metadata:
            raise ValueError(f"KLD metadata differs: {path}")
        raw = np.asarray(handle.get_tensor(KLD_TENSOR))
        if raw.dtype != np.float32:
            raise ValueError(f"KLD tensor dtype differs: {path}: {raw.dtype}")
        values = raw.astype(np.float64)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError(f"KLD tensor changed while read: {path}")
    if values.shape != (EXPECTED_POSITIONS,) or not np.isfinite(values).all():
        raise ValueError(f"KLD tensor shape or finiteness differs: {path}")
    if float(values.min()) < MIN_ROUNDOFF:
        raise ValueError(f"KLD tensor contains non-roundoff negative value: {path}")
    return values, {
        "path": str(path),
        "sha256": actual_sha256,
        "mean": float(values.mean(dtype=np.float64)),
        "minimum": float(values.min()),
        "metadata": metadata,
    }


def _load_summary(path: Path) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    path = _regular_file(path, "summary")
    payload = json.loads(path.read_text())
    if payload.get("schema") != SUMMARY_SCHEMA:
        raise ValueError(f"summary schema differs: {path}")
    candidate = payload.get("candidate_result")
    if not isinstance(candidate, dict) or candidate.get("runs") != EXPECTED_RUNS:
        raise ValueError(f"summary must contain exactly five runs: {path}")
    outputs = candidate.get("paired_per_position_outputs")
    if not isinstance(outputs, list) or len(outputs) != EXPECTED_RUNS:
        raise ValueError(f"summary per-position inventory differs: {path}")
    vectors: list[np.ndarray] = []
    evidence: list[dict[str, Any]] = []
    for run, record in enumerate(outputs, start=1):
        if not isinstance(record, dict) or record.get("independently_validated") is not True:
            raise ValueError(f"run {run} lacks independent validation: {path}")
        if record.get("tensor") != KLD_TENSOR or record.get("positions") != EXPECTED_POSITIONS:
            raise ValueError(f"run {run} tensor contract differs: {path}")
        vector, item = _load_vector(Path(record["host_path"]), record["sha256"])
        scalar = float(candidate["values"][run - 1])
        vector_mean = float(vector.mean(dtype=np.float64))
        closure_atol = _float32_mean_closure_atol(vector, scalar)
        if not math.isclose(vector_mean, scalar, rel_tol=0.0, abs_tol=closure_atol):
            raise ValueError(f"run {run} vector/scalar closure failed: {path}")
        item["run"] = run
        item["summary_scalar_mean"] = scalar
        item["stored_vector_mean"] = vector_mean
        item["scalar_minus_vector_mean"] = scalar - vector_mean
        item["float32_mean_closure_atol"] = closure_atol
        vectors.append(vector)
        evidence.append(item)
    stacked = np.stack(vectors)
    stored_group_mean = float(stacked.mean(dtype=np.float64))
    summary_group_mean = float(candidate["mean_kld"])
    group_closure_atol = _float32_mean_closure_atol(
        stacked.reshape(-1), summary_group_mean
    )
    if not math.isclose(
        stored_group_mean,
        summary_group_mean,
        rel_tol=0.0,
        abs_tol=group_closure_atol,
    ):
        raise ValueError(f"summary mean closure failed: {path}")
    return payload, stacked, {
        "path": str(path),
        "sha256": _sha256(path),
        "summary_scalar_mean": summary_group_mean,
        "stored_vectors_mean": stored_group_mean,
        "scalar_minus_vectors_mean": summary_group_mean - stored_group_mean,
        "float32_mean_closure_atol": group_closure_atol,
        "vectors": evidence,
    }


def _identity(payload: dict[str, Any]) -> dict[str, Any]:
    construction = payload.get("candidate_construction") or {}
    runtime = payload.get("runtime") or {}
    overlay = runtime.get("runtime_overlay") or payload.get("runtime_overlay") or {}
    exact_args = runtime.get("exact_r33_extra_docker_args") or {}
    treatment = payload.get("selected_treatment_evidence") or {}
    primary = payload.get("primary_same_base_image_baseline") or {}
    regime = payload.get("regime") or {}
    return {
        "candidate": payload.get("candidate"),
        "selected_sqg_layers": payload.get("selected_sqg_layers"),
        "runtime_image_id": runtime.get("candidate_image_id")
        or payload.get("runtime_image_id"),
        "runtime_overlay_sha256": overlay.get("manifest_sha256")
        or overlay.get("sha256"),
        "exact_runtime_args_sha256": exact_args.get("sha256"),
        "reference_sha256": payload.get("reference_sha256"),
        "reference_token_ids_u32le_sha256": payload.get("reference_token_ids_u32le_sha256"),
        "run_seal_sha256": construction.get("run_seal_sha256"),
        "candidate_manifest_sha256": construction.get("candidate_manifest_sha256"),
        "selected_treatment_manifest_sha256": treatment.get("manifest_sha256"),
        "baseline_identity": {
            "summary": primary.get("summary"),
            "runs": primary.get("runs"),
            "mean_kld": primary.get("mean_kld"),
            "sample_sd_kld": primary.get("sample_sd_kld"),
        },
        "kv_cache_dtype": regime.get("kv_cache_dtype"),
        "decode_context_parallel_size": regime.get("decode_context_parallel_size"),
        "dcp_comm_backend": regime.get("dcp_comm_backend"),
        "dcp_kv_cache_interleave_size": regime.get("dcp_kv_cache_interleave_size"),
    }


def _upper_cvar(values: np.ndarray, fraction: float = 0.01) -> float:
    count = max(1, math.ceil(values.size * fraction))
    return float(np.partition(values, values.size - count)[-count:].mean(dtype=np.float64))


def _tail_indices(values: np.ndarray, fraction: float = 0.01) -> set[int]:
    count = max(1, math.ceil(values.size * fraction))
    return set(np.argpartition(values, values.size - count)[-count:].tolist())


def _directional_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Return left-minus-right metrics for two mean per-position vectors."""
    delta = left - right
    positive = np.maximum(delta, 0.0)
    left_tail = _tail_indices(left)
    right_tail = _tail_indices(right)
    union = left_tail | right_tail
    return {
        "mean_delta": float(delta.mean(dtype=np.float64)),
        "fraction_left_lower": float(np.mean(delta < 0.0)),
        "fraction_equal": float(np.mean(delta == 0.0)),
        "delta_p99": float(np.quantile(delta, 0.99)),
        "delta_p99_5": float(np.quantile(delta, 0.995)),
        "positive_delta_p99": float(np.quantile(positive, 0.99)),
        "positive_delta_cvar_1pct": _upper_cvar(positive),
        "positive_delta_mass": float(positive.sum(dtype=np.float64)),
        "negative_delta_mass": float(np.maximum(-delta, 0.0).sum(dtype=np.float64)),
        "maximum_regression": float(delta.max()),
        "left_absolute_kld_p99": float(np.quantile(left, 0.99)),
        "left_absolute_kld_cvar_1pct": _upper_cvar(left),
        "worst_1pct_overlap_count": float(len(left_tail & right_tail)),
        "worst_1pct_jaccard": float(len(left_tail & right_tail) / len(union)),
    }


def _quantile_table(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
    }


def analyze(summary_a: Path, summary_b: Path) -> dict[str, Any]:
    payload_a, vectors_a, evidence_a = _load_summary(summary_a)
    payload_b, vectors_b, evidence_b = _load_summary(summary_b)
    identity_a = _identity(payload_a)
    identity_b = _identity(payload_b)
    if identity_a != identity_b:
        differing = sorted(key for key in identity_a if identity_a[key] != identity_b[key])
        raise ValueError(f"same-checkpoint identity differs: {differing}")

    mean_a = vectors_a.mean(axis=0, dtype=np.float64)
    mean_b = vectors_b.mean(axis=0, dtype=np.float64)
    observed_ab = _directional_metrics(mean_a, mean_b)
    observed_ba = _directional_metrics(mean_b, mean_a)

    pooled = np.concatenate([vectors_a, vectors_b], axis=0)
    all_indices = set(range(2 * EXPECTED_RUNS))
    partition_metrics: list[dict[str, float]] = []
    # Requiring index zero in the left group removes complement duplicates.
    for tail in itertools.combinations(range(1, 2 * EXPECTED_RUNS), EXPECTED_RUNS - 1):
        left_indices = (0, *tail)
        right_indices = tuple(sorted(all_indices - set(left_indices)))
        left = pooled[list(left_indices)].mean(axis=0, dtype=np.float64)
        right = pooled[list(right_indices)].mean(axis=0, dtype=np.float64)
        partition_metrics.append(_directional_metrics(left, right))
        partition_metrics.append(_directional_metrics(right, left))
    if len(partition_metrics) != 252:
        raise AssertionError("balanced partition enumeration did not produce 252 orientations")

    keys = list(partition_metrics[0])
    envelopes = {
        key: _quantile_table([metrics[key] for metrics in partition_metrics])
        for key in keys
    }
    return {
        "schema": "glm52-same-checkpoint-tail-null-v1",
        "method": {
            "checkpoint_groups": "two independent five-boot sets of the exact same candidate",
            "positions": EXPECTED_POSITIONS,
            "pooled_boots": 10,
            "unique_unoriented_balanced_partitions": 126,
            "directional_partition_orientations": 252,
            "tail_fraction": 0.01,
            "purpose": "empirical null envelope for KLD tail gates; not a treatment effect",
            "roundoff_floor": MIN_ROUNDOFF,
        },
        "same_checkpoint_identity": identity_a,
        "input_a": evidence_a,
        "input_b": evidence_b,
        "observed_arbitrary_group_comparison": {
            "a_minus_b": observed_ab,
            "b_minus_a": observed_ba,
            "warning": "A/B direction is arbitrary because both groups use the same checkpoint",
        },
        "balanced_partition_null_envelopes": envelopes,
        "recommended_gate_calibration": {
            "directional_harm_metrics": {
                key: envelopes[key]["p95"]
                for key in (
                    "delta_p99",
                    "delta_p99_5",
                    "positive_delta_p99",
                    "positive_delta_cvar_1pct",
                    "positive_delta_mass",
                    "maximum_regression",
                )
            },
            "two_sided_mean_absolute_95": max(
                abs(envelopes["mean_delta"]["p05"]),
                abs(envelopes["mean_delta"]["p95"]),
            ),
            "win_fraction_null_95_interval": [
                envelopes["fraction_left_lower"]["p05"],
                envelopes["fraction_left_lower"]["p95"],
            ],
            "rule": "A future treatment fails a tail metric only when its directional harm exceeds the corresponding same-checkpoint p95 null envelope; retain raw preregistered values alongside the null-calibrated verdict.",
        },
        "limitations": [
            "The envelope describes fresh-boot variation for one checkpoint, prompt, and FP8-KV runtime regime.",
            "Balanced partitions reuse ten observed boots and are not independent experiments.",
            "This calibration does not increase the statistical power of a four-layer treatment comparison.",
        ],
    }


def _format_metric(value: float) -> str:
    return f"{value:.12g}"


def render_markdown(result: dict[str, Any]) -> str:
    observed = result["observed_arbitrary_group_comparison"]["a_minus_b"]
    gates = result["recommended_gate_calibration"]
    harm = gates["directional_harm_metrics"]
    method = result["method"]
    lines = [
        "# Same-checkpoint KLD tail null calibration",
        "",
        "## Result",
        "",
        (
            "This report pools two independently collected five-boot groups of the "
            "exact same sealed late-block SQG checkpoint. It enumerates all "
            f"{method['unique_unoriented_balanced_partitions']} unique balanced "
            "5-vs-5 partitions and both directions "
            f"({method['directional_partition_orientations']} directional comparisons)."
        ),
        "",
        (
            "The observed group-A minus group-B mean is "
            f"`{_format_metric(observed['mean_delta'])}`. Its direction is arbitrary: "
            "there is no treatment difference between the groups."
        ),
        "",
        "## Empirical 95% null gates",
        "",
        "| Metric | Fail only above |",
        "|---|---:|",
    ]
    for key, value in harm.items():
        lines.append(f"| `{key}` | `{_format_metric(value)}` |")
    interval = gates["win_fraction_null_95_interval"]
    lines.extend(
        [
            "",
            (
                "The two-sided 95% absolute mean-delta envelope is "
                f"`{_format_metric(gates['two_sided_mean_absolute_95'])}`. The "
                "same-checkpoint 95% interval for the fraction of positions where "
                f"the arbitrarily named left group is lower is "
                f"`[{_format_metric(interval[0])}, {_format_metric(interval[1])}]`."
            ),
            "",
            "The retained decision rule is:",
            "",
            f"> {gates['rule']}",
            "",
            "## Assumptions and limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.extend(
        [
            "",
            "This calibration changes the interpretation of the tail gate; it does "
            "not create power to resolve a small four-layer codebook effect.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_a", type=Path)
    parser.add_argument("summary_b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    markdown_output = args.output.with_suffix(".md")
    if markdown_output.exists() or markdown_output.is_symlink():
        raise SystemExit(f"refusing to overwrite output: {markdown_output}")
    result = analyze(args.summary_a, args.summary_b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    markdown_output.write_text(render_markdown(result))
    print(json.dumps({
        "output": str(args.output.resolve()),
        "markdown_output": str(markdown_output.resolve()),
        "sha256": _sha256(args.output),
        "markdown_sha256": _sha256(markdown_output),
        "mean_delta_a_minus_b": result["observed_arbitrary_group_comparison"]["a_minus_b"]["mean_delta"],
        "positive_delta_cvar_1pct_p95": result["recommended_gate_calibration"]["directional_harm_metrics"]["positive_delta_cvar_1pct"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
