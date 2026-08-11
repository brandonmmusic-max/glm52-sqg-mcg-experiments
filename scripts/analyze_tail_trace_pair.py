#!/usr/bin/env python3
"""Relate paired final-logit KLD tails to MoE routing and residual traces.

The two arms must have been traced with ``SQG_TAIL_TRACE=1`` on the same token
sequence.  Rows are joined by absolute token position, never by file order.
Final-logit KLD entry ``i`` is aligned with the layer trace at input position
``i`` because that hidden state produces the logits for token ``i + 1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
import torch


TRACE_SCHEMA = "glm52-sqg-tail-trace-v1"
KLD_SCHEMA = "glm52-paired-position-kld-v1"
KLD_TENSOR = "kld_ref_to_model"
TRACE_KEYS = {
    "positions",
    "hidden",
    "router_logits",
    "topk_weights",
    "topk_ids",
    "moe_output",
    "residual_before_moe",
    "residual_stream_after_moe",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_kld(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"KLD input must be a real regular file: {path}")
    before = path.stat()
    with safe_open(path, framework="np") as handle:
        metadata = handle.metadata() or {}
        keys = set(handle.keys())
        if keys != {KLD_TENSOR}:
            raise ValueError(f"KLD tensor inventory differs: {path}: {keys}")
        if metadata.get("schema") != KLD_SCHEMA:
            raise ValueError(f"KLD schema differs: {path}")
        values = np.asarray(handle.get_tensor(KLD_TENSOR), dtype=np.float64)
    after = path.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError(f"KLD input changed while read: {path}")
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"KLD vector is malformed: {path}")
    if metadata.get("positions") != str(values.size):
        raise ValueError(f"KLD metadata position count differs: {path}")
    return values, {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "positions": int(values.size),
        "mean": float(values.mean(dtype=np.float64)),
        "metadata": metadata,
    }


def _load_trace_file(path: Path) -> tuple[int, dict[str, np.ndarray], dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"trace input must be a real regular file: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("schema") != TRACE_SCHEMA:
            raise ValueError(f"trace schema differs: {path}")
        keys = set(handle.keys())
        if keys != TRACE_KEYS:
            raise ValueError(f"trace tensor inventory differs: {path}: {keys}")
        tensors = {}
        for name in keys:
            tensor = handle.get_tensor(name)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            tensors[name] = tensor.numpy()
    try:
        layer = int(metadata["layer"])
    except (KeyError, ValueError) as error:
        raise ValueError(f"trace layer metadata is malformed: {path}") from error
    rows = int(tensors["positions"].shape[0])
    if rows == 0 or any(int(value.shape[0]) != rows for value in tensors.values()):
        raise ValueError(f"trace rows do not align: {path}")
    return layer, tensors, metadata


def _load_trace_root(root: Path) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, Any]]]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"trace root is absent or unsafe: {root}")
    paths = sorted(root.glob("layer-*-rank-*-call-*.safetensors"))
    if not paths:
        raise ValueError(f"trace root contains no trace tensors: {root}")
    fragments: dict[tuple[int, int], list[dict[str, Any]]] = {}
    evidence: list[dict[str, Any]] = []
    for path in paths:
        layer, tensors, metadata = _load_trace_file(path)
        keep = np.asarray(tensors["positions"] >= 0)
        if not keep.any():
            continue
        invocation = int(metadata["invocation"])
        rank = int(metadata["rank"])
        values = {name: np.asarray(value[keep]) for name, value in tensors.items()}
        fragments.setdefault((layer, invocation), []).append(
            {"rank": rank, "values": values, "path": path}
        )
        evidence.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "layer": layer,
                "rank": rank,
                "invocation": invocation,
                "rows": int(keep.sum()),
                "unique_positions": int(np.unique(values["positions"]).size),
                "analysis_selected": False,
            }
        )
    merged: dict[int, dict[str, np.ndarray]] = {}
    layers = sorted({layer for layer, _ in fragments})
    for layer in layers:
        candidates: list[tuple[int, int, list[dict[str, Any]]]] = []
        for (candidate_layer, invocation), parts in fragments.items():
            if candidate_layer != layer:
                continue
            reference_positions = parts[0]["values"]["positions"].astype(
                np.int64, copy=False
            )
            if not np.array_equal(
                reference_positions,
                np.arange(reference_positions.size, dtype=np.int64),
            ):
                continue
            if any(
                not np.array_equal(
                    part["values"]["positions"].astype(np.int64, copy=False),
                    reference_positions,
                )
                for part in parts[1:]
            ):
                continue
            candidates.append((reference_positions.size, invocation, parts))
        if not candidates:
            raise ValueError(f"layer {layer} has no complete ordered trace invocation")
        _, invocation, parts = max(candidates, key=lambda item: (item[0], item[1]))
        parts.sort(key=lambda part: part["rank"])
        if [part["rank"] for part in parts] != [0, 1, 2, 3]:
            raise ValueError(f"layer {layer} selected invocation lacks exact DCP4 ranks")
        reference = parts[0]["values"]
        for part in parts[1:]:
            for name in TRACE_KEYS:
                if not np.array_equal(reference[name], part["values"][name]):
                    raise ValueError(
                        f"layer {layer} invocation {invocation} DCP rank replicas "
                        f"differ for {name}"
                    )
        merged[layer] = reference
        selected_paths = {str(part["path"].resolve()) for part in parts}
        for item in evidence:
            if item["path"] in selected_paths:
                item["analysis_selected"] = True
                item["dcp_rank_replica_exact"] = True
    return merged, evidence


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "p99_5": float(np.quantile(values, 0.995)),
        "p99_9": float(np.quantile(values, 0.999)),
        "max": float(values.max()),
    }


def _upper_cvar(values: np.ndarray, fraction: float = 0.01) -> float:
    count = max(1, math.ceil(values.size * fraction))
    return float(np.partition(values, values.size - count)[-count:].mean(dtype=np.float64))


def _row_relative_delta(candidate: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    delta = candidate.astype(np.float64) - baseline.astype(np.float64)
    denominator = np.maximum(
        np.square(baseline.astype(np.float64)).sum(axis=1),
        np.finfo(np.float64).tiny,
    )
    return np.square(delta).sum(axis=1) / denominator


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks, including ties, without scipy."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    begin = 0
    while begin < values.size:
        end = begin + 1
        while end < values.size and sorted_values[end] == sorted_values[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + end - 1)
        begin = end
    return ranks


def _correlation(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
        x = x.astype(np.float64) - x.mean(dtype=np.float64)
        y = y.astype(np.float64) - y.mean(dtype=np.float64)
        denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
        return None if denominator == 0.0 else float(np.dot(x, y) / denominator)

    return {
        "pearson": pearson(a, b),
        "spearman": pearson(_rankdata(a), _rankdata(b)),
    }


def _routing_metrics(
    baseline_ids: np.ndarray,
    candidate_ids: np.ndarray,
    baseline_weights: np.ndarray,
    candidate_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    if baseline_ids.shape != candidate_ids.shape or baseline_ids.ndim != 2:
        raise ValueError("top-k ID shapes differ")
    rows, topk = baseline_ids.shape
    overlap = np.empty(rows, dtype=np.int16)
    weight_l1 = np.empty(rows, dtype=np.float64)
    exact_order = np.all(baseline_ids == candidate_ids, axis=1)
    exact_set = np.empty(rows, dtype=np.bool_)
    for row in range(rows):
        b_ids = baseline_ids[row].astype(np.int64, copy=False)
        c_ids = candidate_ids[row].astype(np.int64, copy=False)
        b_map = {int(expert): float(weight) for expert, weight in zip(b_ids, baseline_weights[row], strict=True)}
        c_map = {int(expert): float(weight) for expert, weight in zip(c_ids, candidate_weights[row], strict=True)}
        shared = set(b_map).intersection(c_map)
        overlap[row] = len(shared)
        exact_set[row] = len(shared) == topk
        weight_l1[row] = sum(
            abs(b_map.get(expert, 0.0) - c_map.get(expert, 0.0))
            for expert in set(b_map).union(c_map)
        )
    return {
        "overlap": overlap,
        "weight_l1": weight_l1,
        "exact_order": exact_order,
        "exact_set": exact_set,
    }


def _render(report: dict[str, Any]) -> str:
    kld = report["kld"]
    lines = [
        "# Paired KLD-tail, routing, and residual trace",
        "",
        f"Candidate improved **{kld['improved_fraction']:.3%}** of positions and worsened "
        f"**{kld['worsened_fraction']:.3%}**.",
        "",
        f"Mean candidate-minus-baseline KLD: `{kld['mean_delta']:.9e}`; "
        f"positive-tail CVaR 1%: `{kld['positive_delta_cvar_1pct']:.9e}`.",
        "",
        "| Layer | Route sets changed | Mean route L1 | Hidden rel. delta | MoE rel. delta | Residual-after rel. delta | Tail residual-after rel. delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, metric in report["layers"].items():
        lines.append(
            f"| {layer} | {metric['route_set_changed_fraction']:.3%} | "
            f"{metric['route_weight_l1_mean']:.6e} | "
            f"{metric['hidden_relative_delta']['mean']:.6e} | "
            f"{metric['moe_output_relative_delta']['mean']:.6e} | "
            f"{metric['residual_after_relative_delta']['mean']:.6e} | "
            f"{metric['tail']['residual_after_relative_delta_mean']:.6e} |"
        )
    lines.extend(["", "## Worst positive KLD positions", "", "| Position | Baseline KLD | Candidate KLD | Delta |", "|---:|---:|---:|---:|"])
    for item in report["worst_positive_positions"]:
        lines.append(
            f"| {item['position']} | {item['baseline_kld']:.9e} | "
            f"{item['candidate_kld']:.9e} | {item['delta']:.9e} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--baseline-kld", type=Path, required=True)
    parser.add_argument("--candidate-kld", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worst-positions", type=int, default=20)
    args = parser.parse_args()
    if args.worst_positions <= 0:
        raise ValueError("worst position count must be positive")

    baseline_kld, baseline_kld_evidence = _load_kld(args.baseline_kld.resolve())
    candidate_kld, candidate_kld_evidence = _load_kld(args.candidate_kld.resolve())
    if baseline_kld.shape != candidate_kld.shape:
        raise ValueError("paired KLD vectors differ in length")
    baseline, baseline_evidence = _load_trace_root(args.baseline_trace.resolve())
    candidate, candidate_evidence = _load_trace_root(args.candidate_trace.resolve())
    if set(baseline) != set(candidate):
        raise ValueError("paired trace layer inventories differ")

    kld_positions = np.arange(baseline_kld.size, dtype=np.int64)
    delta_kld = candidate_kld - baseline_kld
    positive_tail_count = max(1, math.ceil(delta_kld.size * 0.01))
    tail_indices = np.argpartition(delta_kld, delta_kld.size - positive_tail_count)[-positive_tail_count:]
    tail_mask = np.zeros(delta_kld.size, dtype=np.bool_)
    tail_mask[tail_indices] = True
    layer_metrics: dict[str, Any] = {}
    row_arrays: dict[str, np.ndarray] = {
        "positions": kld_positions.astype(np.int32),
        "baseline_kld": baseline_kld.astype(np.float32),
        "candidate_kld": candidate_kld.astype(np.float32),
        "candidate_minus_baseline_kld": delta_kld.astype(np.float32),
    }

    for layer in sorted(baseline):
        b = baseline[layer]
        c = candidate[layer]
        b_positions = b["positions"].astype(np.int64, copy=False)
        c_positions = c["positions"].astype(np.int64, copy=False)
        if not np.array_equal(b_positions, c_positions):
            raise ValueError(f"layer {layer} paired trace positions differ")
        indices = np.searchsorted(b_positions, kld_positions)
        if np.any(indices >= b_positions.size) or not np.array_equal(b_positions[indices], kld_positions):
            raise ValueError(f"layer {layer} trace does not cover every KLD position")
        b = {name: value[indices] for name, value in b.items()}
        c = {name: value[indices] for name, value in c.items()}
        routing = _routing_metrics(b["topk_ids"], c["topk_ids"], b["topk_weights"], c["topk_weights"])
        hidden_relative = _row_relative_delta(c["hidden"], b["hidden"])
        moe_relative = _row_relative_delta(c["moe_output"], b["moe_output"])
        residual_before_relative = _row_relative_delta(c["residual_before_moe"], b["residual_before_moe"])
        residual_after_relative = _row_relative_delta(c["residual_stream_after_moe"], b["residual_stream_after_moe"])
        prefix = f"layer_{layer:03d}"
        row_arrays.update(
            {
                f"{prefix}__route_overlap": routing["overlap"],
                f"{prefix}__route_weight_l1": routing["weight_l1"].astype(np.float32),
                f"{prefix}__hidden_relative_delta": hidden_relative.astype(np.float32),
                f"{prefix}__moe_output_relative_delta": moe_relative.astype(np.float32),
                f"{prefix}__residual_before_relative_delta": residual_before_relative.astype(np.float32),
                f"{prefix}__residual_after_relative_delta": residual_after_relative.astype(np.float32),
            }
        )
        positive_delta = np.maximum(delta_kld, 0.0)
        layer_metrics[str(layer)] = {
            "positions": int(kld_positions.size),
            "route_exact_order_fraction": float(routing["exact_order"].mean()),
            "route_exact_set_fraction": float(routing["exact_set"].mean()),
            "route_set_changed_fraction": float((~routing["exact_set"]).mean()),
            "route_overlap_mean": float(routing["overlap"].mean(dtype=np.float64)),
            "route_weight_l1_mean": float(routing["weight_l1"].mean(dtype=np.float64)),
            "hidden_relative_delta": {**_quantiles(hidden_relative), "mean": float(hidden_relative.mean(dtype=np.float64))},
            "moe_output_relative_delta": {**_quantiles(moe_relative), "mean": float(moe_relative.mean(dtype=np.float64))},
            "residual_before_relative_delta": {**_quantiles(residual_before_relative), "mean": float(residual_before_relative.mean(dtype=np.float64))},
            "residual_after_relative_delta": {**_quantiles(residual_after_relative), "mean": float(residual_after_relative.mean(dtype=np.float64))},
            "correlation_with_positive_kld_delta": {
                "route_weight_l1": _correlation(routing["weight_l1"], positive_delta),
                "moe_output_relative_delta": _correlation(moe_relative, positive_delta),
                "residual_after_relative_delta": _correlation(residual_after_relative, positive_delta),
            },
            "tail": {
                "positions": int(tail_mask.sum()),
                "route_set_changed_fraction": float((~routing["exact_set"])[tail_mask].mean()),
                "route_weight_l1_mean": float(routing["weight_l1"][tail_mask].mean(dtype=np.float64)),
                "moe_output_relative_delta_mean": float(moe_relative[tail_mask].mean(dtype=np.float64)),
                "residual_after_relative_delta_mean": float(residual_after_relative[tail_mask].mean(dtype=np.float64)),
            },
        }

    worst = np.argsort(delta_kld)[::-1][: args.worst_positions]
    report: dict[str, Any] = {
        "schema": "glm52-sqg-paired-tail-trace-analysis-v1",
        "alignment": "KLD index i joins trace token position i; position i predicts token i+1",
        "baseline_trace_root": str(args.baseline_trace.resolve()),
        "candidate_trace_root": str(args.candidate_trace.resolve()),
        "baseline_trace_evidence": baseline_evidence,
        "candidate_trace_evidence": candidate_evidence,
        "baseline_kld_evidence": baseline_kld_evidence,
        "candidate_kld_evidence": candidate_kld_evidence,
        "kld": {
            "positions": int(delta_kld.size),
            "baseline_mean": float(baseline_kld.mean(dtype=np.float64)),
            "candidate_mean": float(candidate_kld.mean(dtype=np.float64)),
            "mean_delta": float(delta_kld.mean(dtype=np.float64)),
            "median_delta": float(np.median(delta_kld)),
            "improved_fraction": float((delta_kld < 0.0).mean()),
            "worsened_fraction": float((delta_kld > 0.0).mean()),
            "delta_quantiles": _quantiles(delta_kld),
            "positive_delta_cvar_1pct": _upper_cvar(delta_kld),
            "positive_delta_sum": float(np.maximum(delta_kld, 0.0).sum(dtype=np.float64)),
            "negative_delta_sum": float(np.minimum(delta_kld, 0.0).sum(dtype=np.float64)),
        },
        "tail_definition": "largest 1 percent of candidate-minus-baseline per-position KLD deltas",
        "layers": layer_metrics,
        "worst_positive_positions": [
            {
                "position": int(index),
                "baseline_kld": float(baseline_kld[index]),
                "candidate_kld": float(candidate_kld[index]),
                "delta": float(delta_kld[index]),
            }
            for index in worst
        ],
    }

    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays_output = output.with_suffix(".npz")
    report_output = output.with_suffix(".md")
    for path in (arrays_output, report_output):
        if path.exists() or path.is_symlink():
            raise ValueError(f"output already exists: {path}")
    np.savez_compressed(arrays_output, **row_arrays)
    report["row_arrays"] = {
        "path": str(arrays_output),
        "sha256": _sha256(arrays_output),
    }
    partial = Path(f"{output}.partial")
    partial.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(partial, output)
    report_output.write_text(_render(report), encoding="utf-8")
    print(report_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
