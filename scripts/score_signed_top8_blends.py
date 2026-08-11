#!/usr/bin/env python3
"""Score H13 blends after the signed router-weighted top-8 expert sum.

Unlike the earlier per-expert metric, this scorer preserves each expert output's
vector sign, applies the exact captured route gate, sums all eight routed
outputs for every token, and only then squares the error.  It reports both
cross-expert cancellation and positive-tail risk. Selection and holdout roles
are invoked separately; in this experiment the existing holdout is
encoder-unseen but analysis-seen from Test 7, so it is secondary confirmation
rather than a newly blind document set.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_hessian_weighted_nmse import (  # noqa: E402
    DEFAULT_CAPTURE_ROOT,
    _load_hidden_chunk,
    _load_reference_functions,
    _read_source_triplet,
)
from scripts.compare_raw_encoded_nmse import (  # noqa: E402
    DEFAULT_BF16_ROOT,
    DEFAULT_MCG_ROOT,
    _load_sqg_manifests,
)
from scripts.compare_recalibrated_sqg import (  # noqa: E402
    HIDDEN,
    LAYERS,
    PROJECTIONS,
    ROLES,
    TOPK,
    _decode_sqg,
    _route_indices,
)


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _parse_candidate(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("candidate must be LABEL=/absolute/path")
    label, path_raw = raw.split("=", 1)
    if not label or not label.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError("candidate label is invalid")
    path = Path(path_raw)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("candidate path must be absolute")
    return label, path


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("tail values must be a non-empty vector")
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
    if not 0.0 < fraction <= 1.0:
        raise ValueError("CVaR fraction must be in (0,1]")
    count = max(1, math.ceil(values.size * fraction))
    boundary = values.size - count
    tail = np.partition(values, boundary)[boundary:]
    return float(tail.mean(dtype=np.float64))


def tail_comparison(
    candidate_error: np.ndarray,
    baseline_error: np.ndarray,
    reference_energy: np.ndarray,
) -> dict[str, Any]:
    """Describe per-position wins and positive-tail risk versus baseline."""

    candidate = np.asarray(candidate_error, dtype=np.float64)
    baseline = np.asarray(baseline_error, dtype=np.float64)
    energy = np.asarray(reference_energy, dtype=np.float64)
    if candidate.shape != baseline.shape or candidate.shape != energy.shape:
        raise ValueError("per-position vectors must have the same shape")
    if candidate.ndim != 1 or candidate.size == 0:
        raise ValueError("per-position vectors must be non-empty")
    scale = np.maximum(energy, np.finfo(np.float64).tiny)
    candidate_relative = candidate / scale
    baseline_relative = baseline / scale
    delta = candidate_relative - baseline_relative
    positive = np.maximum(delta, 0.0)
    improved = delta < 0.0
    worsened = delta > 0.0
    raw_delta = candidate - baseline
    return {
        "positions": int(delta.size),
        "improved_positions": int(improved.sum()),
        "worsened_positions": int(worsened.sum()),
        "unchanged_positions": int((delta == 0.0).sum()),
        "improved_fraction": float(improved.mean()),
        "worsened_fraction": float(worsened.mean()),
        "mean_relative_delta": float(delta.mean(dtype=np.float64)),
        "median_relative_delta": float(np.median(delta)),
        "candidate_squared_error": {
            **_quantiles(candidate),
            "mean": float(candidate.mean(dtype=np.float64)),
            "upper_cvar_1pct": _upper_cvar(candidate),
        },
        "baseline_squared_error": {
            **_quantiles(baseline),
            "mean": float(baseline.mean(dtype=np.float64)),
            "upper_cvar_1pct": _upper_cvar(baseline),
        },
        "squared_error_delta": {
            **_quantiles(raw_delta),
            "upper_cvar_1pct": _upper_cvar(raw_delta),
        },
        "candidate_relative_error": {
            **_quantiles(candidate_relative),
            "mean": float(candidate_relative.mean(dtype=np.float64)),
            "upper_cvar_1pct": _upper_cvar(candidate_relative),
        },
        "baseline_relative_error": {
            **_quantiles(baseline_relative),
            "mean": float(baseline_relative.mean(dtype=np.float64)),
            "upper_cvar_1pct": _upper_cvar(baseline_relative),
        },
        "relative_delta": {
            **_quantiles(delta),
            "upper_cvar_1pct": _upper_cvar(delta),
            "positive_mean_all_positions": float(positive.mean(dtype=np.float64)),
            "positive_max": float(positive.max()),
        },
    }


def _reduce_rows(
    reference_sum: torch.Tensor,
    delta_sums: dict[str, torch.Tensor],
    *,
    row_chunk: int = 2048,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = reference_sum.shape[0]
    reference_energy = np.empty(rows, dtype=np.float64)
    errors = {label: np.empty(rows, dtype=np.float64) for label in delta_sums}
    for begin in range(0, rows, row_chunk):
        end = min(rows, begin + row_chunk)
        reference_energy[begin:end] = (
            reference_sum[begin:end]
            .square()
            .sum(dim=1, dtype=torch.float64)
            .cpu()
            .numpy()
        )
        for label, values in delta_sums.items():
            errors[label][begin:end] = (
                values[begin:end]
                .square()
                .sum(dim=1, dtype=torch.float64)
                .cpu()
                .numpy()
            )
    return reference_energy, errors


def _score_layer(
    layer: int,
    gpu: int,
    candidates_raw: dict[str, str],
    baseline_label: str,
    role: str,
    mcg_root_raw: str,
    bf16_root_raw: str,
    capture_root_raw: str,
    project_root_raw: str,
    chunk_rows: int,
    output_root_raw: str,
) -> dict[str, Any]:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(gpu)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(f"cuda:{gpu}")
    candidates = {label: Path(path) for label, path in candidates_raw.items()}
    reference_root = candidates[baseline_label]
    mcg_root = Path(mcg_root_raw)
    bf16_root = Path(bf16_root_raw)
    capture_root = Path(capture_root_raw)
    project_root = Path(project_root_raw)
    output_root = Path(output_root_raw)

    (
        decode_exl3_weight,
        _,
        unpack_trellis_states,
        _,
        tensor_sha256,
        _,
    ) = _load_reference_functions(project_root)
    sidecar = json.loads(
        (mcg_root / f"r7-experts-layer-{layer:03d}.json").read_text()
    )
    manifests = {
        label: _load_sqg_manifests(root, layer)
        for label, root in candidates.items()
    }
    capture_layer = capture_root / f"layer_{layer:03d}"
    hidden_words, topk_weights, route_indices = _route_indices(capture_layer)
    role_path = capture_layer / "role_ids.u8.bin"
    total_rows = role_path.stat().st_size
    role_ids = np.memmap(role_path, mode="r", dtype="u1", shape=(total_rows,))
    selected_rows = np.flatnonzero(
        np.asarray(role_ids) == ROLES[role]
    ).astype(np.int64, copy=False)
    compact = np.full(total_rows, -1, dtype=np.int32)
    compact[selected_rows] = np.arange(selected_rows.size, dtype=np.int32)
    nrows = int(selected_rows.size)
    if nrows <= 0:
        raise RuntimeError(f"layer {layer} role {role} has no rows")

    reference_sum = torch.zeros((nrows, HIDDEN), dtype=torch.float32, device=device)
    delta_sums = {
        label: torch.zeros((nrows, HIDDEN), dtype=torch.float32, device=device)
        for label in candidates
    }
    individual_sse = {label: 0.0 for label in candidates}
    per_expert_sse = {label: np.zeros(256, dtype=np.float64) for label in candidates}
    route_count = 0

    with torch.inference_mode():
        for expert in range(256):
            rows, slots = route_indices[role][expert]
            if rows.size == 0:
                continue
            compact_rows = torch.from_numpy(compact[rows].astype(np.int64)).to(device)
            if bool((compact_rows < 0).any().item()):
                raise RuntimeError("role route failed compact-row mapping")
            gates = torch.from_numpy(
                np.array(topk_weights[rows, slots], dtype=np.float32, copy=True)
            ).to(device)
            source_hf = _read_source_triplet(bf16_root, sidecar, layer, expert)
            source = {
                projection: source_hf[projection].T.float().to(device)
                for projection in PROJECTIONS
            }
            weights = {
                label: _decode_sqg(
                    root=root,
                    reference_root=reference_root,
                    layer=layer,
                    expert=expert,
                    manifest=manifests[label][expert],
                    sidecar=sidecar,
                    device=device,
                    decode_exl3_weight=decode_exl3_weight,
                    unpack_trellis_states=unpack_trellis_states,
                    tensor_sha256=tensor_sha256,
                    verify_hashes=expert == 0,
                )
                for label, root in candidates.items()
            }

            for begin in range(0, rows.size, chunk_rows):
                end = min(rows.size, begin + chunk_rows)
                hidden = _load_hidden_chunk(hidden_words, rows[begin:end], device)
                gate = gates[begin:end, None]
                indices = compact_rows[begin:end]
                source_gate = hidden @ source["gate_proj"]
                source_up = hidden @ source["up_proj"]
                source_output = (F.silu(source_gate) * source_up) @ source["down_proj"]
                reference_sum.index_add_(0, indices, source_output * gate)
                for label in candidates:
                    gate_output = hidden @ weights[label]["gate_proj"]
                    up_output = hidden @ weights[label]["up_proj"]
                    candidate_output = (
                        F.silu(gate_output) * up_output
                    ) @ weights[label]["down_proj"]
                    signed_delta = (candidate_output - source_output) * gate
                    delta_sums[label].index_add_(0, indices, signed_delta)
                    route_sse = float(
                        signed_delta.square().sum(dtype=torch.float64).item()
                    )
                    individual_sse[label] += route_sse
                    per_expert_sse[label][expert] += route_sse
                    del gate_output, up_output, candidate_output, signed_delta
                del hidden, gate, indices, source_gate, source_up, source_output

            route_count += int(rows.size)
            del source_hf, source, weights, compact_rows, gates
            if (expert + 1) % 16 == 0:
                print(
                    f"layer {layer} {role}: {expert + 1}/256 experts",
                    flush=True,
                )

    reference_energy, errors = _reduce_rows(reference_sum, delta_sums)
    baseline_error = errors[baseline_label]
    metrics: dict[str, Any] = {}
    for label in candidates:
        summed_sse = float(errors[label].sum(dtype=np.float64))
        denominator = float(reference_energy.sum(dtype=np.float64))
        metrics[label] = {
            "signed_top8_nmse": summed_sse / denominator,
            "signed_top8_sse": summed_sse,
            "reference_signed_top8_energy": denominator,
            "sum_individual_route_sse": individual_sse[label],
            "summed_over_individual_sse": summed_sse / individual_sse[label],
            "cross_expert_error_term": summed_sse - individual_sse[label],
            "tail_vs_baseline": tail_comparison(
                errors[label], baseline_error, reference_energy
            ),
        }

    doc_epochs = np.memmap(
        capture_layer / "doc_epochs.u32le.bin",
        mode="r",
        dtype="<u4",
        shape=(total_rows,),
    )
    token_positions = np.memmap(
        capture_layer / "token_positions.u16le.bin",
        mode="r",
        dtype="<u2",
        shape=(total_rows,),
    )
    layer_output = output_root / f"layer_{layer:03d}-{role}.npz"
    np.savez_compressed(
        layer_output,
        global_rows=selected_rows,
        doc_epochs=np.asarray(doc_epochs[selected_rows]),
        token_positions=np.asarray(token_positions[selected_rows]),
        reference_energy=reference_energy,
        **{f"error__{label}": values for label, values in errors.items()},
    )
    result = {
        "layer": layer,
        "gpu": gpu,
        "role": role,
        "rows": nrows,
        "routes": route_count,
        "selected_global_rows_sha256": _sha256_array(selected_rows),
        "row_output": str(layer_output),
        "metrics": metrics,
        "per_expert_individual_route_sse": {
            label: values.tolist() for label, values in per_expert_sse.items()
        },
    }
    result_path = output_root / f"layer_{layer:03d}-{role}.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _aggregate(
    layer_results: dict[int, dict[str, Any]],
    labels: list[str],
    baseline_label: str,
    role: str,
) -> dict[str, Any]:
    aggregate_reference: np.ndarray | None = None
    aggregate_errors: dict[str, np.ndarray] = {}
    aggregate_individual = {label: 0.0 for label in labels}
    identity: tuple[np.ndarray, np.ndarray] | None = None
    for layer in LAYERS:
        result = layer_results[layer]
        with np.load(result["row_output"]) as payload:
            docs = payload["doc_epochs"]
            positions = payload["token_positions"]
            if identity is None:
                identity = (docs.copy(), positions.copy())
            elif not (
                np.array_equal(identity[0], docs)
                and np.array_equal(identity[1], positions)
            ):
                raise RuntimeError("capture role rows do not align across layers")
            reference = payload["reference_energy"]
            aggregate_reference = (
                reference.copy()
                if aggregate_reference is None
                else aggregate_reference + reference
            )
            for label in labels:
                values = payload[f"error__{label}"]
                if label not in aggregate_errors:
                    aggregate_errors[label] = values.copy()
                else:
                    aggregate_errors[label] += values
                aggregate_individual[label] += float(
                    result["metrics"][label]["sum_individual_route_sse"]
                )
    if aggregate_reference is None or identity is None:
        raise RuntimeError("no layer results were aggregated")
    baseline_error = aggregate_errors[baseline_label]
    metrics = {}
    for label in labels:
        summed = float(aggregate_errors[label].sum(dtype=np.float64))
        denominator = float(aggregate_reference.sum(dtype=np.float64))
        metrics[label] = {
            "signed_top8_nmse": summed / denominator,
            "signed_top8_sse": summed,
            "reference_signed_top8_energy": denominator,
            "sum_individual_route_sse": aggregate_individual[label],
            "summed_over_individual_sse": summed / aggregate_individual[label],
            "cross_expert_error_term": summed - aggregate_individual[label],
            "tail_vs_baseline": tail_comparison(
                aggregate_errors[label], baseline_error, aggregate_reference
            ),
        }

    baseline_tail = metrics[baseline_label]["tail_vs_baseline"][
        "baseline_relative_error"
    ]
    baseline_absolute_tail = metrics[baseline_label]["tail_vs_baseline"][
        "baseline_squared_error"
    ]
    eligible = []
    for label in labels:
        if label == baseline_label:
            continue
        tail = metrics[label]["tail_vs_baseline"]
        candidate_distribution = tail["candidate_relative_error"]
        candidate_absolute_distribution = tail["candidate_squared_error"]
        if (
            metrics[label]["signed_top8_nmse"]
            <= metrics[baseline_label]["signed_top8_nmse"]
            and candidate_absolute_distribution["upper_cvar_1pct"]
            <= baseline_absolute_tail["upper_cvar_1pct"]
            and candidate_absolute_distribution["p99"]
            <= baseline_absolute_tail["p99"]
            and candidate_distribution["upper_cvar_1pct"]
            <= baseline_tail["upper_cvar_1pct"]
            and candidate_distribution["p99"] <= baseline_tail["p99"]
        ):
            eligible.append(label)
    if eligible:
        winner = max(
            eligible,
            key=lambda label: (
                metrics[label]["tail_vs_baseline"]["improved_fraction"],
                -metrics[label]["tail_vs_baseline"]["candidate_relative_error"][
                    "upper_cvar_1pct"
                ],
                -metrics[label]["signed_top8_nmse"],
            ),
        )
        selection_status = "tail_and_mean_constraints_passed"
        diagnostic_fallback = None
    else:
        diagnostic_fallback = min(
            [label for label in labels if label != baseline_label],
            key=lambda label: (
                metrics[label]["tail_vs_baseline"]["candidate_squared_error"][
                    "upper_cvar_1pct"
                ],
                metrics[label]["tail_vs_baseline"]["candidate_relative_error"][
                    "upper_cvar_1pct"
                ],
                metrics[label]["signed_top8_nmse"],
                -metrics[label]["tail_vs_baseline"]["improved_fraction"],
            ),
        )
        winner = baseline_label
        selection_status = (
            "no_nonbaseline_candidate_passed_all_hard_constraints_baseline_retained"
        )

    return {
        "role": role,
        "layers": list(LAYERS),
        "positions": int(aggregate_reference.size),
        "baseline_label": baseline_label,
        "selection_policy": {
            "hard_constraints": [
                "aggregate signed-top8 NMSE <= baseline",
                "per-position squared-error upper-CVaR 1% <= baseline",
                "per-position squared-error p99 <= baseline",
                "per-position relative-error upper-CVaR 1% <= baseline",
                "per-position relative-error p99 <= baseline",
            ],
            "eligible_ranking": (
                "maximize improved-position fraction, then minimize upper-CVaR 1%, "
                "then minimize aggregate signed-top8 NMSE"
            ),
            "fallback_ranking": (
                "minimize squared-error upper-CVaR 1%, then relative-error "
                "upper-CVaR 1%, then aggregate signed-top8 NMSE, then "
                "maximize improved-position fraction"
            ),
            "status": selection_status,
            "eligible": eligible,
            "winner": winner,
            "diagnostic_fallback_nonbaseline": diagnostic_fallback,
        },
        "metrics": metrics,
    }


def _render(result: dict[str, Any]) -> str:
    aggregate = result["aggregate"]
    lines = [
        f"# Signed top-8 H13 blend score — {aggregate['role']}",
        "",
        "Errors are summed as signed router-weighted expert-output vectors before squaring.",
        "",
        "| Blend | Signed top-8 NMSE | Sum/individual SSE | Positions improved | Squared-error CVaR 1% | Relative-error CVaR 1% | p99 relative error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metric in aggregate["metrics"].items():
        tail = metric["tail_vs_baseline"]
        lines.append(
            f"| {label} | {metric['signed_top8_nmse']:.9e} | "
            f"{metric['summed_over_individual_sse']:.6f} | "
            f"{tail['improved_fraction']:.3%} | "
            f"{tail['candidate_squared_error']['upper_cvar_1pct']:.9e} | "
            f"{tail['candidate_relative_error']['upper_cvar_1pct']:.9e} | "
            f"{tail['candidate_relative_error']['p99']:.9e} |"
        )
    policy = aggregate["selection_policy"]
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Status: `{policy['status']}`",
            f"- Eligible candidates: `{policy['eligible']}`",
            f"- Selected candidate: **`{policy['winner']}`**",
            f"- Diagnostic nonbaseline fallback: `{policy['diagnostic_fallback_nonbaseline']}`",
            "",
            "The improved-position fraction is optimized only after aggregate and positive-tail constraints pass.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        type=_parse_candidate,
        metavar="LABEL=/ABSOLUTE/PATH",
    )
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--role", choices=("selection", "holdout"), required=True)
    parser.add_argument("--mcg-root", type=Path, default=DEFAULT_MCG_ROOT)
    parser.add_argument("--bf16-root", type=Path, default=DEFAULT_BF16_ROOT)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.chunk_rows <= 0:
        raise ValueError("chunk rows must be positive")
    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate):
        raise ValueError("candidate labels must be unique")
    if args.baseline_label not in candidates:
        raise ValueError("baseline label is not among candidates")
    for label, path in candidates.items():
        if not path.is_dir():
            raise ValueError(f"candidate {label} is absent: {path}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    layer_output_root = output.parent / f"{output.stem}-layers"
    layer_output_root.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    layer_results: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        futures = {
            pool.submit(
                _score_layer,
                layer,
                gpu,
                {label: str(path.resolve()) for label, path in candidates.items()},
                args.baseline_label,
                args.role,
                str(args.mcg_root.resolve()),
                str(args.bf16_root.resolve()),
                str(args.capture_root.resolve()),
                str(PROJECT_ROOT),
                args.chunk_rows,
                str(layer_output_root),
            ): layer
            for gpu, layer in enumerate(LAYERS)
        }
        for future in as_completed(futures):
            layer = futures[future]
            layer_results[layer] = future.result()
            print(f"layer {layer}: signed top-8 scoring complete", flush=True)

    aggregate = _aggregate(
        layer_results,
        list(candidates),
        args.baseline_label,
        args.role,
    )
    result = {
        "schema": "glm52-signed-top8-h13-blend-score-v1",
        "candidates": {label: str(path.resolve()) for label, path in candidates.items()},
        "baseline_label": args.baseline_label,
        "role": args.role,
        "method": {
            "signed_sum_before_square": True,
            "exact_captured_top8_ids_and_applied_gates": True,
            "bf16_reference_expert_functions": True,
            "candidate_packed_bytes_decoded": True,
            "tail_control_precedes_win_rate": True,
            "fit_rows_used": False,
            "selection_rows_used": args.role == "selection",
            "holdout_rows_used": args.role == "holdout",
            "holdout_encoder_unseen_but_analysis_seen_from_test7": (
                args.role == "holdout"
            ),
        },
        "aggregate": aggregate,
        "layers": {str(layer): layer_results[layer] for layer in LAYERS},
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = output.with_suffix(".md")
    report.write_text(_render(result), encoding="utf-8")
    print(json.dumps(aggregate["selection_policy"], indent=2, sort_keys=True))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
