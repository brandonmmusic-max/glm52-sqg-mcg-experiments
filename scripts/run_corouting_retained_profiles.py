#!/usr/bin/env python3
"""Select retained layer-77 profiles with an expert-unary safety bound.

The retained winner-native cells were encoded before this experiment.  This
runner changes no tensor bytes.  On selection documents it chooses one
existing cell per panel expert to minimize the exact signed, gate-weighted
sum of routed expert errors, subject to a relative bound on every expert's
individual error.  Only the frozen selection winner is then evaluated on the
untouched holdout documents.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA = "glm52-alpha025-retained-corouting-v1"
SLACKS = (0.0, 0.0025, 0.005, 0.01)
BASELINE_CELL_ID = "draw-00__identity"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--layer", type=int, default=77)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--chunk-rows", type=int, default=256)
    return parser


def _configure_threads(count: int) -> None:
    if count <= 0:
        raise ValueError("threads must be positive")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(count)


def _chunks(total: int, rows: int):
    for begin in range(0, total, rows):
        yield begin, min(total, begin + rows)


def _sum_squares(values: "torch.Tensor", *, chunk_rows: int = 256) -> float:
    import torch

    total = torch.zeros((), dtype=torch.float64, device=values.device)
    for begin, end in _chunks(values.shape[0], chunk_rows):
        total += values[begin:end].square().sum(dtype=torch.float64)
    return float(total)


def _unary_costs(values: "torch.Tensor", *, chunk_rows: int = 256) -> "torch.Tensor":
    import torch

    modes = values.shape[1]
    total = torch.zeros((modes,), dtype=torch.float64, device=values.device)
    for begin, end in _chunks(values.shape[0], chunk_rows):
        total += values[begin:end].square().sum(dim=(0, 2), dtype=torch.float64)
    return total


def optimize_corouted_profiles(
    positions: Mapping[int, "torch.Tensor"],
    errors: Mapping[int, "torch.Tensor"],
    *,
    role_rows: int,
    hidden: int,
    unary_relative_slack: float,
    maximum_sweeps: int = 20,
) -> dict[str, Any]:
    """Coordinate-select exact sparse expert contributions.

    This is the sparse, full-feature equivalent of QSRT's analysis selector:
    the objective is ``sum_rows ||sum_experts error[e, mode[e]]||^2``.
    No random projection or proxy feature space is used.
    """

    import torch

    if not math.isfinite(unary_relative_slack) or unary_relative_slack < 0:
        raise ValueError("unary slack must be finite and non-negative")
    experts = tuple(sorted(errors))
    if not experts or set(experts) != set(positions):
        raise ValueError("co-routing expert domains differ")
    device = errors[experts[0]].device
    modes = errors[experts[0]].shape[1]
    if modes < 2:
        raise ValueError("co-routing requires multiple candidates")
    unary: dict[int, torch.Tensor] = {}
    allowed: dict[int, torch.Tensor] = {}
    unary_choice: dict[int, int] = {}
    for expert in experts:
        values = errors[expert]
        if values.ndim != 3 or values.shape[1:] != (modes, hidden):
            raise ValueError("candidate error shapes differ")
        if positions[expert].shape != (values.shape[0],):
            raise ValueError("candidate positions do not align")
        costs = _unary_costs(values)
        best = costs.min()
        keep = costs <= best * (1.0 + unary_relative_slack) + 1e-30
        if not bool(keep.any()):
            raise RuntimeError("unary bound removed every candidate")
        unary[expert] = costs
        allowed[expert] = keep
        unary_choice[expert] = int(costs.masked_fill(~keep, float("inf")).argmin())

    def build_aggregate(selection: Mapping[int, int]) -> "torch.Tensor":
        aggregate = torch.zeros(
            (role_rows, hidden), dtype=torch.float32, device=device
        )
        for expert in experts:
            aggregate.index_add_(
                0,
                positions[expert],
                errors[expert][:, int(selection[expert]), :],
            )
        return aggregate

    def run_start(initial: Mapping[int, int]) -> tuple[dict[int, int], "torch.Tensor", int, int]:
        selection = {expert: int(initial[expert]) for expert in experts}
        aggregate = build_aggregate(selection)
        total_changes = 0
        completed_sweeps = 0
        for completed_sweeps in range(1, maximum_sweeps + 1):
            sweep_changes = 0
            for expert in experts:
                row_index = positions[expert]
                values = errors[expert]
                current = selection[expert]
                base = aggregate.index_select(0, row_index) - values[:, current, :]
                base_energy = _sum_squares(base)
                dot = torch.einsum("rh,rmh->m", base, values).double()
                costs = unary[expert] + 2.0 * dot + base_energy
                costs = costs.masked_fill(~allowed[expert], float("inf"))
                winner = int(costs.argmin())
                if winner == current:
                    continue
                aggregate.index_copy_(0, row_index, base + values[:, winner, :])
                selection[expert] = winner
                sweep_changes += 1
            total_changes += sweep_changes
            if sweep_changes == 0:
                break
        return selection, aggregate, completed_sweeps, total_changes

    starts: list[dict[int, int]] = [dict(unary_choice)]
    for mode in range(modes):
        starts.append(
            {
                expert: mode if bool(allowed[expert][mode]) else unary_choice[expert]
                for expert in experts
            }
        )
    best: tuple[dict[int, int], "torch.Tensor", int, int] | None = None
    best_objective = float("inf")
    for initial in starts:
        trial = run_start(initial)
        objective = _sum_squares(trial[1])
        if objective < best_objective:
            if best is not None:
                old_aggregate = best[1]
                del old_aggregate
            best = trial
            best_objective = objective
        else:
            rejected_aggregate = trial[1]
            del rejected_aggregate
            del trial
    if best is None:
        raise RuntimeError("co-routing solver produced no candidate")
    selection, aggregate, completed_sweeps, changes = best
    selected_unary = sum(
        float(unary[expert][selection[expert]]) for expert in experts
    )
    return {
        "selection": selection,
        "aggregate": aggregate,
        "objective": best_objective,
        "selected_unary": selected_unary,
        "cross_term": best_objective - selected_unary,
        "sweeps": completed_sweeps,
        "changes": changes,
        "unary_costs": {
            expert: [float(value) for value in unary[expert]] for expert in experts
        },
        "allowed_modes": {
            expert: [bool(value) for value in allowed[expert]] for expert in experts
        },
    }


def _position_metrics(error: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    error_energy = np.square(error.astype(np.float64, copy=False)).sum(axis=1)
    reference_energy = np.square(reference.astype(np.float64, copy=False)).sum(axis=1)
    relative = np.divide(
        error_energy,
        reference_energy,
        out=np.zeros_like(error_energy),
        where=reference_energy > 1e-30,
    )

    def summary(values: np.ndarray) -> dict[str, float]:
        ordered = np.sort(values)
        tail_count = max(1, math.ceil(0.01 * ordered.size))
        return {
            "mean": float(values.mean()),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "upper_cvar_1pct": float(ordered[-tail_count:].mean()),
            "max": float(values.max()),
        }

    return {"relative_error": summary(relative), "relative_values": relative}


def _score_aggregate(
    aggregate: "torch.Tensor",
    reference: "torch.Tensor",
    document_epochs: np.ndarray,
    *,
    chunk_rows: int,
) -> dict[str, Any]:
    from src.fresh_pipeline_evaluation import document_scores_from_routed_aggregates

    error_np = aggregate.detach().cpu().numpy()
    reference_np = reference.detach().cpu().numpy()
    result = document_scores_from_routed_aggregates(
        error_np,
        reference_np,
        document_epochs,
        chunk_rows=chunk_rows,
    )
    position = _position_metrics(error_np, reference_np)
    result["position_relative_error"] = position["relative_error"]
    result["_position_relative_values"] = position["relative_values"]
    return result


def _aggregate_for(
    reference: "torch.Tensor",
    positions: Mapping[int, "torch.Tensor"],
    errors: Mapping[int, "torch.Tensor"],
    modes: Mapping[int, int],
    panel: Sequence[int],
) -> "torch.Tensor":
    import torch

    aggregate = torch.zeros_like(reference)
    for expert in panel:
        aggregate.index_add_(
            0,
            positions[expert],
            errors[expert][:, int(modes[expert]), :],
        )
    return aggregate


def _candidate_output(hidden: "torch.Tensor", decoded: Mapping[str, "torch.Tensor"]) -> "torch.Tensor":
    import torch.nn.functional as functional

    return (
        functional.silu(hidden @ decoded["gate_proj"])
        * (hidden @ decoded["up_proj"])
    ) @ decoded["down_proj"]


def _reference_output(hidden: "torch.Tensor", weights: Mapping[str, "torch.Tensor"]) -> "torch.Tensor":
    import torch.nn.functional as functional

    middle = functional.silu(functional.linear(hidden, weights["gate_proj"]))
    middle *= functional.linear(hidden, weights["up_proj"])
    return functional.linear(middle, weights["down_proj"])


def _load_decoded(
    root: Path,
    *,
    layer: int,
    expert: int,
    lut_by_bits: Mapping[int, "torch.Tensor"],
    device: "torch.device",
) -> dict[str, "torch.Tensor"]:
    import torch

    from src.fresh_pipeline_artifacts import expert_stem, load_decoded_expert

    decoded = load_decoded_expert(
        root / f"{expert_stem(layer, expert)}.json",
        lut_by_bits=lut_by_bits,
    )
    return {
        name: value.to(device=device, dtype=torch.float32)
        for name, value in decoded.items()
    }


def _source_weights(runtime: Any, expert: int, device: "torch.device") -> dict[str, "torch.Tensor"]:
    import torch

    source = runtime.source.load_expert_bf16(runtime.layer, expert, device="cpu")
    return {
        "gate_proj": source.gate_hf.to(device=device, dtype=torch.float32),
        "up_proj": source.up_hf.to(device=device, dtype=torch.float32),
        "down_proj": source.down_hf.to(device=device, dtype=torch.float32),
    }


def _role_layout(runtime: Any, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    role_rows = runtime.capture.role_rows(role)
    if not role_rows.size:
        raise ValueError(f"{role} capture is empty")
    compact = np.full(runtime.capture.rows, -1, dtype=np.int64)
    compact[role_rows] = np.arange(role_rows.size, dtype=np.int64)
    documents = np.asarray(runtime.capture.doc_epochs[role_rows], dtype=np.int64)
    return role_rows, compact, documents


def _selection_inputs(
    runtime: Any,
    kquant_runtime: Any,
    roots: Sequence[Path],
    panel: Sequence[int],
    *,
    device: "torch.device",
    chunk_rows: int,
) -> tuple[
    dict[int, "torch.Tensor"],
    dict[int, "torch.Tensor"],
    "torch.Tensor",
    np.ndarray,
]:
    import torch

    role_rows, compact, documents = _role_layout(runtime, "selection")
    hidden_width = int(runtime.capture.hidden_words.shape[1])
    reference = torch.zeros(
        (role_rows.size, hidden_width), dtype=torch.float32, device=device
    )
    positions_by_expert: dict[int, torch.Tensor] = {}
    errors_by_expert: dict[int, torch.Tensor] = {}
    lut_by_bits = {
        bits: kquant_runtime.lut_bytes(bits).detach().cpu().contiguous()
        for bits in (3, 4)
    }
    for expert in panel:
        routed = runtime.capture.routed_rows(expert, "selection")
        positions_np = compact[routed.row_indices]
        if bool((positions_np < 0).any()):
            raise RuntimeError("selection role position map is inconsistent")
        positions = torch.from_numpy(positions_np).to(device=device)
        values = torch.empty(
            (routed.rows, len(roots), hidden_width),
            dtype=torch.float32,
            device=device,
        )
        decoded = [
            _load_decoded(
                root,
                layer=runtime.layer,
                expert=expert,
                lut_by_bits=lut_by_bits,
                device=device,
            )
            for root in roots
        ]
        source = _source_weights(runtime, expert, device)
        for begin, end in _chunks(routed.rows, chunk_rows):
            absolute = routed.row_indices[begin:end]
            hidden = runtime.capture.load_hidden(
                absolute, device=device, dtype=torch.float32
            )
            gates = routed.applied_gates[begin:end].to(
                device=device, dtype=torch.float32
            )[:, None]
            with torch.no_grad():
                target = _reference_output(hidden, source)
                reference.index_add_(
                    0, positions[begin:end], target * gates
                )
                for mode, candidate in enumerate(decoded):
                    values[begin:end, mode, :] = (
                        _candidate_output(hidden, candidate) - target
                    ) * gates
        positions_by_expert[expert] = positions
        errors_by_expert[expert] = values
        del decoded, source
        gc.collect()
        torch.cuda.empty_cache()
    return positions_by_expert, errors_by_expert, reference, documents


def _holdout_aggregates(
    runtime: Any,
    kquant_runtime: Any,
    roots: Sequence[Path],
    panel: Sequence[int],
    *,
    baseline_mode: int,
    selected_modes: Mapping[int, int],
    device: "torch.device",
    chunk_rows: int,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", np.ndarray]:
    import torch

    role_rows, compact, documents = _role_layout(runtime, "holdout")
    shape = (role_rows.size, int(runtime.capture.hidden_words.shape[1]))
    baseline = torch.zeros(shape, dtype=torch.float32, device=device)
    selected = torch.zeros(shape, dtype=torch.float32, device=device)
    reference = torch.zeros(shape, dtype=torch.float32, device=device)
    lut_by_bits = {
        bits: kquant_runtime.lut_bytes(bits).detach().cpu().contiguous()
        for bits in (3, 4)
    }
    for expert in panel:
        routed = runtime.capture.routed_rows(expert, "holdout")
        positions_np = compact[routed.row_indices]
        if bool((positions_np < 0).any()):
            raise RuntimeError("holdout role position map is inconsistent")
        positions = torch.from_numpy(positions_np).to(device=device)
        candidate_modes = {baseline_mode, int(selected_modes[expert])}
        decoded = {
            mode: _load_decoded(
                roots[mode],
                layer=runtime.layer,
                expert=expert,
                lut_by_bits=lut_by_bits,
                device=device,
            )
            for mode in candidate_modes
        }
        source = _source_weights(runtime, expert, device)
        for begin, end in _chunks(routed.rows, chunk_rows):
            absolute = routed.row_indices[begin:end]
            hidden = runtime.capture.load_hidden(
                absolute, device=device, dtype=torch.float32
            )
            gates = routed.applied_gates[begin:end].to(
                device=device, dtype=torch.float32
            )[:, None]
            with torch.no_grad():
                target = _reference_output(hidden, source)
                reference.index_add_(0, positions[begin:end], target * gates)
                baseline.index_add_(
                    0,
                    positions[begin:end],
                    (_candidate_output(hidden, decoded[baseline_mode]) - target)
                    * gates,
                )
                selected.index_add_(
                    0,
                    positions[begin:end],
                    (
                        _candidate_output(
                            hidden, decoded[int(selected_modes[expert])]
                        )
                        - target
                    )
                    * gates,
                )
        del decoded, source
        gc.collect()
        torch.cuda.empty_cache()
    return baseline, selected, reference, documents


def _comparison_metrics(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_values = base.pop("_position_relative_values")
    candidate_values = candidate.pop("_position_relative_values")
    delta = candidate_values - baseline_values
    improved = candidate_values < baseline_values
    ordered_positive = np.sort(delta[delta > 0])
    cvar = (
        float(ordered_positive[-max(1, math.ceil(0.01 * delta.size)) :].mean())
        if ordered_positive.size
        else 0.0
    )
    return {
        "baseline": base,
        "candidate": candidate,
        "relative_to_baseline_percent": (
            float(candidate["aggregate_relative_error"])
            / float(base["aggregate_relative_error"])
            - 1.0
        )
        * 100.0,
        "position_win_fraction": float(improved.mean()),
        "position_worsen_fraction": float((delta > 0).mean()),
        "position_mean_relative_delta": float(delta.mean()),
        "position_median_relative_delta": float(np.median(delta)),
        "position_positive_delta_cvar_1pct_all_positions": cvar,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.layer != 77:
        raise ValueError("the retained co-routing pilot is preregistered for layer 77")
    if args.chunk_rows <= 0:
        raise ValueError("chunk rows must be positive")
    _configure_threads(args.threads)

    import torch

    from scripts.encode_final_shard import _open_fast_sealed_runtime
    from src.fresh_pipeline_artifacts import expert_stem, validate_expert_artifact
    from src.fresh_pipeline_common import (
        atomic_json,
        canonical_sha256,
        derive_seed,
        load_json_object,
        sha256_file,
    )
    from src.fresh_pipeline_runner import (
        FAMILYWISE_ALPHA,
        _load_bound_kquant_runtime,
        paired_document_bootstrap,
    )

    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    runtime = _open_fast_sealed_runtime(
        args.preflight, layer=args.layer, device=args.device
    )
    target = torch.device(args.device)
    search_root = runtime.layer_root / "winner_native_profile_search"
    selection_path = search_root / "selection.json"
    profile_prereg_path = search_root / "preregistration.json"
    selection = load_json_object(selection_path)
    profile_prereg = load_json_object(profile_prereg_path)
    if selection.get("complete") is not True or profile_prereg.get("complete") is not True:
        raise ValueError("winner-native profile evidence is incomplete")
    retained = tuple(str(value) for value in selection["retained_for_corouting_analysis"])
    if retained != (
        "draw-00__identity",
        "draw-03__identity",
        "draw-01__identity",
        "draw-02__identity",
    ):
        raise ValueError(f"retained candidate contract differs: {retained!r}")
    if selection.get("selected_cell_id") != BASELINE_CELL_ID:
        raise ValueError("frozen winner-native baseline is not draw-00 identity")
    panel = tuple(int(value) for value in profile_prereg["selection_panel"])
    roots = tuple(search_root / "cells" / cell / "experts" for cell in retained)
    baseline_mode = retained.index(BASELINE_CELL_ID)

    artifact_hashes: dict[str, dict[str, str]] = {}
    for cell, root in zip(retained, roots):
        cell_hashes: dict[str, str] = {}
        for expert in panel:
            manifest = root / f"{expert_stem(runtime.layer, expert)}.json"
            evidence = validate_expert_artifact(manifest)
            if int(evidence["expert"]) != expert or int(evidence["layer"]) != runtime.layer:
                raise ValueError("retained artifact binding differs")
            cell_hashes[str(expert)] = sha256_file(manifest)
        artifact_hashes[cell] = cell_hashes

    output_root = search_root / "corouting_retained_r1"
    output_root.mkdir(parents=True, exist_ok=True)
    prereg_path = output_root / "preregistration.json"
    result_path = output_root / "result.json"
    prereg: dict[str, Any] = {
        "schema": SCHEMA,
        "complete": True,
        "layer": runtime.layer,
        "winner_native_selection_id": selection["selection_id"],
        "winner_native_selection_sha256": sha256_file(selection_path),
        "winner_native_preregistration_sha256": sha256_file(profile_prereg_path),
        "baseline_cell_id": BASELINE_CELL_ID,
        "retained_cells": list(retained),
        "panel_experts": list(panel),
        "candidate_artifact_manifest_sha256": artifact_hashes,
        "arms": [
            {"arm_id": f"unary_slack_{slack:.4f}", "unary_relative_slack": slack}
            for slack in SLACKS
        ],
        "selection": {
            "role": "selection",
            "objective": "exact_signed_applied_gate_weighted_top8_output_sse",
            "features": int(runtime.capture.hidden_words.shape[1]),
            "random_projection": False,
            "coordinate_starts": "unary_plus_each_uniform_retained_mode",
            "maximum_sweeps": 20,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "bonferroni_comparisons": len(SLACKS),
            "winner_rule": (
                "lowest_exact_selection_error_below_baseline_with_positive_"
                "bonferroni_paired_document_bootstrap_lower_bound_else_baseline"
            ),
        },
        "holdout": {
            "role": "holdout",
            "evaluated_candidates": "frozen_selection_winner_only",
            "used_for_selection": False,
        },
        "rate_contract": "unchanged_independent_per_tensor_k3_k4",
        "new_encodes": 0,
        "uniform_k3": False,
        "mcg_inputs": 0,
    }
    prereg["preregistration_id"] = canonical_sha256(prereg)
    if prereg_path.exists():
        if load_json_object(prereg_path) != prereg:
            raise ValueError("co-routing preregistration differs")
    else:
        atomic_json(prereg_path, prereg)
    if result_path.exists():
        print(json.dumps(load_json_object(result_path), indent=2, sort_keys=True))
        return 0

    kquant_runtime = _load_bound_kquant_runtime(runtime)
    positions, errors, reference, selection_documents = _selection_inputs(
        runtime,
        kquant_runtime,
        roots,
        panel,
        device=target,
        chunk_rows=args.chunk_rows,
    )
    baseline_selection = {expert: baseline_mode for expert in panel}

    baseline_aggregate = _aggregate_for(
        reference, positions, errors, baseline_selection, panel
    )
    baseline_score = _score_aggregate(
        baseline_aggregate,
        reference,
        selection_documents,
        chunk_rows=args.chunk_rows,
    )
    baseline_score["assignment"] = {
        str(expert): BASELINE_CELL_ID for expert in panel
    }
    selection_arms: dict[str, dict[str, Any]] = {}
    confidence = 1.0 - FAMILYWISE_ALPHA / len(SLACKS)
    for slack in SLACKS:
        arm_id = f"unary_slack_{slack:.4f}"
        optimized = optimize_corouted_profiles(
            positions,
            errors,
            role_rows=reference.shape[0],
            hidden=reference.shape[1],
            unary_relative_slack=slack,
        )
        candidate_score = _score_aggregate(
            optimized.pop("aggregate"),
            reference,
            selection_documents,
            chunk_rows=args.chunk_rows,
        )
        comparison = _comparison_metrics(
            dict(baseline_score), candidate_score
        )
        bootstrap = paired_document_bootstrap(
            comparison["baseline"],
            comparison["candidate"],
            seed=derive_seed(
                runtime.settings.run_id,
                runtime.layer,
                arm_id,
                "retained-corouting-selection",
            ),
            iterations=10_000,
            confidence=confidence,
        )
        modes = {int(expert): int(mode) for expert, mode in optimized.pop("selection").items()}
        selection_arms[arm_id] = {
            "unary_relative_slack": slack,
            "assignment_mode_index": {str(expert): mode for expert, mode in modes.items()},
            "assignment_cell_id": {
                str(expert): retained[mode] for expert, mode in modes.items()
            },
            "solver": optimized,
            "comparison": comparison,
            "paired_document_bootstrap_vs_baseline": bootstrap,
        }
        torch.cuda.empty_cache()

    eligible = [
        (arm_id, arm)
        for arm_id, arm in selection_arms.items()
        if float(arm["comparison"]["relative_to_baseline_percent"]) < 0
        and bool(
            arm["paired_document_bootstrap_vs_baseline"]["lower_bound_gt_zero"]
        )
    ]
    if eligible:
        selected_arm_id, selected_arm = min(
            eligible,
            key=lambda item: (
                float(item[1]["comparison"]["candidate"]["aggregate_relative_error"]),
                item[0],
            ),
        )
        selected_modes = {
            int(expert): int(mode)
            for expert, mode in selected_arm["assignment_mode_index"].items()
        }
    else:
        selected_arm_id = "baseline_fallback"
        selected_modes = baseline_selection

    del errors, positions, baseline_aggregate, reference
    gc.collect()
    torch.cuda.empty_cache()

    holdout_base, holdout_candidate, holdout_reference, holdout_documents = (
        _holdout_aggregates(
            runtime,
            kquant_runtime,
            roots,
            panel,
            baseline_mode=baseline_mode,
            selected_modes=selected_modes,
            device=target,
            chunk_rows=args.chunk_rows,
        )
    )
    holdout_base_score = _score_aggregate(
        holdout_base,
        holdout_reference,
        holdout_documents,
        chunk_rows=args.chunk_rows,
    )
    holdout_candidate_score = _score_aggregate(
        holdout_candidate,
        holdout_reference,
        holdout_documents,
        chunk_rows=args.chunk_rows,
    )
    holdout_comparison = _comparison_metrics(
        holdout_base_score, holdout_candidate_score
    )
    holdout_bootstrap = paired_document_bootstrap(
        holdout_comparison["baseline"],
        holdout_comparison["candidate"],
        seed=derive_seed(
            runtime.settings.run_id,
            runtime.layer,
            selected_arm_id,
            "retained-corouting-holdout",
        ),
        iterations=10_000,
        confidence=0.95,
    )

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "complete": True,
        "layer": runtime.layer,
        "preregistration_id": prereg["preregistration_id"],
        "preregistration_sha256": sha256_file(prereg_path),
        "selection_arms": selection_arms,
        "selected_arm_id": selected_arm_id,
        "selected_assignment_mode_index": {
            str(expert): int(mode) for expert, mode in selected_modes.items()
        },
        "selected_assignment_cell_id": {
            str(expert): retained[int(mode)] for expert, mode in selected_modes.items()
        },
        "holdout_comparison": holdout_comparison,
        "holdout_paired_document_bootstrap": holdout_bootstrap,
        "selection_documents": int(np.unique(selection_documents).size),
        "holdout_documents": int(np.unique(holdout_documents).size),
        "panel_experts": list(panel),
        "new_encodes": 0,
        "uniform_k3": False,
        "mcg_inputs": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    result["result_id"] = canonical_sha256(result)
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
