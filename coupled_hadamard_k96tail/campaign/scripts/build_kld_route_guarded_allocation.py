#!/usr/bin/env python3
"""Shift a measured K96 allocation toward source-worst KLD routes.

This is deliberately a guarded, source-checkpoint fit signal rather than an
end-to-end quality claim.  It preserves the exact layer-native K96 budget and
candidate records, maximizes KLD-weighted route coverage over the frozen worst
positions, and permits at most the requested regression against the measured
base K96 allocation in both total and fixed-body calibration objectives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPERTS = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-allocation", type=Path, required=True)
    parser.add_argument("--score-manifest", type=Path, required=True)
    parser.add_argument("--kld-json", type=Path, required=True)
    parser.add_argument("--routes-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--worst-count", type=int, default=40)
    parser.add_argument("--total-regression-limit", type=float, default=0.01)
    parser.add_argument("--body-regression-limit", type=float, default=0.01)
    return parser


def selected_candidate(expert: dict[str, Any], rates: dict[str, int]) -> dict[str, Any]:
    matches = [item for item in expert["candidates"] if item.get("rates") == rates]
    if len(matches) != 1:
        raise ValueError(f"candidate domain differs for rates {rates}")
    return dict(matches[0])


def selection_for_lambda(
    delta_total: np.ndarray, bonus: np.ndarray, value: float
) -> frozenset[int]:
    adjusted = delta_total - value * bonus
    ordered = sorted(range(EXPERTS), key=lambda expert: (adjusted[expert], expert))
    return frozenset(ordered[:96])


def scalarization_intervals(delta_total: np.ndarray, bonus: np.ndarray) -> list[float]:
    crossings = {0.0}
    for left in range(EXPERTS):
        for right in range(left + 1, EXPERTS):
            denominator = float(bonus[left] - bonus[right])
            if denominator == 0.0:
                continue
            value = float((delta_total[left] - delta_total[right]) / denominator)
            if math.isfinite(value) and value > 0.0:
                crossings.add(value)
    ordered = sorted(crossings)
    probes = [0.0]
    for left, right in zip(ordered, ordered[1:], strict=False):
        probes.append(left + (right - left) * 0.5)
    if ordered[-1] > 0.0:
        probes.append(ordered[-1] * 2.0)
    return probes


def main() -> None:
    args = args_parser().parse_args()
    if not 3 <= args.layer <= 78:
        raise ValueError("routed layer must lie in 3..78")
    if args.worst_count <= 0:
        raise ValueError("worst-count must be positive")
    for limit in (args.total_regression_limit, args.body_regression_limit):
        if not math.isfinite(limit) or not 0.0 <= limit <= 0.1:
            raise ValueError("regression limits must be finite and in [0,0.1]")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite allocation: {args.output}")

    base = json.loads(args.base_allocation.read_text(encoding="utf-8"))
    scores = json.loads(args.score_manifest.read_text(encoding="utf-8"))
    kld_record = json.loads(args.kld_json.read_text(encoding="utf-8"))
    if (
        base.get("complete") is not True
        or base.get("production_eligible") is not True
        or int(base.get("layer", -1)) != args.layer
        or base.get("histogram") != {"3": 672, "4": 96}
        or sum(int(base.get("projection_k4", {}).get(name, -1)) for name in PROJECTIONS)
        != 96
        or scores.get("complete") is not True
        or int(scores.get("layer", -1)) != args.layer
        or set(scores.get("experts", {})) != {str(i) for i in range(EXPERTS)}
        or kld_record.get("complete") is not True
    ):
        raise ValueError("base K96, score, or KLD contract differs")

    if args.layer == 78:
        from src.fresh_pipeline_common import atomic_json, canonical_sha256

        result = dict(base)
        result.update(
            {
                "method": "layer_native_k96_mtp78_without_unavailable_source_route_signal_v1",
                "loss_definition": "layer_native_candidate_calibration_objective_v6_mtp78",
                "kld_route_policy": {
                    "applicable": False,
                    "reason": "sealed_source_route_capture_indexes_model_layers_0_through_77_and_has_no_mtp78_route_plane",
                    "end_to_end_quality_claim": False,
                    "base_allocation_retained_exactly": True,
                    "source_kld": {
                        "path": str(args.kld_json.resolve()),
                        "sha256": sha256_file(args.kld_json),
                    },
                    "source_routes": {
                        "path": str(args.routes_npz.resolve()),
                        "sha256": sha256_file(args.routes_npz),
                    },
                    "base_allocation": {
                        "path": str(args.base_allocation.resolve()),
                        "sha256": sha256_file(args.base_allocation),
                        "allocation_id": base["allocation_id"],
                    },
                    "score_manifest": {
                        "path": str(args.score_manifest.resolve()),
                        "sha256": sha256_file(args.score_manifest),
                        "score_manifest_id": scores["score_manifest_id"],
                    },
                },
                "production_eligible": True,
            }
        )
        result.pop("allocation_id", None)
        result["allocation_id"] = canonical_sha256(result)
        atomic_json(args.output, result)
        print(
            json.dumps(
                {
                    "allocation_id": result["allocation_id"],
                    "output": str(args.output.resolve()),
                    "mtp78_route_signal_applicable": False,
                    "swaps": 0,
                },
                sort_keys=True,
            )
        )
        return

    position_kld = np.asarray(kld_record["position_kld"], dtype=np.float64)
    if (
        position_kld.ndim != 1
        or not np.isfinite(position_kld).all()
        or float(position_kld.min(initial=0.0)) < -1e-6
    ):
        raise ValueError("KLD position vector differs")
    negative_roundoff_count = int((position_kld < 0.0).sum())
    position_kld = np.maximum(position_kld, 0.0)
    with np.load(args.routes_npz, allow_pickle=False) as loaded:
        routes_all = np.asarray(loaded["routed_experts"], dtype=np.uint8)
    if routes_all.ndim != 3 or routes_all.shape[1:] != (78, 8):
        raise ValueError("source route tensor shape differs")
    routes = routes_all[: position_kld.size, args.layer, :]
    if routes.shape != (position_kld.size, 8):
        raise ValueError("KLD and route position domains differ")
    worst_count = min(args.worst_count, position_kld.size)
    worst_positions = np.argsort(position_kld, kind="stable")[-worst_count:]
    worst_positions = np.asarray(sorted(worst_positions.tolist()), dtype=np.int64)
    worst_kld = position_kld[worst_positions]
    weighted_denominator = float(8.0 * worst_kld.sum(dtype=np.float64))
    if weighted_denominator <= 0.0:
        raise ValueError("worst-position KLD mass is empty")
    bonus = np.asarray(
        [
            float(((routes[worst_positions] == expert) * worst_kld[:, None]).sum(dtype=np.float64))
            for expert in range(EXPERTS)
        ],
        dtype=np.float64,
    )

    candidate_grid: dict[int, list[dict[str, Any]]] = {}
    base_records: dict[int, dict[str, Any]] = {}
    base_k4_count = 0
    for expert in range(EXPERTS):
        record = scores["experts"][str(expert)]
        candidates = [dict(item) for item in record.get("candidates", [])]
        if len(candidates) != 8:
            raise ValueError(f"expert {expert}: complete triplet grid differs")
        rates_seen = {
            tuple(int(item["rates"][name]) for name in PROJECTIONS)
            for item in candidates
        }
        if len(rates_seen) != 8:
            raise ValueError(f"expert {expert}: triplet rate domain differs")
        for item in candidates:
            rates = item.get("rates")
            expected_k4 = sum(int(rates[name]) == 4 for name in PROJECTIONS)
            if int(item.get("k4_count", -1)) != expected_k4:
                raise ValueError(f"expert {expert}: candidate K4 count differs")
            for field in (
                "allocation_fit_objective",
                "fit_allocation_relative_error_sum",
                "fixed_tail_fit_relative_error_sum",
                "fixed_body_fit_relative_error_sum",
                "raw_fit_allocation_sse",
            ):
                value = float(item[field])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"expert {expert}: candidate {field} differs")
        candidate_grid[expert] = candidates
        assignment = base["expert_assignments"][str(expert)]
        base_record = selected_candidate(record, assignment.get("rates"))
        if base_record.get("candidate_id") != assignment.get("candidate_id"):
            raise ValueError(f"expert {expert}: base candidate binding differs")
        base_records[expert] = base_record
        base_k4_count += int(base_record["k4_count"])
    if base_k4_count != 96:
        raise ValueError("base K96 tensor count differs")

    def summarize(selection: dict[int, dict[str, Any]]) -> dict[str, Any]:
        total = math.fsum(
            float(selection[e]["fit_allocation_relative_error_sum"])
            for e in range(EXPERTS)
        )
        body = math.fsum(
            float(selection[e]["fixed_body_fit_relative_error_sum"])
            for e in range(EXPERTS)
        )
        tail = math.fsum(
            float(selection[e]["fixed_tail_fit_relative_error_sum"])
            for e in range(EXPERTS)
        )
        raw = math.fsum(
            float(selection[e]["raw_fit_allocation_sse"])
            for e in range(EXPERTS)
        )
        allocation = math.fsum(
            float(selection[e]["allocation_fit_objective"])
            for e in range(EXPERTS)
        )
        route_mass = math.fsum(
            float(bonus[e]) * int(selection[e]["k4_count"])
            for e in range(EXPERTS)
        )
        path = tuple(str(selection[e]["candidate_id"]) for e in range(EXPERTS))
        return {
            "total": total,
            "body": body,
            "tail": tail,
            "raw": raw,
            "allocation": allocation,
            "route_mass": route_mass,
            "path": path,
        }

    base_summary = summarize(base_records)
    base_total = float(base_summary["total"])
    base_body = float(base_summary["body"])
    if base_total <= 0.0 or base_body <= 0.0:
        raise ValueError("base K96 guard objective is empty")

    # The layer-native candidate domain permits K4 on gate, up, or down and
    # may spend multiple K4 tensors on one expert.  Scalarize exact triplet-DP
    # loss against KLD-weighted route mass per K4 tensor; every probe still
    # closes at exactly 96 K4 tensors, then passes both calibration guards.
    from src.sqg_k34_allocation import TripletCandidateScore, solve_triplet_dp

    scales: list[float] = []
    for expert in range(EXPERTS):
        if bonus[expert] <= 0.0:
            continue
        candidates = candidate_grid[expert]
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                k4_delta = abs(
                    int(candidates[left]["k4_count"])
                    - int(candidates[right]["k4_count"])
                )
                if k4_delta == 0:
                    continue
                objective_delta = abs(
                    float(candidates[left]["allocation_fit_objective"])
                    - float(candidates[right]["allocation_fit_objective"])
                )
                scale = objective_delta / (float(bonus[expert]) * k4_delta)
                if math.isfinite(scale) and scale > 0.0:
                    scales.append(scale)
    if scales:
        low = max(min(scales) / 8.0, max(scales) * 1e-10)
        high = max(scales) * 8.0
        probes = [0.0, *np.geomspace(low, high, num=97).tolist()]
    else:
        probes = [0.0]

    best_selection = dict(base_records)
    best_summary = dict(base_summary)
    best_scalar = 0.0
    best_rank = (
        -float(base_summary["route_mass"]),
        base_total,
        base_body,
        base_summary["path"],
    )
    seen: set[tuple[str, ...]] = {base_summary["path"]}
    valid_selections = 1
    for scalar in probes:
        candidates_by_expert: dict[int, list[TripletCandidateScore]] = {}
        for expert in range(EXPERTS):
            adjusted = [
                float(item["allocation_fit_objective"])
                - scalar * float(bonus[expert]) * int(item["k4_count"])
                for item in candidate_grid[expert]
            ]
            offset = max(0.0, -min(adjusted))
            candidates_by_expert[expert] = [
                TripletCandidateScore(
                    candidate_id=str(item["candidate_id"]),
                    rates=tuple(int(item["rates"][name]) for name in PROJECTIONS),
                    loss=adjusted[index] + offset,
                    record=item,
                )
                for index, item in enumerate(candidate_grid[expert])
            ]
        solved = solve_triplet_dp(
            candidates_by_expert, target_k4=96, expected_experts=EXPERTS
        )
        selection = {
            expert: dict(selected.record or {})
            for expert, selected in solved.selected.items()
        }
        summary = summarize(selection)
        path = summary["path"]
        if path in seen:
            continue
        seen.add(path)
        if (
            float(summary["total"])
            > base_total * (1.0 + args.total_regression_limit) + 1e-12
            or float(summary["body"])
            > base_body * (1.0 + args.body_regression_limit) + 1e-12
        ):
            continue
        valid_selections += 1
        rank = (
            -float(summary["route_mass"]),
            float(summary["total"]),
            float(summary["body"]),
            path,
        )
        if rank < best_rank:
            best_rank = rank
            best_selection = selection
            best_summary = summary
            best_scalar = scalar

    selected_records = best_selection
    selected_total = float(best_summary["total"])
    selected_body = float(best_summary["body"])
    scalar = best_scalar
    assignments: dict[str, Any] = {}
    bit_map: dict[str, int] = {}
    projection_k4 = {name: 0 for name in PROJECTIONS}
    for expert in range(EXPERTS):
        candidate = selected_records[expert]
        assignments[str(expert)] = {
            key: candidate[key]
            for key in (
                "rates",
                "k4_count",
                "candidate_id",
                "raw_fit_allocation_sse",
                "fit_allocation_relative_error_sum",
                "fixed_tail_fit_relative_error_sum",
                "fixed_body_fit_relative_error_sum",
                "allocation_fit_objective",
            )
        }
        for projection in PROJECTIONS:
            bits = int(candidate["rates"][projection])
            bit_map[f"model.layers.{args.layer}.mlp.experts.{expert}.{projection}"] = bits
            projection_k4[projection] += bits == 4
    if sum(projection_k4.values()) != 96:
        raise AssertionError("guarded triplet DP changed the exact K96 budget")

    selected_experts = {
        expert for expert, item in selected_records.items() if int(item["k4_count"]) > 0
    }
    base_selected = {
        expert for expert, item in base_records.items() if int(item["k4_count"]) > 0
    }
    selected_routes = np.isin(
        routes, np.asarray(sorted(selected_experts), dtype=np.uint8)
    )
    worst_route_slots = selected_routes[worst_positions]
    body_mask = np.ones(position_kld.size, dtype=np.bool_)
    body_mask[worst_positions] = False
    in_experts = sorted(selected_experts - base_selected)
    out_experts = sorted(base_selected - selected_experts)
    changed_experts = sorted(
        expert
        for expert in range(EXPERTS)
        if selected_records[expert]["candidate_id"]
        != base_records[expert]["candidate_id"]
    )
    result = dict(base)
    result.update(
        {
            "method": "layer_native_k96_exact_triplet_dp_then_source_worst40_kld_route_mass_with_total_and_body_guards_v2",
            "loss_definition": "source_checkpoint_kld_route_mass_per_k4_tensor_with_layer_native_exact_triplet_candidate_calibration_guards_v2",
            "expert_assignments": assignments,
            "bit_map": dict(sorted(bit_map.items())),
            "projection_k4": projection_k4,
            "objective": {
                **dict(base["objective"]),
                "base_k96_selected_fit_allocation_relative_error_sum": base_total,
                "base_k96_selected_fixed_body_fit_relative_error_sum": base_body,
                "selected_fit_allocation_relative_error_sum": selected_total,
                "selected_fixed_body_fit_relative_error_sum": selected_body,
                "selected_fixed_tail_fit_relative_error_sum": float(best_summary["tail"]),
                "selected_raw_fit_allocation_sse": float(best_summary["raw"]),
                "exact_dp": float(best_summary["allocation"]),
                "relative_total_regression_fraction": selected_total / base_total - 1.0,
                "relative_body_regression_fraction": selected_body / base_body - 1.0,
                "total_regression_limit": args.total_regression_limit,
                "body_regression_limit": args.body_regression_limit,
                "body_guard_pass": True,
            },
            "kld_route_policy": {
                "fit_signal": "sealed_source_checkpoint_tp4_pp1_dcp1_position_kld_and_routes",
                "scalarized_choice": "exact_layer_native_triplet_dp_at_96_k4_tensors",
                "route_mass_definition": "worst40_kld_weighted_route_occurrence_times_candidate_k4_tensor_count",
                "end_to_end_quality_claim": False,
                "final_same_window_kld_is_in_sample_for_allocation": True,
                "worst_count": worst_count,
                "worst_positions": worst_positions.tolist(),
                "worst_kld_sum": float(worst_kld.sum(dtype=np.float64)),
                "source_total_kld_sum": float(position_kld.sum(dtype=np.float64)),
                "negative_kld_roundoff_values_clamped_to_zero": negative_roundoff_count,
                "source_worst_fraction": float(worst_kld.sum(dtype=np.float64) / position_kld.sum(dtype=np.float64)),
                "selected_weighted_route_mass_fraction": float(best_summary["route_mass"] / (3.0 * weighted_denominator)),
                "selected_worst_route_slot_fraction": float(worst_route_slots.mean()),
                "selected_body_route_slot_fraction": float(selected_routes[body_mask].mean()),
                "base_weighted_route_mass_fraction": float(base_summary["route_mass"] / (3.0 * weighted_denominator)),
                "base_worst_route_slot_fraction": float(np.isin(routes[worst_positions], sorted(base_selected)).mean()),
                "base_body_route_slot_fraction": float(np.isin(routes[body_mask], sorted(base_selected)).mean()),
                "scalarization_lambda": scalar,
                "distinct_scalarized_selections_evaluated": len(seen),
                "valid_guarded_selections_evaluated": valid_selections,
                "scalarization_probe_count": len(probes),
                "changed_experts": changed_experts,
                "swapped_in_experts": in_experts,
                "swapped_out_experts": out_experts,
                "source_kld": {"path": str(args.kld_json.resolve()), "sha256": sha256_file(args.kld_json)},
                "source_routes": {"path": str(args.routes_npz.resolve()), "sha256": sha256_file(args.routes_npz)},
                "base_allocation": {"path": str(args.base_allocation.resolve()), "sha256": sha256_file(args.base_allocation), "allocation_id": base["allocation_id"]},
                "score_manifest": {"path": str(args.score_manifest.resolve()), "sha256": sha256_file(args.score_manifest), "score_manifest_id": scores["score_manifest_id"]},
            },
            "production_eligible": True,
        }
    )
    result.pop("allocation_id", None)
    from src.fresh_pipeline_common import atomic_json, canonical_sha256

    result["allocation_id"] = canonical_sha256(result)
    atomic_json(args.output, result)
    print(json.dumps({
        "allocation_id": result["allocation_id"],
        "output": str(args.output.resolve()),
        "weighted_route_mass_fraction": result["kld_route_policy"]["selected_weighted_route_mass_fraction"],
        "swaps": len(changed_experts),
        "total_regression_fraction": result["objective"]["relative_total_regression_fraction"],
        "body_regression_fraction": result["objective"]["relative_body_regression_fraction"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
