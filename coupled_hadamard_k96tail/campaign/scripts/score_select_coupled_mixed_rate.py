#!/usr/bin/env python3
"""Score, select, and hold out coupled mixed-K3/K4 GLM experts.

Draw selection follows the conservative QSRT rule: the fit split may propose
draw 6 over draw 0, the document-disjoint selection split must confirm the
same direction, and otherwise the expert falls back to draw 0.  The frozen
selection manifest is published before any holdout row is read.  Scores use
signed, applied-gate-weighted top-8 expert sums, so cross-expert cancellation
and reinforcement are retained.
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
from typing import Any, Mapping

import numpy as np


NUM_EXPERTS = 256
HIDDEN = 6144
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ARTIFACT_SCHEMA = "glm52-coupled-mixed-k3-k4-expert-v4"
SELECTION_SCHEMA = "glm52-coupled-mixed-rate-selection-v4"
SCORE_SCHEMA = "glm52-coupled-mixed-rate-signed-top8-score-v4"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--qsrt-root", type=Path, required=True)
    parser.add_argument("--source-sqg-root", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--draws", type=int, nargs="+", default=(0, 6))
    parser.add_argument(
        "--phase",
        choices=("all", "draw", "selected"),
        default="all",
        help="run one parallelizable draw score, or freeze/score the selected mix",
    )
    parser.add_argument(
        "--draw",
        type=int,
        choices=(0, 6),
        help="required only for --phase draw",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--chunk-rows", type=int, default=256)
    return parser


def _artifact_path(root: Path, layer: int, draw: int, expert: int) -> Path:
    return (
        root
        / f"layer_{layer:03d}"
        / f"draw_{draw:02d}"
        / "experts"
        / f"layer-{layer:03d}-expert-{expert:03d}.json"
    )


def _load_manifest(
    root: Path,
    layer: int,
    draw: int,
    expert: int,
    *,
    profile_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    path = _artifact_path(root, layer, draw, expert)
    value = json.loads(path.read_text())
    if (
        value.get("schema") != ARTIFACT_SCHEMA
        or value.get("complete") is not True
        or value.get("layer") != layer
        or value.get("expert") != expert
        or value.get("intermediate_draw") != draw
        or value.get("activation") != "silu_gate_times_up"
        or value.get("rate_contract") != "independent_per_tensor_k3_k4"
        or value.get("uniform_k3_candidate") is not False
        or value.get("final_profile_binding") != profile_binding
        or float(value.get("selected_beta", -1.0))
        != float(profile_binding["selected_beta"])
    ):
        raise ValueError(f"coupled artifact binding differs: {path}")
    bits = value.get("bits")
    if not isinstance(bits, dict) or set(bits) != set(PROJECTIONS):
        raise ValueError(f"coupled artifact rate map is malformed: {path}")
    if any(bits[projection] not in (3, 4) for projection in PROJECTIONS):
        raise ValueError(f"coupled artifact contains a non-K3/K4 tensor: {path}")
    shard = path.parent / str(value.get("shard", ""))
    if not shard.is_file():
        raise FileNotFoundError(shard)
    return value, shard


def _decode_expert(
    manifest: Mapping[str, Any],
    shard: Path,
    *,
    device: Any,
    lut_by_bits: Mapping[int, Any],
    safe_open: Any,
    decode_stored_fp16: Any,
    unpack_trellis_states: Any,
    decode_regularized_states: Any,
    NativeProjection: Any,
    tensor_prefix: Any,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    layer = int(manifest["layer"])
    expert = int(manifest["expert"])
    output: dict[str, Any] = {}
    native: dict[str, Any] = {}
    with safe_open(shard, framework="pt", device="cpu") as handle:
        for projection in PROJECTIONS:
            prefix = tensor_prefix(layer, expert, projection)
            bits = int(manifest["bits"][projection])
            marker = handle.get_tensor(f"{prefix}.sqg")
            if int(marker) != 0x53514731:
                raise ValueError(f"{prefix}: SQG marker differs")
            trellis = handle.get_tensor(f"{prefix}.trellis").to(device)
            suh = handle.get_tensor(f"{prefix}.suh").to(device)
            svh = handle.get_tensor(f"{prefix}.svh").to(device)
            output[projection] = decode_stored_fp16(
                trellis,
                suh,
                svh,
                bits=bits,
                codebook_e4m3=lut_by_bits[bits],
            )
            states = unpack_trellis_states(trellis, bits)
            labels = decode_regularized_states(
                states, lut_by_bits[bits]
            ).contiguous()
            if not torch.equal(labels, labels.to(torch.float8_e4m3fn).float()):
                raise RuntimeError(f"{prefix}: labels are not exact finite E4M3")
            native[projection] = NativeProjection(
                weight=labels,
                suh=suh,
                svh=svh,
                bits=bits,
            )
    return output, native


def _reference_output(hidden: Any, weights: Mapping[str, Any], F: Any) -> Any:
    gate = F.linear(hidden, weights["gate_proj"])
    up = F.linear(hidden, weights["up_proj"])
    return F.linear(F.silu(gate) * up, weights["down_proj"])


def _execute_full_w4a8_coupled(
    hidden: Any,
    execution: Any,
    projections: Mapping[str, Any],
    hadamard: Any,
    *,
    prepare_gate_up_operand: Any,
    prepare_down_operand: Any,
    native_label_gemm: Any,
    apply_output_transform: Any,
    torch: Any,
) -> Any:
    """Execute the exact native W4A8 path in updated-QSRT coordinates."""

    gate = projections["gate_proj"]
    up = projections["up_proj"]
    down = projections["down_proj"]
    if not torch.equal(gate.suh, up.suh):
        raise RuntimeError("coupled gate/up input profiles differ")
    transformed_hidden = execution.transform_inputs(hidden)
    operand, observation, _ = prepare_gate_up_operand(
        transformed_hidden, gate.suh, hadamard, quantize_a8=True
    )
    if observation is None or bool(observation.preclamp_overflow.any()):
        raise RuntimeError("coupled h-A8 candidate overflowed")
    gate_output = apply_output_transform(
        native_label_gemm(operand, gate.weight), gate.svh, hadamard
    )
    up_output = apply_output_transform(
        native_label_gemm(operand, up.weight), up.svh, hadamard
    )
    middle = execution.decode_middle_outputs(gate_output, up_output)
    down_operand, down_observation, _ = prepare_down_operand(
        middle, down.suh, hadamard, quantize_a8=True
    )
    if down_observation is None or bool(
        down_observation.preclamp_overflow.any()
    ):
        raise RuntimeError("coupled act-A8 candidate overflowed")
    transformed_output = apply_output_transform(
        native_label_gemm(down_operand, down.weight), down.svh, hadamard
    )
    return execution.decode_output(transformed_output)


def _new_role_accumulator(capture: Any, role: str) -> dict[str, Any]:
    rows = capture.role_rows(role)
    if not rows.size:
        raise ValueError(f"{role} capture rows are empty")
    compact = np.full(capture.rows, -1, dtype=np.int64)
    compact[rows] = np.arange(rows.size, dtype=np.int64)
    return {
        "role_rows": rows,
        "compact": compact,
        "error": np.zeros((rows.size, HIDDEN), dtype=np.float32),
        "reference": np.zeros((rows.size, HIDDEN), dtype=np.float32),
        "individual": {},
    }


def _score_assignment(
    runtime: Any,
    candidate_root: Path,
    assignment: Mapping[int, int],
    *,
    profile_binding: Mapping[str, Any],
    roles: tuple[str, ...],
    device: Any,
    chunk_rows: int,
    lut_by_bits: Mapping[int, Any],
    CoupledHadamardSpec: Any,
    coupled_execution: Any,
    safe_open: Any,
    decode_stored_fp16: Any,
    unpack_trellis_states: Any,
    decode_regularized_states: Any,
    NativeProjection: Any,
    tensor_prefix: Any,
    document_scores_from_routed_aggregates: Any,
    hadamard: Any,
    prepare_gate_up_operand: Any,
    prepare_down_operand: Any,
    native_label_gemm: Any,
    apply_output_transform: Any,
    torch: Any,
    F: Any,
) -> dict[str, Any]:
    accumulators = {
        role: _new_role_accumulator(runtime.capture, role) for role in roles
    }
    artifact_ids: dict[str, str] = {}
    started = time.monotonic()
    for expert in range(NUM_EXPERTS):
        draw = int(assignment[expert])
        manifest, shard = _load_manifest(
            candidate_root,
            runtime.layer,
            draw,
            expert,
            profile_binding=profile_binding,
        )
        artifact_ids[str(expert)] = str(manifest["manifest_id"])
        decoded, native = _decode_expert(
            manifest,
            shard,
            device=device,
            lut_by_bits=lut_by_bits,
            safe_open=safe_open,
            decode_stored_fp16=decode_stored_fp16,
            unpack_trellis_states=unpack_trellis_states,
            decode_regularized_states=decode_regularized_states,
            NativeProjection=NativeProjection,
            tensor_prefix=tensor_prefix,
            torch=torch,
        )
        spec_record = manifest["coupled_transform"]
        spec = CoupledHadamardSpec(
            residual_block_size=int(spec_record["residual_block_size"]),
            preactivation_block_size=int(
                spec_record["preactivation_block_size"]
            ),
            postactivation_block_size=int(
                spec_record["postactivation_block_size"]
            ),
            residual_draw=int(spec_record["residual_draw"]),
            intermediate_draw=int(spec_record["intermediate_draw"]),
            activation="silu",
        )
        weights_tuple = (
            decoded["gate_proj"].T.contiguous(),
            decoded["up_proj"].T.contiguous(),
            decoded["down_proj"].T.contiguous(),
        )
        execution = coupled_execution(weights_tuple, spec)
        source = runtime.source.load_expert_bf16(
            runtime.layer, expert, device=runtime.device
        )
        source_gpu = {
            "gate_proj": source.gate_hf.to(
                device=device, dtype=torch.float32
            ),
            "up_proj": source.up_hf.to(device=device, dtype=torch.float32),
            "down_proj": source.down_hf.to(
                device=device, dtype=torch.float32
            ),
        }
        for role, accumulator in accumulators.items():
            routed = runtime.capture.routed_rows(expert, role)
            numerator = 0.0
            denominator = 0.0
            for begin in range(0, routed.rows, chunk_rows):
                end = min(routed.rows, begin + chunk_rows)
                absolute = routed.row_indices[begin:end]
                positions = accumulator["compact"][absolute]
                if bool((positions < 0).any()):
                    raise RuntimeError("role position map is inconsistent")
                hidden = runtime.capture.load_hidden(
                    absolute, device=device, dtype=torch.float32
                )
                with torch.no_grad():
                    candidate = _execute_full_w4a8_coupled(
                        hidden,
                        execution,
                        native,
                        hadamard,
                        prepare_gate_up_operand=prepare_gate_up_operand,
                        prepare_down_operand=prepare_down_operand,
                        native_label_gemm=native_label_gemm,
                        apply_output_transform=apply_output_transform,
                        torch=torch,
                    )
                    reference = _reference_output(hidden, source_gpu, F)
                    gates = routed.applied_gates[begin:end].to(
                        device=device, dtype=torch.float32
                    )[:, None]
                    delta = candidate - reference
                    numerator += float(
                        torch.sum(
                            delta.square().sum(dim=1) * gates[:, 0].square(),
                            dtype=torch.float64,
                        )
                    )
                    denominator += float(
                        torch.sum(
                            reference.square().sum(dim=1)
                            * gates[:, 0].square(),
                            dtype=torch.float64,
                        )
                    )
                    accumulator["error"][positions] += (
                        delta * gates
                    ).cpu().numpy()
                    accumulator["reference"][positions] += (
                        reference * gates
                    ).cpu().numpy()
                del hidden, candidate, reference, delta, gates
            if not math.isfinite(numerator) or denominator <= 0:
                raise ValueError(
                    f"layer {runtime.layer} expert {expert} {role}: invalid SSE"
                )
            accumulator["individual"][str(expert)] = {
                "draw": draw,
                "routed_rows": routed.rows,
                "error_energy": numerator,
                "reference_energy": denominator,
                "relative_error": numerator / denominator,
            }
        del decoded, native, weights_tuple, execution, source, source_gpu
        gc.collect()
        torch.cuda.empty_cache()
        if (expert + 1) % 16 == 0:
            print(
                f"layer {runtime.layer} score roles={roles}: "
                f"{expert + 1}/{NUM_EXPERTS}",
                flush=True,
            )

    role_reports: dict[str, Any] = {}
    for role, accumulator in accumulators.items():
        energies = document_scores_from_routed_aggregates(
            accumulator["error"],
            accumulator["reference"],
            np.asarray(
                runtime.capture.doc_epochs[accumulator["role_rows"]],
                dtype=np.int64,
            ),
            chunk_rows=chunk_rows,
        )
        role_reports[role] = {
            **energies,
            "role_rows": int(accumulator["role_rows"].size),
            "individual_experts": accumulator["individual"],
            "aggregation": (
                "signed_applied_gate_weighted_top8_sum_then_square_"
                "retains_cross_expert_terms"
            ),
        }
    return {
        "schema": SCORE_SCHEMA,
        "complete": True,
        "layer": runtime.layer,
        "assignment": {str(key): int(value) for key, value in assignment.items()},
        "draw_histogram": {
            str(draw): sum(value == draw for value in assignment.values())
            for draw in sorted(set(assignment.values()))
        },
        "artifact_manifest_ids": artifact_ids,
        "roles": role_reports,
        "elapsed_seconds": time.monotonic() - started,
    }


def _score_draw_portfolio(
    runtime: Any,
    candidate_root: Path,
    draws: tuple[int, ...],
    *,
    profile_binding: Mapping[str, Any],
    roles: tuple[str, ...],
    device: Any,
    chunk_rows: int,
    lut_by_bits: Mapping[int, Any],
    CoupledHadamardSpec: Any,
    coupled_execution: Any,
    safe_open: Any,
    decode_stored_fp16: Any,
    unpack_trellis_states: Any,
    decode_regularized_states: Any,
    NativeProjection: Any,
    tensor_prefix: Any,
    document_scores_from_routed_aggregates: Any,
    hadamard: Any,
    prepare_gate_up_operand: Any,
    prepare_down_operand: Any,
    native_label_gemm: Any,
    apply_output_transform: Any,
    torch: Any,
    F: Any,
) -> dict[int, dict[str, Any]]:
    """Score all coupled draws in one source/capture pass per expert."""

    accumulators = {
        draw: {role: _new_role_accumulator(runtime.capture, role) for role in roles}
        for draw in draws
    }
    artifact_ids = {draw: {} for draw in draws}
    started = time.monotonic()
    for expert in range(NUM_EXPERTS):
        candidates: dict[int, tuple[Any, dict[str, Any]]] = {}
        for draw in draws:
            manifest, shard = _load_manifest(
                candidate_root,
                runtime.layer,
                draw,
                expert,
                profile_binding=profile_binding,
            )
            artifact_ids[draw][str(expert)] = str(manifest["manifest_id"])
            decoded, native = _decode_expert(
                manifest,
                shard,
                device=device,
                lut_by_bits=lut_by_bits,
                safe_open=safe_open,
                decode_stored_fp16=decode_stored_fp16,
                unpack_trellis_states=unpack_trellis_states,
                decode_regularized_states=decode_regularized_states,
                NativeProjection=NativeProjection,
                tensor_prefix=tensor_prefix,
                torch=torch,
            )
            spec_record = manifest["coupled_transform"]
            spec = CoupledHadamardSpec(
                residual_block_size=int(spec_record["residual_block_size"]),
                preactivation_block_size=int(
                    spec_record["preactivation_block_size"]
                ),
                postactivation_block_size=int(
                    spec_record["postactivation_block_size"]
                ),
                residual_draw=int(spec_record["residual_draw"]),
                intermediate_draw=int(spec_record["intermediate_draw"]),
                activation="silu",
            )
            weights = (
                decoded["gate_proj"].T.contiguous(),
                decoded["up_proj"].T.contiguous(),
                decoded["down_proj"].T.contiguous(),
            )
            candidates[draw] = (coupled_execution(weights, spec), native)
            del decoded, weights

        source = runtime.source.load_expert_bf16(
            runtime.layer, expert, device=runtime.device
        )
        source_gpu = {
            "gate_proj": source.gate_hf.to(device=device, dtype=torch.float32),
            "up_proj": source.up_hf.to(device=device, dtype=torch.float32),
            "down_proj": source.down_hf.to(device=device, dtype=torch.float32),
        }
        for role in roles:
            routed = runtime.capture.routed_rows(expert, role)
            for begin in range(0, routed.rows, chunk_rows):
                end = min(routed.rows, begin + chunk_rows)
                absolute = routed.row_indices[begin:end]
                hidden = runtime.capture.load_hidden(
                    absolute, device=device, dtype=torch.float32
                )
                reference = _reference_output(hidden, source_gpu, F)
                gates = routed.applied_gates[begin:end].to(
                    device=device, dtype=torch.float32
                )[:, None]
                for draw in draws:
                    accumulator = accumulators[draw][role]
                    positions = accumulator["compact"][absolute]
                    if bool((positions < 0).any()):
                        raise RuntimeError("role position map is inconsistent")
                    execution, native = candidates[draw]
                    with torch.no_grad():
                        candidate = _execute_full_w4a8_coupled(
                            hidden,
                            execution,
                            native,
                            hadamard,
                            prepare_gate_up_operand=prepare_gate_up_operand,
                            prepare_down_operand=prepare_down_operand,
                            native_label_gemm=native_label_gemm,
                            apply_output_transform=apply_output_transform,
                            torch=torch,
                        )
                        delta = candidate - reference
                        importance = gates[:, 0].square()
                        numerator = float(
                            torch.sum(
                                delta.square().sum(dim=1) * importance,
                                dtype=torch.float64,
                            )
                        )
                        denominator = float(
                            torch.sum(
                                reference.square().sum(dim=1) * importance,
                                dtype=torch.float64,
                            )
                        )
                        accumulator["error"][positions] += (
                            delta * gates
                        ).cpu().numpy()
                        accumulator["reference"][positions] += (
                            reference * gates
                        ).cpu().numpy()
                    individual = accumulator["individual"].setdefault(
                        str(expert),
                        {
                            "draw": draw,
                            "routed_rows": 0,
                            "error_energy": 0.0,
                            "reference_energy": 0.0,
                        },
                    )
                    individual["routed_rows"] += int(end - begin)
                    individual["error_energy"] += numerator
                    individual["reference_energy"] += denominator
                    del candidate, delta, importance
                del hidden, reference, gates
        for draw in draws:
            for role in roles:
                individual = accumulators[draw][role]["individual"][str(expert)]
                denominator = float(individual["reference_energy"])
                if denominator <= 0 or not math.isfinite(denominator):
                    raise ValueError(
                        f"layer {runtime.layer} expert {expert} {role}: invalid SSE"
                    )
                individual["relative_error"] = (
                    float(individual["error_energy"]) / denominator
                )
        del candidates, source, source_gpu
        gc.collect()
        torch.cuda.empty_cache()
        if (expert + 1) % 16 == 0:
            print(
                f"layer {runtime.layer} portfolio draws={draws} roles={roles}: "
                f"{expert + 1}/{NUM_EXPERTS}",
                flush=True,
            )

    reports: dict[int, dict[str, Any]] = {}
    for draw in draws:
        role_reports: dict[str, Any] = {}
        for role, accumulator in accumulators[draw].items():
            energies = document_scores_from_routed_aggregates(
                accumulator["error"],
                accumulator["reference"],
                np.asarray(
                    runtime.capture.doc_epochs[accumulator["role_rows"]],
                    dtype=np.int64,
                ),
                chunk_rows=chunk_rows,
            )
            role_reports[role] = {
                **energies,
                "role_rows": int(accumulator["role_rows"].size),
                "individual_experts": accumulator["individual"],
                "aggregation": (
                    "signed_applied_gate_weighted_top8_sum_then_square_"
                    "retains_cross_expert_terms"
                ),
            }
        reports[draw] = {
            "schema": SCORE_SCHEMA,
            "complete": True,
            "layer": runtime.layer,
            "assignment": {str(expert): draw for expert in range(NUM_EXPERTS)},
            "draw_histogram": {str(draw): NUM_EXPERTS},
            "artifact_manifest_ids": artifact_ids[draw],
            "roles": role_reports,
            "elapsed_seconds": time.monotonic() - started,
            "portfolio_scored_in_one_source_capture_pass": True,
        }
    return reports


def _select_draws(
    draw_reports: Mapping[int, Mapping[str, Any]],
    *,
    draws: tuple[int, ...],
) -> tuple[dict[int, int], dict[str, Any]]:
    if draws != (0, 6):
        raise ValueError("the preregistered conservative draw portfolio is (0,6)")
    selected: dict[int, int] = {}
    decisions: dict[str, Any] = {}
    for expert in range(NUM_EXPERTS):
        fit = {
            draw: float(
                draw_reports[draw]["roles"]["fit"]["individual_experts"][
                    str(expert)
                ]["relative_error"]
            )
            for draw in draws
        }
        confirmation = {
            draw: float(
                draw_reports[draw]["roles"]["selection"][
                    "individual_experts"
                ][str(expert)]["relative_error"]
            )
            for draw in draws
        }
        proposed = 6 if fit[6] < fit[0] else 0
        accepted = proposed == 6 and confirmation[6] < confirmation[0]
        chosen = 6 if accepted else 0
        selected[expert] = chosen
        decisions[str(expert)] = {
            "fit_relative_error": {str(key): value for key, value in fit.items()},
            "selection_relative_error": {
                str(key): value for key, value in confirmation.items()
            },
            "proposed_draw": proposed,
            "selected_draw": chosen,
            "nonzero_draw_required_fit_and_selection_win": True,
        }
    return selected, decisions


def main() -> int:
    args = _parser().parse_args()
    if args.threads <= 0 or args.chunk_rows <= 0:
        raise ValueError("thread and chunk counts must be positive")
    draws = tuple(dict.fromkeys(args.draws))
    if draws != (0, 6):
        raise ValueError("this registered test requires --draws 0 6")
    if (args.phase == "draw") != (args.draw is not None):
        raise ValueError("--draw is required exactly when --phase draw")
    project_root = Path(__file__).resolve().parents[1]
    qsrt_root = args.qsrt_root.resolve()
    for root in (project_root, qsrt_root, project_root / "kquant"):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(args.threads)

    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from kquant.sqg_e4m3 import sqg_xor_cheb_t12_bytes
    from qsrt.qsrt_coupled import CoupledHadamardSpec, coupled_execution
    from scripts.encode_coupled_mixed_rate_shard import _load_3p0625_bit_map
    from scripts.score_glm52_w4a8_activation_quality import (
        NativeProjection,
        apply_output_transform,
        native_label_gemm,
        prepare_down_operand,
        prepare_gate_up_operand,
    )
    from src.fresh_pipeline_common import (
        atomic_json,
        canonical_sha256,
        sha256_file,
        tensor_prefix,
    )
    from src.fresh_pipeline_evaluation import document_scores_from_routed_aggregates
    from src.glm52_fresh_sqg.reference import (
        decode_regularized_states,
        decode_stored_fp16,
        normalized_hadamard,
        unpack_trellis_states,
    )
    from src.sqg_checkpoint_source import open_sqg_transcode_runtime

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("exact routed scoring requires CUDA")
    allocation_path = args.allocation.resolve()
    allocation_payload = json.loads(allocation_path.read_text())
    runtime = open_sqg_transcode_runtime(
        args.preflight,
        model_root=args.source_sqg_root,
        layer=args.layer,
        device=args.device,
        bit_map=_load_3p0625_bit_map(allocation_path, layer=args.layer),
        probe_rows=1024,
    )
    allocation_binding = {
        "file": allocation_path.name,
        "schema": allocation_payload.get("schema"),
        "sha256": sha256_file(allocation_path),
        "allocation_id": allocation_payload.get("allocation_id"),
        "histogram": allocation_payload.get("histogram"),
        "bpw": allocation_payload.get("bpw"),
    }
    if not isinstance(allocation_binding["allocation_id"], str):
        raise ValueError("coupled allocation lacks a sealed allocation_id")
    profile_binding = allocation_payload.get("final_profile_binding")
    selected_beta = float(allocation_payload.get("selected_beta", -1.0))
    if (
        not isinstance(profile_binding, dict)
        or profile_binding.get("schema")
        != "glm52-updated-qsrt-coupled-final-profile-binding-v1"
        or profile_binding.get("complete") is not True
        or profile_binding.get("no_b300_owner_speed_rescue") is not True
        or float(profile_binding.get("selected_beta", -2.0)) != selected_beta
    ):
        raise ValueError("allocation no-shortcut final profile binding differs")
    candidate_root = args.candidate_root.resolve()
    layer_root = candidate_root / f"layer_{runtime.layer:03d}"
    lut_by_bits = {
        bits: sqg_xor_cheb_t12_bytes(bits).to(device).contiguous()
        for bits in (3, 4)
    }
    hadamard = normalized_hadamard(device=device, dtype=torch.float32, size=128)

    draw_reports: dict[int, dict[str, Any]] = {}
    requested_draws = (int(args.draw),) if args.phase == "draw" else draws
    draw_paths = {
        draw: layer_root / f"draw_{draw:02d}_fit_selection_score.json"
        for draw in requested_draws
    }
    if args.phase == "all" and not any(path.exists() for path in draw_paths.values()):
        portfolio = _score_draw_portfolio(
            runtime,
            candidate_root,
            draws,
            profile_binding=profile_binding,
            roles=("fit", "selection"),
            device=device,
            chunk_rows=args.chunk_rows,
            lut_by_bits=lut_by_bits,
            CoupledHadamardSpec=CoupledHadamardSpec,
            coupled_execution=coupled_execution,
            safe_open=safe_open,
            decode_stored_fp16=decode_stored_fp16,
            unpack_trellis_states=unpack_trellis_states,
            decode_regularized_states=decode_regularized_states,
            NativeProjection=NativeProjection,
            tensor_prefix=tensor_prefix,
            document_scores_from_routed_aggregates=(
                document_scores_from_routed_aggregates
            ),
            hadamard=hadamard,
            prepare_gate_up_operand=prepare_gate_up_operand,
            prepare_down_operand=prepare_down_operand,
            native_label_gemm=native_label_gemm,
            apply_output_transform=apply_output_transform,
            torch=torch,
            F=F,
        )
        for draw, report in portfolio.items():
            report["allocation_binding"] = allocation_binding
            report["selected_beta"] = selected_beta
            report["final_profile_binding"] = profile_binding
            report["score_id"] = canonical_sha256(report)
            atomic_json(draw_paths[draw], report)

    for draw in requested_draws:
        path = draw_paths[draw]
        if path.is_file():
            report = json.loads(path.read_text())
            if (
                report.get("schema") != SCORE_SCHEMA
                or report.get("complete") is not True
                or report.get("layer") != runtime.layer
                or set(report.get("roles", {})) != {"fit", "selection"}
                or report.get("allocation_binding") != allocation_binding
                or report.get("final_profile_binding") != profile_binding
            ):
                raise ValueError(f"existing draw score differs: {path}")
        else:
            report = _score_assignment(
                runtime,
                candidate_root,
                {expert: draw for expert in range(NUM_EXPERTS)},
                profile_binding=profile_binding,
                roles=("fit", "selection"),
                device=device,
                chunk_rows=args.chunk_rows,
                lut_by_bits=lut_by_bits,
                CoupledHadamardSpec=CoupledHadamardSpec,
                coupled_execution=coupled_execution,
                safe_open=safe_open,
                decode_stored_fp16=decode_stored_fp16,
                unpack_trellis_states=unpack_trellis_states,
                decode_regularized_states=decode_regularized_states,
                NativeProjection=NativeProjection,
                tensor_prefix=tensor_prefix,
                document_scores_from_routed_aggregates=(
                    document_scores_from_routed_aggregates
                ),
                hadamard=hadamard,
                prepare_gate_up_operand=prepare_gate_up_operand,
                prepare_down_operand=prepare_down_operand,
                native_label_gemm=native_label_gemm,
                apply_output_transform=apply_output_transform,
                torch=torch,
                F=F,
            )
            report["allocation_binding"] = allocation_binding
            report["selected_beta"] = selected_beta
            report["final_profile_binding"] = profile_binding
            report["score_id"] = canonical_sha256(report)
            atomic_json(path, report)
        draw_reports[draw] = report

    if args.phase == "draw":
        print(
            json.dumps(
                {
                    "schema": "glm52-coupled-mixed-rate-draw-result-v1",
                    "complete": True,
                    "layer": runtime.layer,
                    "draw": args.draw,
                    "score_id": draw_reports[int(args.draw)]["score_id"],
                    "roles": {
                        role: draw_reports[int(args.draw)]["roles"][role][
                            "aggregate_relative_error"
                        ]
                        for role in ("fit", "selection")
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.phase == "selected":
        # The two draw arms were scored concurrently on separate GPUs.  Load
        # and validate both frozen reports before publishing selection.
        draw_reports = {}
        for draw in draws:
            path = layer_root / f"draw_{draw:02d}_fit_selection_score.json"
            report = json.loads(path.read_text())
            if (
                report.get("schema") != SCORE_SCHEMA
                or report.get("complete") is not True
                or report.get("layer") != runtime.layer
                or set(report.get("roles", {})) != {"fit", "selection"}
                or report.get("allocation_binding") != allocation_binding
                or report.get("final_profile_binding") != profile_binding
                or report.get("assignment")
                != {str(expert): draw for expert in range(NUM_EXPERTS)}
            ):
                raise ValueError(f"parallel draw score differs: {path}")
            draw_reports[draw] = report

    assignment, decisions = _select_draws(draw_reports, draws=draws)
    selection_path = layer_root / "coupled_selection_manifest.json"
    selection: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "complete": True,
        "layer": runtime.layer,
        "draw_portfolio": list(draws),
        "selection_policy": (
            "fit_proposes_nonzero_draw_and_disjoint_selection_must_confirm_"
            "otherwise_draw0"
        ),
        "selected_draws": {str(key): value for key, value in assignment.items()},
        "draw_histogram": {
            str(draw): sum(value == draw for value in assignment.values())
            for draw in draws
        },
        "decisions": decisions,
        "holdout_used": False,
        "uniform_k3_candidate": False,
        "rate_contract": "independent_per_tensor_k3_k4",
        "allocation_binding": allocation_binding,
        "selected_beta": selected_beta,
        "final_profile_binding": profile_binding,
        "source_score_ids": {
            str(draw): str(draw_reports[draw]["score_id"]) for draw in draws
        },
    }
    selection["selection_id"] = canonical_sha256(selection)
    if selection_path.is_file():
        existing_selection = json.loads(selection_path.read_text())
        if existing_selection != selection:
            raise ValueError("existing coupled selection manifest differs")
    else:
        atomic_json(selection_path, selection)

    final_path = layer_root / "selected_selection_holdout_score.json"
    if final_path.is_file():
        final = json.loads(final_path.read_text())
        if (
            final.get("schema") != SCORE_SCHEMA
            or final.get("complete") is not True
            or final.get("selection_id") != selection["selection_id"]
            or final.get("allocation_binding") != allocation_binding
            or final.get("final_profile_binding") != profile_binding
        ):
            raise ValueError("existing selected score differs")
    else:
        final = _score_assignment(
            runtime,
            candidate_root,
            assignment,
            profile_binding=profile_binding,
            roles=("selection", "holdout"),
            device=device,
            chunk_rows=args.chunk_rows,
            lut_by_bits=lut_by_bits,
            CoupledHadamardSpec=CoupledHadamardSpec,
            coupled_execution=coupled_execution,
            safe_open=safe_open,
            decode_stored_fp16=decode_stored_fp16,
            unpack_trellis_states=unpack_trellis_states,
            decode_regularized_states=decode_regularized_states,
            NativeProjection=NativeProjection,
            tensor_prefix=tensor_prefix,
            document_scores_from_routed_aggregates=(
                document_scores_from_routed_aggregates
            ),
            hadamard=hadamard,
            prepare_gate_up_operand=prepare_gate_up_operand,
            prepare_down_operand=prepare_down_operand,
            native_label_gemm=native_label_gemm,
            apply_output_transform=apply_output_transform,
            torch=torch,
            F=F,
        )
        final["selection_id"] = selection["selection_id"]
        final["selection_manifest"] = selection_path.name
        final["allocation_binding"] = allocation_binding
        final["selected_beta"] = selected_beta
        final["final_profile_binding"] = profile_binding
        final["score_id"] = canonical_sha256(final)
        atomic_json(final_path, final)

    summary = {
        "schema": "glm52-coupled-mixed-rate-layer-result-v2",
        "complete": True,
        "layer": runtime.layer,
        "selection_id": selection["selection_id"],
        "draw_histogram": selection["draw_histogram"],
        "uniform_k3_candidate": False,
        "rate_contract": "independent_per_tensor_k3_k4",
        "allocation_binding": allocation_binding,
        "selected_beta": selected_beta,
        "final_profile_binding": profile_binding,
        "draw_scores": {
            str(draw): {
                role: draw_reports[draw]["roles"][role][
                    "aggregate_relative_error"
                ]
                for role in ("fit", "selection")
            }
            for draw in draws
        },
        "selected_score": {
            role: final["roles"][role]["aggregate_relative_error"]
            for role in ("selection", "holdout")
        },
        "selection_path": str(selection_path),
        "score_path": str(final_path),
    }
    atomic_json(layer_root / "coupled_result_summary.json", summary, overwrite=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
