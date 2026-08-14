#!/usr/bin/env python3
"""Score layer-native coupled K3/K4 triplets with a fixed fit-tail objective.

Worker mode reconstructs one frozen-checkpoint expert, applies the updated
QSRT H512/H128/H128 coupled coordinates at draw zero, encodes all eight
gate/up/down K3/K4 triplets with candidate-conditioned ``(H, B)`` down
targets, and executes the native direct-E4M3 W4A8 path.  Candidate fitting
uses only fit/calibration documents; row scoring uses the disjoint
fit/allocation documents and never reads selection or holdout data.

Finalize mode first sums the all-K3 unary errors and source reference energy
by routed position, freezes the worst two percent by relative unary error, then
solves an exact-budget multiple-choice DP on whole-universe relative error
plus a fixed-tail relative-error penalty.  It also reports a worst-forty-document
control, but that diagnostic cannot participate in selection.  The default is
the original 48-K4 target; the same measurements can also produce 72/96-K4
budget-curve controls without re-encoding candidates.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "kquant"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

EXPERTS = 256
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
RATE_TRIPLETS = tuple(
    (gate, up, down)
    for gate in (3, 4)
    for up in (3, 4)
    for down in (3, 4)
)
ALL_K3_INDEX = RATE_TRIPLETS.index((3, 3, 3))
EXPERT_SCHEMA = "glm52-coupled-tail-triplet-expert-scores-v5"
ALLOCATION_SCHEMA = "glm52-coupled-tail-aware-layer-native-allocation-v7"
LOSS_DEFINITION = (
    "fit_allocation_realized_coupled_full_w4a8_relative_unary_error_plus_"
    "fixed_all_k3_worst_2pct_position_relative_unary_error_with_body_guard_v6"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(root: Path, layer: int, expert: int) -> tuple[Path, Path]:
    directory = root / f"layer_{layer:03d}" / "experts"
    stem = f"expert_{expert:03d}"
    return directory / f"{stem}.json", directory / f"{stem}.row-sse.npz"


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _all_k3_map(layer: int) -> dict[str, int]:
    return {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}": 3
        for expert in range(EXPERTS)
        for projection in PROJECTIONS
    }


def _execute_upstream(
    hidden: Any,
    execution: Any,
    native: dict[tuple[int, int], dict[str, Any]],
    hadamard: Any,
    prepare_gate_up_operand: Any,
    native_label_gemm: Any,
    apply_output_transform: Any,
) -> dict[tuple[int, int], Any]:
    transformed = execution.transform_inputs(hidden)
    first = next(iter(native.values()))["gate_proj"]
    operand, observation, _ = prepare_gate_up_operand(
        transformed, first.suh, hadamard, quantize_a8=True
    )
    if observation is None or bool(observation.preclamp_overflow.any()):
        raise RuntimeError("coupled h-A8 candidate overflowed")
    products: dict[int, Any] = {}
    result: dict[tuple[int, int], Any] = {}
    for pair, projections in native.items():
        gate = projections["gate_proj"]
        up = projections["up_proj"]
        for projection in (gate, up):
            if id(projection) not in products:
                products[id(projection)] = native_label_gemm(
                    operand, projection.weight
                )
        gate_output = apply_output_transform(
            products[id(gate)], gate.svh, hadamard
        )
        up_output = apply_output_transform(products[id(up)], up.svh, hadamard)
        result[pair] = execution.decode_middle_outputs(gate_output, up_output)
    return result


def _execute_down(
    middle: Any,
    down: Any,
    execution: Any,
    hadamard: Any,
    prepare_down_operand: Any,
    native_label_gemm: Any,
    apply_output_transform: Any,
) -> Any:
    operand, observation, _ = prepare_down_operand(
        middle, down.suh, hadamard, quantize_a8=True
    )
    if observation is None or bool(observation.preclamp_overflow.any()):
        raise RuntimeError("coupled act-A8 candidate overflowed")
    transformed = apply_output_transform(
        native_label_gemm(operand, down.weight), down.svh, hadamard
    )
    return execution.decode_output(transformed)


def _fit_and_encode_coupled_down_grid(
    *,
    runtime: Any,
    execution: Any,
    routed: Any,
    calibration_mask: np.ndarray,
    native_pairs: dict[tuple[int, int], dict[str, Any]],
    preliminary_down: dict[tuple[int, int], dict[int, Any]],
    down_configs: dict[int, Any],
    official_down_exl: Any,
    source_gpu: dict[str, Any],
    beta: float,
    expert: int,
    device: Any,
    hadamard: Any,
    chunk_rows: int,
    kquant_runtime: Any,
    lut: dict[int, Any],
    prepare_down_operand: Any,
    prepare_gate_up_operand: Any,
    native_label_gemm: Any,
    apply_output_transform: Any,
    apply_frozen_h2_shrinkage: Any,
    canonical_sha256: Any,
    tensor_sha256: Any,
    DenseHessian: Any,
    SharedResidualProfile: Any,
    SyntheticTensorBinding: Any,
    encode_uniform_sqg: Any,
    realize_shared_residual_profile: Any,
    effective_canonical_operand_from_quantized_transform: Any,
    kquant_prefinalize_operand_from_label_operand: Any,
    CrossTermStatistics: Any,
    shrink_cross_term_objective: Any,
    solve_with_minimal_official_prior: Any,
    native_projection: Any,
    down_bits_domain: tuple[int, ...] = (3, 4),
) -> tuple[
    dict[tuple[int, int], dict[int, Any]],
    dict[tuple[int, int], dict[int, Any]],
    dict[tuple[int, int], dict[int, dict[str, Any]]],
]:
    """Fit exact coupled-coordinate candidate down targets on fit/calibration."""

    import torch
    import torch.nn.functional as F

    pairs = tuple(sorted(native_pairs))
    expected_pairs = {(gate, up) for gate in (3, 4) for up in (3, 4)}
    if not pairs or any(pair not in expected_pairs for pair in pairs):
        raise ValueError("coupled down grid has an invalid upstream pair domain")
    down_bits_domain = tuple(dict.fromkeys(down_bits_domain))
    if not down_bits_domain or any(bits not in (3, 4) for bits in down_bits_domain):
        raise ValueError("coupled down grid has an invalid down-rate domain")
    cells = tuple((*pair, down) for pair in pairs for down in down_bits_domain)
    if set(preliminary_down) != set(pairs):
        raise ValueError("coupled down grid preliminary/upstream pair domains differ")
    rows = routed.row_indices[calibration_mask]
    importance_all = routed.gate_square_weights[
        torch.from_numpy(calibration_mask)
    ]
    if rows.size == 0:
        raise ValueError(f"layer {runtime.layer} expert {expert}: empty calibration split")

    anchor_profiles: dict[tuple[int, int, int], Any] = {}
    anchor_vectors: dict[tuple[int, int, int], Any] = {}
    for pair in pairs:
        for bits in down_bits_domain:
            anchor = SharedResidualProfile.from_stored_vector(
                preliminary_down[pair][bits].suh,
                side="input",
                profile_id=(
                    f"layer-{runtime.layer:03d}/expert-{expert:03d}/"
                    f"coupled-{pair[0]}{pair[1]}-down-k{bits}-anchor-v2"
                ),
                derivation=(
                    "fit_preliminary_coupled_h2_frozen_sqg_source_down_encode"
                ),
            )
            realized = realize_shared_residual_profile(anchor, runtime.device)
            cell = (*pair, bits)
            anchor_profiles[cell] = realized
            anchor_vectors[cell] = realized.expected_stored_fp16().to(device)

    raw_h = {
        cell: torch.zeros(
            (official_down_exl.shape[0], official_down_exl.shape[0]),
            dtype=torch.float32,
            device=device,
        )
        for cell in cells
    }
    raw_encoder_h = {cell: torch.zeros_like(raw_h[cell]) for cell in cells}
    raw_b = {
        cell: torch.zeros(
            official_down_exl.shape, dtype=torch.float32, device=device
        )
        for cell in cells
    }
    denominator = torch.zeros((), dtype=torch.float64, device=device)
    teacher_energy = torch.zeros((), dtype=torch.float64, device=device)
    importance_chunks: list[Any] = []
    for begin in range(0, rows.size, chunk_rows):
        end = min(rows.size, begin + chunk_rows)
        hidden = runtime.capture.load_hidden(
            rows[begin:end], device=device, dtype=torch.float32
        )
        middle_by_pair = _execute_upstream(
            hidden,
            execution,
            native_pairs,
            hadamard,
            prepare_gate_up_operand,
            native_label_gemm,
            apply_output_transform,
        )
        gate = torch.matmul(hidden.float(), source_gpu["gate_proj"])
        up = torch.matmul(hidden.float(), source_gpu["up_proj"])
        teacher = torch.matmul(F.silu(gate) * up, source_gpu["down_proj"])
        transformed_teacher = execution.transform_outputs(teacher)
        importance = importance_all[begin:end].to(
            device=device, dtype=torch.float32
        )
        importance_chunks.append(importance.cpu())
        for pair, middle in middle_by_pair.items():
            for bits in down_bits_domain:
                cell = (*pair, bits)
                anchor = anchor_profiles[cell]
                q_label, observation, _ = prepare_down_operand(
                    middle,
                    anchor_vectors[cell],
                    hadamard,
                    quantize_a8=True,
                )
                if observation is None or bool(observation.preclamp_overflow.any()):
                    raise RuntimeError(
                        f"coupled candidate-specific down K{bits} act-A8 overflowed"
                    )
                q_eff = effective_canonical_operand_from_quantized_transform(
                    q_label,
                    anchor_vectors[cell],
                    hadamard,
                    validate_values=False,
                )
                q_pre = kquant_prefinalize_operand_from_label_operand(
                    q_label,
                    anchor.signs,
                    hadamard,
                    validate_values=False,
                )
                raw_h[cell].addmm_(q_eff.T, q_eff * importance[:, None])
                raw_b[cell].addmm_(
                    q_eff.T, transformed_teacher * importance[:, None]
                )
                raw_encoder_h[cell].addmm_(
                    q_pre.T, q_pre * importance[:, None]
                )
                del q_label, observation, q_eff, q_pre
        denominator.add_(importance.double().sum())
        teacher_energy.add_(
            torch.sum(
                transformed_teacher.square().sum(dim=1) * importance,
                dtype=torch.float64,
            )
        )
        del (
            hidden,
            middle_by_pair,
            gate,
            up,
            teacher,
            transformed_teacher,
            importance,
        )
    if not bool(torch.isfinite(denominator)) or not bool(denominator > 0):
        raise ValueError("coupled candidate H/B has no positive routed mass")

    official = official_down_exl.to(device=device, dtype=torch.float32).contiguous()
    importance_vector = torch.cat(importance_chunks)
    encoded: dict[tuple[int, int], dict[int, Any]] = {pair: {} for pair in pairs}
    native: dict[tuple[int, int], dict[int, Any]] = {pair: {} for pair in pairs}
    evidence_grid: dict[tuple[int, int], dict[int, dict[str, Any]]] = {
        pair: {} for pair in pairs
    }
    for gate_bits, up_bits, bits in cells:
        cell = (gate_bits, up_bits, bits)
        for matrix in (raw_h[cell], raw_b[cell], raw_encoder_h[cell]):
            matrix.div_(denominator.to(torch.float32))
        canonical_h_raw = raw_h.pop(cell)
        canonical_h = ((canonical_h_raw + canonical_h_raw.T) * 0.5).contiguous()
        cross_term = raw_b.pop(cell).contiguous()
        encoder_h_raw = raw_encoder_h.pop(cell)
        encoder_h_raw = (
            (encoder_h_raw + encoder_h_raw.T) * 0.5
        ).contiguous()
        identity_scale = canonical_h.diagonal().double().mean()
        stats = CrossTermStatistics(
            hessian=canonical_h,
            cross_term=cross_term,
            weight_sum=denominator,
            rows=int(rows.size),
            validate_values=False,
        )
        fitted = shrink_cross_term_objective(
            stats,
            official,
            local_alpha=beta,
            identity_scale=identity_scale,
        )
        target, solver = solve_with_minimal_official_prior(
            fitted,
            official,
            identity_scale=identity_scale,
        )
        encoder_h, encoder_shrinkage = apply_frozen_h2_shrinkage(
            encoder_h_raw, importance_vector
        )
        evidence: dict[str, Any] = {
            "schema": "glm52-coupled-candidate-conditioned-down-objective-v2",
            "layer": runtime.layer,
            "expert": expert,
            "gate_bits": gate_bits,
            "up_bits": up_bits,
            "down_bits": bits,
            "role": "fit",
            "subfold": "calibration",
            "beta": beta,
            "rows": int(rows.size),
            "gate_square_sum": float(denominator),
            "teacher_energy": float(teacher_energy),
            "teacher_coordinate": "updated_qsrt_coupled_residual_output_h512",
            "operand_coordinate": "updated_qsrt_coupled_postactivation_h128",
            "objective": "candidate_conditioned_full_w4a8_H_B",
            "solver": solver,
            "encoder_h_shrinkage": {
                key: float(value) if isinstance(value, (int, float)) else value
                for key, value in encoder_shrinkage.items()
            },
            "selection_used": False,
            "holdout_used": False,
        }
        evidence_id = canonical_sha256(evidence)
        evidence["evidence_id"] = evidence_id
        binding = SyntheticTensorBinding(
            tensor_name=down_configs[bits].source_binding.tensor_name,
            tensor_sha256=tensor_sha256(target),
            fixture_id=(
                "frozen_sqg_function_source_candidate_conditioned_coupled_"
                f"down_h_b_v2/{evidence_id}"
            ),
        )
        config = replace(
            down_configs[bits],
            source_binding=binding,
            anchored_input_residual_profile=anchor_profiles[cell],
        )
        dense = DenseHessian(
            matrix=encoder_h.contiguous(),
            evidence_id=evidence_id,
            construction=(
                "fit_calibration_candidate_specific_coupled_full_w4a8_"
                "cross_term_qpre_encoder_h_v2"
            ),
            split_id="fit",
            normalization_count=1,
            routed_sample_count=int(rows.size),
        )
        result = encode_uniform_sqg(
            target.contiguous(), dense, config, runtime=kquant_runtime
        )
        anchor = anchor_profiles[cell]
        if not torch.equal(result.suh, anchor.expected_stored_fp16()):
            raise RuntimeError("candidate-conditioned down encode changed anchored suh")
        pair = (gate_bits, up_bits)
        encoded[pair][bits] = result
        native[pair][bits] = native_projection(
            result, bits=bits, lut=lut[bits], device=device
        )
        evidence_grid[pair][bits] = evidence
        del (
            canonical_h_raw,
            canonical_h,
            cross_term,
            encoder_h_raw,
            encoder_h,
            target,
            dense,
        )
    return encoded, native, evidence_grid


def _worker(args: argparse.Namespace) -> None:
    import torch
    from scripts.coupled_recipe_core import (
        build_profile_global_h13,
        candidate_payloads,
        fit_coupled_down,
        load_final_recipe_profile,
        prepare_coupled_expert,
        score_coupled_candidates,
    )
    from scripts.score_sqg_w4a8_triplet_candidates import (
        subfold_contract,
    )
    from src.fresh_pipeline_common import atomic_json
    from src.sqg_checkpoint_source import open_sqg_transcode_runtime

    qsrt_root = args.qsrt_root.resolve()
    if str(qsrt_root) not in sys.path:
        sys.path.insert(0, str(qsrt_root))
    runtime = open_sqg_transcode_runtime(
        args.preflight,
        model_root=args.source_sqg_root,
        layer=args.layer,
        device=args.device,
        bit_map=_all_k3_map(args.layer),
        probe_rows=1024,
    )
    gate_profile, down_profile, profile_selection, beta, profile_binding = (
        load_final_recipe_profile(
            runtime,
            selection_path=args.profile_selection,
            binding_path=args.profile_binding,
            layer=args.layer,
        )
    )
    global_h13, global_h13_evidence = build_profile_global_h13(
        runtime,
        gate_profile,
        cell_id=str(profile_selection["selected_cell_id"]),
        chunk_rows=args.chunk_rows,
    )

    for expert in range(args.start, args.end):
        manifest_path, score_path = _paths(args.output_root.resolve(), args.layer, expert)
        if manifest_path.is_file() and score_path.is_file():
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                value.get("schema") != EXPERT_SCHEMA
                or value.get("complete") is not True
                or value.get("layer") != args.layer
                or value.get("expert") != expert
                or value.get("row_sse_sha256") != _sha256_file(score_path)
                or value.get("profile_selection", {}).get("selection_id")
                != profile_selection["selection_id"]
                or value.get("final_profile_binding", {}).get("binding_id")
                != profile_binding["binding_id"]
                or float(value.get("beta", -1.0)) != beta
            ):
                raise ValueError(f"coupled score resume binding differs: {manifest_path}")
            continue
        if manifest_path.exists() or score_path.exists():
            raise ValueError(f"partial coupled score artifact exists: {manifest_path}")

        prepared = prepare_coupled_expert(
            runtime,
            expert=expert,
            gate_profile=gate_profile,
            down_profile=down_profile,
            global_h13=global_h13,
            global_evidence=global_h13_evidence,
            triplets=RATE_TRIPLETS,
            qsrt_root=qsrt_root,
            chunk_rows=args.chunk_rows,
        )
        encoded_down, native_down, down_evidence = fit_coupled_down(
            prepared, beta=beta
        )
        score = score_coupled_candidates(
            prepared,
            native_down,
            score_role="fit/allocation",
            return_rows=True,
        )
        row_indices = score.pop("row_indices")
        document_epochs = score.pop("document_epochs")
        scores = score.pop("row_sse")
        reference_energy = score.pop("row_reference_energy")
        _atomic_npz(
            score_path,
            row_indices=row_indices,
            document_epochs=document_epochs,
            row_sse=scores,
            row_reference_energy=reference_energy,
        )
        payloads_by_triplet = candidate_payloads(prepared, encoded_down)
        candidates = []
        for index, rates in enumerate(RATE_TRIPLETS):
            pair = rates[:2]
            payloads = payloads_by_triplet[str(rates)]
            preliminary_id = prepared.preliminary_h2_evidence[
                f"k{pair[0]}_k{pair[1]}"
            ]["evidence_id"]
            material = {
                "layer": args.layer,
                "expert": expert,
                "rates": dict(zip(PROJECTIONS, rates, strict=True)),
                "payload_sha256": payloads,
                "draw": 0,
                "beta": beta,
                "final_profile_binding_id": profile_binding["binding_id"],
                "preliminary_h2_evidence_id": preliminary_id,
                "down_objective_evidence_id": down_evidence[pair][rates[2]][
                    "evidence_id"
                ],
            }
            candidates.append(
                {
                    "candidate_id": _canonical_sha256(material),
                    "rates": material["rates"],
                    "k4_count": sum(rate == 4 for rate in rates),
                    "payload_sha256": payloads,
                    "preliminary_h2_evidence_id": preliminary_id,
                    "down_objective_evidence_id": down_evidence[pair][rates[2]][
                        "evidence_id"
                    ],
                    "raw_fit_allocation_sse": float(
                        scores[index].sum(dtype=np.float64)
                    ),
                    "fit_allocation_reference_energy": float(
                        reference_energy.sum(dtype=np.float64)
                    ),
                    "full_w4a8_realized": True,
                }
            )
        manifest: dict[str, Any] = {
            "schema": EXPERT_SCHEMA,
            "complete": True,
            "layer": args.layer,
            "expert": expert,
            "role": "fit",
            "calibration_subfold": "calibration",
            "allocation_subfold": "allocation",
            "subfold_contract": subfold_contract(args.layer),
            "selection_used": False,
            "holdout_used": False,
            "draw": 0,
            "beta": beta,
            "coupled_transform": {
                "residual_hadamard": 512,
                "preactivation_hadamard": 128,
                "postactivation_hadamard": 128,
                "activation": "silu",
            },
            "profile_selection": profile_selection,
            "final_profile_binding": dict(profile_binding),
            "profile_global_h13_evidence": global_h13_evidence,
            "coupled_h13_evidence": prepared.coupled_h13_evidence,
            "row_count": int(row_indices.size),
            "allocation_document_count": int(np.unique(document_epochs).size),
            "calibration_row_count": int(prepared.calibration_mask.sum()),
            "preliminary_h2_evidence": prepared.preliminary_h2_evidence,
            "row_sse": score_path.name,
            "row_sse_sha256": _sha256_file(score_path),
            "row_reference_energy_present": True,
            "candidates": candidates,
            "source": runtime.source.validation.manifest(),
        }
        manifest["score_id"] = _canonical_sha256(manifest)
        atomic_json(manifest_path, manifest)
        print(
            f"layer {args.layer} expert {expert}: coupled tail grid complete",
            flush=True,
        )
        del encoded_down, native_down, down_evidence, scores, reference_energy
        prepared.release()
        gc.collect()
        torch.cuda.empty_cache()


def _finalize(args: argparse.Namespace) -> None:
    from src.fresh_pipeline_common import atomic_json
    from src.sqg_k34_allocation import TripletCandidateScore, solve_triplet_dp

    root = args.output_root.resolve()
    records: dict[int, dict[str, Any]] = {}
    arrays: dict[
        int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    baseline_by_position: dict[int, float] = {}
    reference_by_position: dict[int, float] = {}
    baseline_by_document: dict[int, float] = {}
    reference_by_document: dict[int, float] = {}
    final_profile_binding: dict[str, Any] | None = None
    profile_selection: dict[str, Any] | None = None
    for expert in range(EXPERTS):
        manifest_path, score_path = _paths(root, args.layer, expert)
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            value.get("schema") != EXPERT_SCHEMA
            or value.get("complete") is not True
            or value.get("layer") != args.layer
            or value.get("expert") != expert
            or value.get("selection_used") is not False
            or value.get("holdout_used") is not False
            or value.get("row_sse_sha256") != _sha256_file(score_path)
        ):
            raise ValueError(f"expert {expert}: coupled score binding differs")
        current_binding = value.get("final_profile_binding")
        current_selection = value.get("profile_selection")
        if (
            not isinstance(current_binding, dict)
            or current_binding.get("schema")
            != "glm52-updated-qsrt-coupled-final-profile-binding-v1"
            or current_binding.get("complete") is not True
            or current_binding.get("no_b300_owner_speed_rescue") is not True
            or not isinstance(current_selection, dict)
            or current_selection.get("selection_id")
            != current_binding.get("profile_selection_id")
        ):
            raise ValueError(f"expert {expert}: no-shortcut profile binding differs")
        if final_profile_binding is None:
            final_profile_binding = current_binding
            profile_selection = current_selection
        elif (
            current_binding != final_profile_binding
            or current_selection != profile_selection
        ):
            raise ValueError("expert score profile bindings are not layer-global")
        calibration_row_count = value.get("calibration_row_count")
        allocation_row_count = value.get("row_count")
        if (
            not isinstance(calibration_row_count, int)
            or isinstance(calibration_row_count, bool)
            or calibration_row_count <= 0
            or not isinstance(allocation_row_count, int)
            or isinstance(allocation_row_count, bool)
            or allocation_row_count <= 0
        ):
            raise ValueError(f"expert {expert}: fit split row counts differ")
        preliminary_h2 = value.get("preliminary_h2_evidence")
        expected_h2_pairs = {"k3_k3", "k3_k4", "k4_k3", "k4_k4"}
        if not isinstance(preliminary_h2, dict) or set(preliminary_h2) != expected_h2_pairs:
            raise ValueError(f"expert {expert}: preliminary H2 evidence set differs")
        for pair, evidence in preliminary_h2.items():
            expected_lineage = {
                "schema": "glm52-coupled-candidate-h2-v2",
                "layer": args.layer,
                "expert": expert,
                "role": "fit/calibration",
                "fit_only": True,
                "row_mask_applied": True,
                "routed_selected_row_count": calibration_row_count,
                "routed_parent_row_count": calibration_row_count + allocation_row_count,
                "selection_used": False,
                "holdout_used": False,
            }
            if not isinstance(evidence, dict) or any(
                evidence.get(key) != expected
                for key, expected in expected_lineage.items()
            ):
                raise ValueError(
                    f"expert {expert}: preliminary H2 {pair} lineage differs"
                )
            routed = evidence.get("routed")
            if (
                not isinstance(routed, dict)
                or routed.get("role") != "fit"
                or routed.get("expert") != expert
                or routed.get("rows")
                != calibration_row_count + allocation_row_count
            ):
                raise ValueError(
                    f"expert {expert}: preliminary H2 {pair} routed binding differs"
                )
            evidence_id = evidence.get("evidence_id")
            if not isinstance(evidence_id, str) or len(evidence_id) != 64:
                raise ValueError(
                    f"expert {expert}: preliminary H2 {pair} ID differs"
                )
        for candidate in value.get("candidates", []):
            rates = candidate.get("rates", {})
            pair = f"k{rates.get('gate_proj')}_k{rates.get('up_proj')}"
            evidence = preliminary_h2.get(pair)
            if (
                evidence is None
                or candidate.get("preliminary_h2_evidence_id")
                != evidence.get("evidence_id")
            ):
                raise ValueError(
                    f"expert {expert}: candidate preliminary H2 binding differs"
                )
        with np.load(score_path, allow_pickle=False) as loaded:
            rows = np.asarray(loaded["row_indices"], dtype=np.int64)
            documents = np.asarray(loaded["document_epochs"], dtype=np.int64)
            scores = np.asarray(loaded["row_sse"], dtype=np.float64)
            reference = np.asarray(
                loaded["row_reference_energy"], dtype=np.float64
            )
        if scores.shape != (8, rows.size) or rows.size != int(value["row_count"]):
            raise ValueError(f"expert {expert}: row score shape differs")
        if reference.shape != (rows.size,) or documents.shape != (rows.size,):
            raise ValueError(f"expert {expert}: row reference shape differs")
        if (
            not np.isfinite(scores).all()
            or (scores < 0).any()
            or not np.isfinite(reference).all()
            or (reference < 0).any()
        ):
            raise ValueError(f"expert {expert}: row score values differ")
        for row, document, loss, energy in zip(
            rows.tolist(),
            documents.tolist(),
            scores[ALL_K3_INDEX].tolist(),
            reference.tolist(),
            strict=True,
        ):
            baseline_by_position[row] = baseline_by_position.get(row, 0.0) + loss
            reference_by_position[row] = reference_by_position.get(row, 0.0) + energy
            baseline_by_document[document] = (
                baseline_by_document.get(document, 0.0) + loss
            )
            reference_by_document[document] = (
                reference_by_document.get(document, 0.0) + energy
            )
        records[expert] = value
        arrays[expert] = (rows, documents, scores, reference)

    if set(baseline_by_position) != set(reference_by_position):
        raise ValueError("baseline and reference position universes differ")
    if final_profile_binding is None or profile_selection is None:
        raise ValueError("layer score set has no final profile binding")
    if set(baseline_by_document) != set(reference_by_document):
        raise ValueError("baseline and reference document universes differ")
    scale_by_position = {
        row: max(reference_by_position[row], np.finfo(np.float64).tiny)
        for row in baseline_by_position
    }
    baseline_relative_by_position = {
        row: baseline_by_position[row] / scale_by_position[row]
        for row in baseline_by_position
    }
    ordered_positions = sorted(
        baseline_relative_by_position.items(),
        key=lambda item: (-item[1], item[0]),
    )
    tail_count = max(1, math.ceil(args.tail_fraction * len(ordered_positions)))
    tail_positions = {row for row, _ in ordered_positions[:tail_count]}
    baseline_total_sse = math.fsum(baseline_by_position.values())
    baseline_total_relative = math.fsum(
        baseline_relative_by_position.values()
    )
    baseline_tail_relative = math.fsum(
        loss for _, loss in ordered_positions[:tail_count]
    )
    scale_by_document = {
        document: max(reference_by_document[document], np.finfo(np.float64).tiny)
        for document in baseline_by_document
    }
    baseline_relative_by_document = {
        document: baseline_by_document[document] / scale_by_document[document]
        for document in baseline_by_document
    }
    ordered_documents = sorted(
        baseline_relative_by_document.items(),
        key=lambda item: (-item[1], item[0]),
    )
    document_tail_count = min(args.diagnostic_tail_document_count, len(ordered_documents))
    document_tail_relative = math.fsum(
        loss for _, loss in ordered_documents[:document_tail_count]
    )
    candidates_by_expert: dict[int, list[TripletCandidateScore]] = {}
    scored_records: dict[str, Any] = {}
    for expert in range(EXPERTS):
        rows, _documents, scores, _ = arrays[expert]
        row_scales = np.asarray(
            [scale_by_position[int(row)] for row in rows], dtype=np.float64
        )
        relative_scores = scores / row_scales[None, :]
        tail_mask = np.fromiter(
            (int(row) in tail_positions for row in rows),
            dtype=np.bool_,
            count=rows.size,
        )
        candidates: list[dict[str, Any]] = []
        dp_candidates: list[TripletCandidateScore] = []
        for index, raw in enumerate(records[expert]["candidates"]):
            raw_sse = float(scores[index].sum(dtype=np.float64))
            total = float(relative_scores[index].sum(dtype=np.float64))
            tail = float(relative_scores[index, tail_mask].sum(dtype=np.float64))
            body = total - tail
            objective = total + args.tail_weight * tail
            candidate = {
                **raw,
                "raw_fit_allocation_sse": raw_sse,
                "fit_allocation_relative_error_sum": total,
                "fixed_tail_fit_relative_error_sum": tail,
                "fixed_body_fit_relative_error_sum": body,
                "allocation_fit_objective": objective,
                "tail_weight": args.tail_weight,
            }
            rates = tuple(int(candidate["rates"][name]) for name in PROJECTIONS)
            dp_candidates.append(
                TripletCandidateScore(
                    candidate_id=candidate["candidate_id"],
                    rates=rates,
                    loss=objective,
                    record=candidate,
                )
            )
            candidates.append(candidate)
        candidates_by_expert[expert] = dp_candidates
        scored_records[str(expert)] = {
            "expert": expert,
            "candidates": candidates,
            "score_id": records[expert]["score_id"],
        }

    solved = solve_triplet_dp(
        candidates_by_expert, target_k4=args.target_k4, expected_experts=EXPERTS
    )
    raw_candidates_by_expert = {
        expert: [
            TripletCandidateScore(
                candidate_id=item.candidate_id,
                rates=item.rates,
                loss=float(
                    (item.record or {})["fit_allocation_relative_error_sum"]
                ),
                record=item.record,
            )
            for item in candidates
        ]
        for expert, candidates in candidates_by_expert.items()
    }
    raw_solved = solve_triplet_dp(
        raw_candidates_by_expert,
        target_k4=args.target_k4,
        expected_experts=EXPERTS,
    )
    body_candidates_by_expert = {
        expert: [
            TripletCandidateScore(
                candidate_id=item.candidate_id,
                rates=item.rates,
                loss=float(
                    (item.record or {})["fixed_body_fit_relative_error_sum"]
                ),
                record=item.record,
            )
            for item in candidates
        ]
        for expert, candidates in candidates_by_expert.items()
    }
    body_solved = solve_triplet_dp(
        body_candidates_by_expert,
        target_k4=args.target_k4,
        expected_experts=EXPERTS,
    )
    assignments: dict[str, Any] = {}
    bit_map: dict[str, int] = {}
    projection_k4 = {name: 0 for name in PROJECTIONS}
    selected_raw_sse = 0.0
    selected_relative = 0.0
    selected_tail = 0.0
    selected_body = 0.0
    for expert, selected in solved.selected.items():
        record = dict(selected.record or {})
        rates = selected.rate_map
        selected_raw_sse += float(record["raw_fit_allocation_sse"])
        selected_relative += float(record["fit_allocation_relative_error_sum"])
        selected_tail += float(record["fixed_tail_fit_relative_error_sum"])
        selected_body += float(record["fixed_body_fit_relative_error_sum"])
        assignments[str(expert)] = {
            "rates": rates,
            "k4_count": selected.k4_count,
            "candidate_id": selected.candidate_id,
            "raw_fit_allocation_sse": record["raw_fit_allocation_sse"],
            "fit_allocation_relative_error_sum": record[
                "fit_allocation_relative_error_sum"
            ],
            "fixed_tail_fit_relative_error_sum": record[
                "fixed_tail_fit_relative_error_sum"
            ],
            "fixed_body_fit_relative_error_sum": record[
                "fixed_body_fit_relative_error_sum"
            ],
            "allocation_fit_objective": record["allocation_fit_objective"],
        }
        for projection, bits in rates.items():
            bit_map[
                f"model.layers.{args.layer}.mlp.experts.{expert}.{projection}"
            ] = bits
            projection_k4[projection] += bits == 4
    rates_flat = list(bit_map.values())
    expected_k3 = 768 - args.target_k4
    expected_bit_units = expected_k3 * 3 + args.target_k4 * 4
    if (
        rates_flat.count(3),
        rates_flat.count(4),
        sum(rates_flat),
    ) != (expected_k3, args.target_k4, expected_bit_units):
        raise AssertionError("tail-aware allocation changed the exact rate budget")
    total_regression = (
        (selected_relative / raw_solved.objective - 1.0)
        if raw_solved.objective > 0.0
        else (0.0 if selected_relative == 0.0 else math.inf)
    )
    body_regression = (
        (selected_body / body_solved.objective - 1.0)
        if body_solved.objective > 0.0
        else (0.0 if selected_body == 0.0 else math.inf)
    )
    body_guard_pass = (
        total_regression <= args.body_regression_limit
        and body_regression <= args.body_regression_limit
    )

    allocation: dict[str, Any] = {
        "schema": ALLOCATION_SCHEMA,
        "complete": True,
        "production_eligible": body_guard_pass,
        "layer": args.layer,
        "method": (
            "coupled_draw0_fit_calibration_candidate_conditioned_h_b_"
            "fit_allocation_fixed_relative_tail_exact_triplet_dp"
        ),
        "loss_definition": LOSS_DEFINITION,
        "role": "fit",
        "selection_used": False,
        "holdout_used": False,
        "selected_beta": float(final_profile_binding["selected_beta"]),
        "final_profile_binding": final_profile_binding,
        "profile_selection": profile_selection,
        "tail_policy": {
            "baseline": (
                "all_k3_draw0_coupled_native_full_w4a8_relative_unary_error"
            ),
            "fraction": args.tail_fraction,
            "position_universe": len(ordered_positions),
            "fixed_tail_positions": tail_count,
            "tail_weight": args.tail_weight,
            "tail_ranking": (
                "summed_all_k3_routed_unary_sse_divided_by_summed_source_"
                "expert_output_energy_per_routed_capture_position"
            ),
            "reference_energy_floor": np.finfo(np.float64).tiny,
            "fit_subfold": "allocation",
            "sealed_wikitext_worst40_used": False,
            "baseline_total_unary_sse": baseline_total_sse,
            "baseline_total_relative_error_sum": baseline_total_relative,
            "baseline_fixed_tail_relative_error_sum": baseline_tail_relative,
            "baseline_tail_fraction_of_relative_total": (
                baseline_tail_relative / baseline_total_relative
                if baseline_total_relative > 0.0
                else 0.0
            ),
            "diagnostic_worst_documents": {
                "participates_in_selection": False,
                "requested_count": args.diagnostic_tail_document_count,
                "document_universe": len(ordered_documents),
                "actual_count": document_tail_count,
                "baseline_relative_error_sum": document_tail_relative,
                "baseline_fraction_of_document_relative_total": (
                    document_tail_relative
                    / math.fsum(baseline_relative_by_document.values())
                    if baseline_relative_by_document
                    else 0.0
                ),
            },
        },
        "expert_assignments": assignments,
        "bit_map": dict(sorted(bit_map.items())),
        "histogram": {"3": expected_k3, "4": args.target_k4},
        "bit_units": expected_bit_units,
        "bpw": expected_bit_units / 768.0,
        "projection_k4": projection_k4,
        "objective": {
            "exact_dp": solved.objective,
            "selected_raw_fit_allocation_sse": selected_raw_sse,
            "selected_fit_allocation_relative_error_sum": selected_relative,
            "selected_fixed_tail_fit_relative_error_sum": selected_tail,
            "selected_fixed_body_fit_relative_error_sum": selected_body,
            "raw_only_exact_dp_optimum": raw_solved.objective,
            "body_only_exact_dp_optimum": body_solved.objective,
            "relative_total_regression_fraction": total_regression,
            "relative_body_regression_fraction": body_regression,
            "body_regression_limit": args.body_regression_limit,
            "body_guard_pass": body_guard_pass,
        },
        "coupled_transform": {
            "residual_hadamard": 512,
            "preactivation_hadamard": 128,
            "postactivation_hadamard": 128,
            "activation": "silu",
            "draw": 0,
        },
    }
    allocation["allocation_id"] = _canonical_sha256(allocation)
    allocation_path = args.allocation_output.resolve()
    atomic_json(allocation_path, allocation)
    score_manifest = {
        "schema": "glm52-coupled-tail-triplet-layer-scores-v1",
        "complete": True,
        "layer": args.layer,
        "loss_definition": LOSS_DEFINITION,
        "role": "fit",
        "selection_used": False,
        "holdout_used": False,
        "selected_beta": allocation["selected_beta"],
        "final_profile_binding": final_profile_binding,
        "profile_selection": profile_selection,
        "tail_policy": allocation["tail_policy"],
        "experts": scored_records,
        "allocation_path": str(allocation_path),
        "allocation_sha256": _sha256_file(allocation_path),
    }
    score_manifest["score_manifest_id"] = _canonical_sha256(score_manifest)
    score_path = (
        root
        / f"layer_{args.layer:03d}"
        / f"tail_triplet_scores.k4-{args.target_k4:03d}.json"
    )
    atomic_json(score_path, score_manifest)
    print(json.dumps(allocation, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--profile-selection", type=Path)
    parser.add_argument("--profile-binding", type=Path)
    parser.add_argument("--source-sqg-root", type=Path)
    parser.add_argument("--qsrt-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allocation-output", type=Path)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--tail-fraction", type=float, default=0.02)
    parser.add_argument("--diagnostic-tail-document-count", type=int, default=40)
    parser.add_argument("--tail-weight", type=float, default=1.0)
    parser.add_argument("--body-regression-limit", type=float, default=0.01)
    parser.add_argument("--target-k4", type=int, default=48)
    parser.add_argument("--finalize", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not 3 <= args.layer <= 78:
        raise ValueError("routed layer must lie in 3..78")
    if not 0.0 < args.tail_fraction < 1.0:
        raise ValueError("tail fraction must lie strictly between zero and one")
    if args.diagnostic_tail_document_count <= 0:
        raise ValueError("diagnostic tail document count must be positive")
    if not math.isfinite(args.tail_weight) or args.tail_weight < 0.0:
        raise ValueError("tail weight must be finite and nonnegative")
    if (
        not math.isfinite(args.body_regression_limit)
        or args.body_regression_limit < 0.0
    ):
        raise ValueError("body regression limit must be finite and nonnegative")
    if not 0 <= args.target_k4 <= 768:
        raise ValueError("target K4 count must lie in 0..768")
    if args.finalize:
        if args.allocation_output is None:
            raise ValueError("finalize mode requires --allocation-output")
        _finalize(args)
        return
    required = {
        "preflight": args.preflight,
        "profile_selection": args.profile_selection,
        "profile_binding": args.profile_binding,
        "source_sqg_root": args.source_sqg_root,
        "qsrt_root": args.qsrt_root,
        "start": args.start,
        "end": args.end,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"worker mode missing required arguments: {missing}")
    if not 0 <= args.start < args.end <= EXPERTS:
        raise ValueError("worker range must satisfy 0 <= start < end <= 256")
    if os.environ.get("FRESH_SQG_RANK_PRIVATE_TRITON") != "1":
        raise ValueError("FRESH_SQG_RANK_PRIVATE_TRITON=1 is required")
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(args.threads)
    import torch

    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("coupled native scoring requires CUDA")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    _worker(args)
    print(
        json.dumps(
            {"complete": True, "elapsed_seconds": time.monotonic() - started},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
