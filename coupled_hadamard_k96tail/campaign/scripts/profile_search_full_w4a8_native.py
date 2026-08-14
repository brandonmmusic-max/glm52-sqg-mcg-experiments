#!/usr/bin/env python3
"""Select a topology-shared profile using the exact full-W4A8 GLM path.

Profile selection precedes the independent per-tensor K3/K4 allocation.  It
therefore must not inherit the production MCG rate map or optimize a uniform
K3 surrogate.  Each profile is compared with two complementary triplets:

* gate K3, up K3, down K4;
* gate K4, up K4, down K3.

Their equal-weight mixture is exactly 3.5 bpw, assigns each projection K3 and
K4 equally often, and costs two candidate-conditioned down encodes rather than
the eight needed by final triplet allocation.  Eight experts, selected from the
octiles of the existing fit-mass-stratified 16-expert panel, are used only to
choose the topology-shared profile.  Selection rows choose the profile;
holdout is an immutable report-only rerun after ``selection.json`` is sealed.

Every candidate uses the exact arithmetic required by the final model:
profile-specific h-A8 H13, 75% layer-global/25% expert-local H13, native E4M3
gate/up labels, GLM SiLU(gate)*up, act-A8, and caller-coordinate anchored (H,B)
down fitting.  Construction uses the final scorer's document-wise
``fit/calibration`` subfold; ``fit/allocation`` remains untouched for the
later v3 allocator.  No MCG payload, transform, scale, or rate assignment is
read.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "kquant"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts.score_glm52_w4a8_activation_quality import (  # noqa: E402
    NativeProjection,
    prepare_down_operand,
    prepare_gate_up_operand,
)
from scripts.score_sqg_w4a8_triplet_candidates import (  # noqa: E402
    DERIVED_H2_CONSTRUCTION,
    H13_CONSTRUCTION,
    INTERMEDIATE,
    LOCAL_H13_ALPHA,
    NUM_EXPERTS,
    PRELIMINARY_H2_CONSTRUCTION,
    _build_expert_h13,
    _config_at_rate,
    _native_projection,
    _preliminary_down_anchors,
    _projection_payload_sha256,
    _teacher_output,
    build_coupled_upstream_candidates,
    execute_full_w4a8_down,
    execute_full_w4a8_upstream_pairs,
    fit_subfold_mask,
    gate_square_weighted_complete_expert_sse,
    validate_frozen_sse_vector,
    validate_route_gates_once,
)
from scripts.w4a8_cross_term import (  # noqa: E402
    CrossTermStatistics,
    effective_canonical_operand_from_quantized_transform,
    kquant_prefinalize_operand_from_label_operand,
    shrink_cross_term_objective,
)
from scripts.w4a8_stable_solve import solve_with_minimal_official_prior  # noqa: E402


SCHEMA = "glm52-full-w4a8-native-profile-selection-v1"
PREREG_SCHEMA = "glm52-full-w4a8-native-profile-preregistration-v1"
EXPERT_SCORE_SCHEMA = "glm52-full-w4a8-native-profile-expert-score-v1"
CELL_SCORE_SCHEMA = "glm52-full-w4a8-native-profile-cell-score-v1"
HOLDOUT_SCHEMA = "glm52-full-w4a8-native-profile-holdout-report-v1"
GLOBAL_H13_SCHEMA = "glm52-full-w4a8-native-profile-global-h13-v1"
ACTIVE_DRAWS = 4
SCREEN_PANEL_POSITIONS = (0, 2, 4, 6, 9, 11, 13, 15)
HIDDEN = 6144
HADAMARD_BLOCK = 128


def balanced_profile_triplets() -> tuple[tuple[int, int, int], ...]:
    """Return the fixed, projection-balanced 3.5-bpw profile objective."""

    triplets = ((3, 3, 4), (4, 4, 3))
    for projection in range(3):
        if sum(cell[projection] for cell in triplets) / len(triplets) != 3.5:
            raise AssertionError("profile objective is not projection-rate-neutral")
    if sum(sum(cell) for cell in triplets) / len(triplets) != 10.5:
        raise AssertionError("profile objective is not exactly 3.5 bpw")
    return triplets


def reduced_mass_panel(full_panel: Sequence[int]) -> tuple[int, ...]:
    """Choose one deterministic representative from each mass octile."""

    values = tuple(int(value) for value in full_panel)
    if len(values) != 16 or len(set(values)) != 16:
        raise ValueError("profile search requires the sealed 16-expert fit panel")
    selected = tuple(values[position] for position in SCREEN_PANEL_POSITIONS)
    if len(set(selected)) != 8 or any(not 0 <= expert < NUM_EXPERTS for expert in selected):
        raise ValueError("reduced fit-mass panel differs")
    return selected


def active_cells(cells: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    selected = tuple(dict(cell) for cell in cells if int(cell["draw"]) < ACTIVE_DRAWS)
    ids = tuple(str(cell["cell_id"]) for cell in selected)
    if len(selected) != 16 or len(set(ids)) != 16:
        raise ValueError("W4A8 profile search requires 16 active profile cells")
    return selected


def assigned_cell_ids(
    cells: Sequence[Mapping[str, Any]], worker_index: int
) -> tuple[str, ...]:
    if worker_index not in range(4):
        raise ValueError("worker index must be 0, 1, 2, or 3")
    values = active_cells(cells)[worker_index::4]
    if len(values) != 4:
        raise ValueError("profile cells do not partition four ways")
    return tuple(str(cell["cell_id"]) for cell in values)


def choose_profile_cell(cell_objectives: Mapping[str, float]) -> str:
    """Choose only from finite selection-role objectives, with a stable tie."""

    if len(cell_objectives) != 16:
        raise ValueError("all 16 active profile cells are required")
    normalized: dict[str, float] = {}
    for cell_id, objective in cell_objectives.items():
        value = float(objective)
        if not cell_id or not math.isfinite(value) or value < 0:
            raise ValueError("profile selection objective is invalid")
        normalized[str(cell_id)] = value
    return min(normalized, key=lambda cell_id: (normalized[cell_id], cell_id))


def validate_cell_score(
    value: Mapping[str, Any],
    *,
    layer: int,
    role: str,
    cell: Mapping[str, Any],
    panel: Sequence[int],
) -> None:
    """Fail closed over a resumable cell score before it can affect choice."""

    from src.fresh_pipeline_common import canonical_sha256

    if (
        value.get("schema") != CELL_SCORE_SCHEMA
        or value.get("complete") is not True
        or int(value.get("layer", -1)) != layer
        or value.get("role") != role
        or value.get("cell") != cell
        or value.get("selection_panel") != list(panel)
        or value.get("balanced_triplets")
        != [list(item) for item in balanced_profile_triplets()]
        or value.get("selection_used_for_construction") is not False
        or value.get("holdout_used_for_choice") is not False
        or int(value.get("mcg_inputs", -1)) != 0
    ):
        raise ValueError("W4A8 profile cell score binding differs")
    relative_error = float(value.get("aggregate_relative_error", math.nan))
    if not math.isfinite(relative_error) or relative_error < 0:
        raise ValueError("W4A8 profile cell score objective differs")
    body = dict(value)
    score_id = body.pop("score_id", None)
    if score_id != canonical_sha256(body):
        raise ValueError("W4A8 profile cell score ID differs")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--full-build-binding", type=Path, required=True)
    parser.add_argument("--beta-choice", type=Path)
    parser.add_argument(
        "--bootstrap-beta-prior",
        action="store_true",
        help="run the one preregistered beta=0.0625 bootstrap profile search",
    )
    parser.add_argument(
        "--owner-fixed-beta-rescue",
        action="store_true",
        help="run the explicit beta=0.25 paid-node identity rescue search",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize", action="store_true")
    mode.add_argument("--worker-index", type=int)
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--finalize-identity-rescue", action="store_true")
    mode.add_argument("--holdout-report", action="store_true")
    return parser


def _configure(args: argparse.Namespace) -> None:
    from scripts.full_w4a8_build_contract import (
        BOOTSTRAP_BETA,
        validate_beta_choice,
        validate_source_binding,
    )

    if not 3 <= args.layer <= 78:
        raise ValueError("profile search layer must be routed and in [3,78]")
    if args.layer == 78:
        if args.preflight is None:
            raise ValueError("MTP78 profile search requires its MTP-native preflight")
        preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
        source = preflight.get("source_seal")
        bit_contract = preflight.get("bit_contract")
        if not isinstance(source, Mapping) or not isinstance(bit_contract, Mapping):
            raise ValueError("MTP78 profile preflight lacks source/bit bindings")
        source_path = Path(str(source.get("path", ""))).resolve()
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError("MTP78 profile source seal is absent or unsafe")
        source_value = json.loads(source_path.read_text(encoding="utf-8"))
        if (
            source_value.get("layers") != [78]
            or set(bit_contract.get("sanitized_layers", {})) != {"78"}
        ):
            raise ValueError("layer 78 profile search requires an MTP-native singleton binding")
    if args.threads <= 0 or args.chunk_rows <= 0:
        raise ValueError("threads and chunk rows must be positive")
    validate_source_binding(args.full_build_binding, project_root=PROJECT_ROOT)
    if args.bootstrap_beta_prior and args.owner_fixed_beta_rescue:
        raise ValueError("profile search beta modes are mutually exclusive")
    if args.owner_fixed_beta_rescue:
        if args.beta_choice is not None or args.beta != 0.25:
            raise ValueError("owner-fixed rescue requires bare beta=0.25")
        args.beta_lineage = {
            "beta_mode": "owner_fixed_speed_recovery",
            "bootstrap_beta_prior": BOOTSTRAP_BETA,
            "beta_choice": None,
        }
    elif args.bootstrap_beta_prior:
        if args.beta_choice is not None:
            raise ValueError("bootstrap profile search cannot consume a beta choice")
        if args.beta is None or float(args.beta) != BOOTSTRAP_BETA:
            raise ValueError("bootstrap profile search beta must be exactly 0.0625")
        args.beta_lineage = {
            "beta_mode": "bootstrap_prior",
            "bootstrap_beta_prior": BOOTSTRAP_BETA,
            "beta_choice": None,
        }
    else:
        if args.beta_choice is None:
            raise ValueError("final profile search requires --beta-choice")
        evidence = validate_beta_choice(
            args.beta_choice,
            project_root=PROJECT_ROOT,
            build_binding_path=args.full_build_binding,
            layer=args.layer,
        )
        if args.beta is not None and float(args.beta) != evidence["selected_beta"]:
            raise ValueError("bare beta differs from the frozen per-layer beta choice")
        args.beta = evidence["selected_beta"]
        args.beta_lineage = {
            "beta_mode": "frozen_per_layer_choice",
            "bootstrap_beta_prior": BOOTSTRAP_BETA,
            "beta_choice": {
                "sha256": evidence["sha256"],
                "choice_id": evidence["choice_id"],
                "layer": args.layer,
                "selected_beta": evidence["selected_beta"],
                "full_build_binding_id": evidence["full_build_binding_id"],
            },
        }
    if not math.isfinite(float(args.beta)) or not 0.0 <= float(args.beta) <= 1.0:
        raise ValueError("resolved beta must be finite and in [0,1]")
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(args.threads)
    torch.set_num_threads(args.threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False


def _paths(root: Path, layer: int) -> tuple[Path, Path, Path]:
    search = root / f"layer_{layer:03d}" / "w4a8_native_profile_search"
    return search, search / "preregistration.json", search / "selection.json"


def _load_context(args: argparse.Namespace) -> tuple[Any, Any, list[dict[str, Any]]]:
    from scripts.encode_final_shard import _open_fast_sealed_runtime
    from src.fresh_pipeline_runner import (
        _build_profile_preregistration,
        _load_h13,
        _load_scale_evidence,
    )

    if args.preflight is None:
        raise ValueError("this mode requires --preflight")
    runtime = _open_fast_sealed_runtime(
        args.preflight, layer=args.layer, device=args.device
    )
    legacy_h13, _ = _load_h13(runtime)
    scales = _load_scale_evidence(runtime)
    template = _build_profile_preregistration(runtime, scales, legacy_h13)
    return runtime, scales, list(active_cells(template["cells"]))


def _preregistration(
    runtime: Any,
    scales: Any,
    cells: Sequence[Mapping[str, Any]],
    *,
    beta: float,
    beta_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    from bmmlaw_r7_encoder.search import mass_stratified_experts
    from src.fresh_pipeline_common import canonical_sha256, sha256_file

    full_panel = mass_stratified_experts(
        runtime.capture.role_gate_square_mass_by_expert("fit"), 16
    )
    panel = reduced_mass_panel(full_panel)
    value: dict[str, Any] = {
        "schema": PREREG_SCHEMA,
        "complete": True,
        "layer": runtime.layer,
        "binding": runtime.capture.binding(),
        "cells": [dict(cell) for cell in cells],
        "cell_count": 16,
        "selection_panel": list(panel),
        "full_fit_mass_panel": list(full_panel),
        "selection_panel_construction": (
            "rounded_octile_positions_of_fit_gate_square_mass_stratified_16_v1"
        ),
        "selection_panel_used_only_for_topology_shared_profile": True,
        "rate_objective": {
            "triplets": [list(cell) for cell in balanced_profile_triplets()],
            "weights": [0.5, 0.5],
            "mean_bits_per_projection": 3.5,
            "each_projection_observed_at_k3_and_k4": True,
            "not_final_rate_allocation": True,
        },
        "arithmetic": {
            "h_a8": "mxfp8_e4m3_ue8m0_k32",
            "weight_labels": "native_exact_e4m3",
            "activation": "torch.nn.functional.silu(gate) * up",
            "act_a8": "mxfp8_e4m3_ue8m0_k32",
            "accumulation": "fp32",
            "candidate_specific_down_h_b": True,
            "caller_coordinate_corrected": True,
            "private_down_suh_anchored_by_rate": True,
            "shared_output_svh_anchored": True,
            "beta": beta,
        },
        "beta_mode": beta_lineage["beta_mode"],
        "bootstrap_beta_prior": beta_lineage["bootstrap_beta_prior"],
        "beta_choice": beta_lineage["beta_choice"],
        "h13": {
            "profile_specific": True,
            "fit_only": True,
            "global_alpha": 0.75,
            "expert_local_alpha": LOCAL_H13_ALPHA,
            "gate_square_weighted": True,
        },
        "roles": {
            "fit": "construction_only_document_disjoint_subfolds",
            "selection": "profile_choice_only",
            "holdout": "report_only_after_selection_seal",
        },
        "profile_scale_evidence_id": scales.evidence["evidence_id"],
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "profile_selection_precedes_rate_allocation": True,
        "final_allocator": "v3_exact_mixed_independent_per_tensor_k3_k4",
        "uniform_k3": False,
        "mcg_inputs": 0,
    }
    value["preregistration_id"] = canonical_sha256(value)
    return value


def _write_or_check(path: Path, expected: Mapping[str, Any]) -> None:
    from src.fresh_pipeline_common import atomic_json, load_json_object

    if path.exists():
        if load_json_object(path) != expected:
            raise ValueError(f"existing contract differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, expected)


def _profile(runtime: Any, scales: Any, cell: Mapping[str, Any]) -> tuple[Any, Any]:
    from src.fresh_pipeline_runner import _profiles_for_cell
    from src.glm52_fresh_sqg.codec import realize_shared_residual_profile

    gate, down = _profiles_for_cell(
        runtime,
        scales,
        draw=int(cell["draw"]),
        family=str(cell["family"]),
    )
    if gate.manifest() != cell["gate_up_input_profile"]:
        raise ValueError("gate/up profile reconstruction differs")
    if down.manifest() != cell["down_output_profile"]:
        raise ValueError("down profile reconstruction differs")
    return (
        realize_shared_residual_profile(gate, runtime.device),
        realize_shared_residual_profile(down, runtime.device),
    )


def _profile_global_h13(
    runtime: Any,
    gate_profile: Any,
    hadamard: torch.Tensor,
    *,
    chunk_rows: int,
    cell_id: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    from src.fresh_pipeline_common import canonical_sha256
    from src.glm52_fresh_sqg.reference import tensor_sha256

    rows = runtime.capture.role_rows("fit")
    docs = np.asarray(runtime.capture.doc_epochs[rows], dtype=np.int64)
    calibration = fit_subfold_mask(docs, layer=runtime.layer, subfold="calibration")
    selected_rows = rows[calibration]
    if selected_rows.size == 0:
        raise ValueError("fit/calibration has no layer rows")
    device = hadamard.device
    accumulator = torch.zeros((HIDDEN, HIDDEN), dtype=torch.float32, device=device)
    denominator = torch.zeros((), dtype=torch.float64, device=device)
    suh = gate_profile.expected_stored_fp16().to(device)
    for begin, end, hidden in runtime.capture.iter_hidden_prefetch(
        selected_rows, chunk_rows=chunk_rows, device=device, dtype=torch.float32
    ):
        local = selected_rows[begin:end]
        q_label, observation, _ = prepare_gate_up_operand(
            hidden, suh, hadamard, quantize_a8=True
        )
        if observation is None or bool(observation.preclamp_overflow.any()):
            raise RuntimeError("profile-global h-A8 overflowed")
        q_pre = kquant_prefinalize_operand_from_label_operand(
            q_label, gate_profile.signs, hadamard, validate_values=False
        )
        gates = torch.from_numpy(
            np.array(runtime.capture.topk_weights[local], dtype=np.float32, copy=True)
        ).to(device)
        importance = gates.square().sum(dim=1)
        accumulator.addmm_(q_pre.T, q_pre * importance[:, None])
        denominator.add_(importance.double().sum())
        del hidden, q_label, observation, q_pre, gates, importance
    accumulator.div_(denominator.to(torch.float32))
    matrix = ((accumulator + accumulator.T) * 0.5).contiguous()
    evidence: dict[str, Any] = {
        "schema": GLOBAL_H13_SCHEMA,
        "layer": runtime.layer,
        "profile_cell_id": cell_id,
        "role": "fit",
        "subfold": "calibration",
        "rows": int(selected_rows.size),
        "documents": int(np.unique(docs[calibration]).size),
        "gate_square_sum": float(denominator),
        "matrix_sha256": tensor_sha256(matrix),
        "gate_input_profile": gate_profile.manifest(),
        "selection_used": False,
        "holdout_used": False,
    }
    evidence["evidence_id"] = canonical_sha256(evidence)
    return matrix, evidence


def _fit_balanced_down(
    runtime: Any,
    weights: Any,
    permutation: Any,
    down_profile: Any,
    routed: Any,
    calibration_mask: np.ndarray,
    upstream_by_pair: Mapping[
        tuple[int, int], Mapping[str, NativeProjection]
    ],
    scale_choice_by_pair: Mapping[tuple[int, int], Mapping[str, Any]],
    preliminary_down: Mapping[int, Any],
    source_gpu: Mapping[str, torch.Tensor],
    *,
    beta: float,
    device: torch.device,
    hadamard: torch.Tensor,
    chunk_rows: int,
    kquant_runtime: Any,
    lut_by_bits: Mapping[int, torch.Tensor],
) -> tuple[dict[tuple[int, int, int], NativeProjection], dict[str, Any]]:
    """Fit only the two complementary down candidates used for profile choice."""

    from src.calibration_hessian import apply_frozen_h2_shrinkage
    from src.fresh_pipeline_common import canonical_sha256
    from src.glm52_fresh_sqg import (
        DenseHessian,
        SharedResidualProfile,
        W4A8DerivedTensorBinding,
        encode_uniform_sqg,
    )
    from src.glm52_fresh_sqg.codec import realize_shared_residual_profile
    from src.glm52_fresh_sqg.reference import tensor_sha256

    rows = routed.row_indices[calibration_mask]
    route_gates = routed.applied_gates[torch.from_numpy(calibration_mask)].to(device)
    triplets = balanced_profile_triplets()
    anchors: dict[int, Any] = {}
    anchor_vectors: dict[int, torch.Tensor] = {}
    for bits in (3, 4):
        profile = SharedResidualProfile.from_stored_vector(
            preliminary_down[bits].suh,
            side="input",
            profile_id=(
                f"layer-{runtime.layer:03d}/expert-{int(weights.expert):03d}/"
                f"profile-search-down-k{bits}-anchor-v1"
            ),
            derivation="fit_calibration_official_bf16_preliminary_down_encode",
        )
        anchors[bits] = realize_shared_residual_profile(profile, runtime.device)
        anchor_vectors[bits] = anchors[bits].expected_stored_fp16().to(device)
    raw_h = {
        cell: torch.zeros((INTERMEDIATE, INTERMEDIATE), dtype=torch.float32, device=device)
        for cell in triplets
    }
    raw_encoder_h = {cell: torch.zeros_like(raw_h[cell]) for cell in triplets}
    raw_b = {
        cell: torch.zeros((INTERMEDIATE, HIDDEN), dtype=torch.float32, device=device)
        for cell in triplets
    }
    denominator = torch.zeros((), dtype=torch.float64, device=device)
    importance_chunks: list[torch.Tensor] = []
    for begin, end, hidden in runtime.capture.iter_hidden_prefetch(
        rows, chunk_rows=chunk_rows, device=device, dtype=torch.float32
    ):
        activations = execute_full_w4a8_upstream_pairs(
            hidden,
            upstream_by_pair,
            hadamard,
        )
        teacher = _teacher_output(hidden, source_gpu)
        importance = route_gates[begin:end].square()
        importance_chunks.append(importance)
        for gate_bits, up_bits, down_bits in triplets:
            cell = (gate_bits, up_bits, down_bits)
            q_label, observation, _ = prepare_down_operand(
                activations[(gate_bits, up_bits)],
                anchor_vectors[down_bits],
                hadamard,
                quantize_a8=True,
            )
            if observation is None or bool(observation.preclamp_overflow.any()):
                raise RuntimeError("profile candidate act-A8 overflowed")
            q_eff = effective_canonical_operand_from_quantized_transform(
                q_label, anchor_vectors[down_bits], hadamard,
                validate_values=False,
            )
            q_pre = kquant_prefinalize_operand_from_label_operand(
                q_label, anchors[down_bits].signs, hadamard,
                validate_values=False,
            )
            raw_h[cell].addmm_(q_eff.T, q_eff * importance[:, None])
            raw_b[cell].addmm_(q_eff.T, teacher * importance[:, None])
            raw_encoder_h[cell].addmm_(q_pre.T, q_pre * importance[:, None])
            del q_label, observation, q_eff, q_pre
        denominator.add_(importance.double().sum())
        del hidden, activations, teacher, importance
    official_down_exl = (
        weights.down_hf.index_select(
            1, permutation.new_to_old.to(device=weights.down_hf.device)
        ).T.float().to(device)
    ).contiguous()
    base_config = _config_at_rate(
        runtime,
        weights,
        "down_proj",
        permutation,
        None,
        down_profile,
        bits=3,
        candidate_sweep=True,
    )
    native: dict[tuple[int, int, int], NativeProjection] = {}
    evidence: dict[str, Any] = {}
    importance_vector = torch.cat(importance_chunks)
    for cell in triplets:
        gate_bits, up_bits, down_bits = cell
        canonical_h = raw_h[cell].div_(denominator.to(torch.float32))
        canonical_h = ((canonical_h + canonical_h.T) * 0.5).contiguous()
        cross_term = raw_b[cell].div_(denominator.to(torch.float32))
        encoder_h_raw = raw_encoder_h[cell].div_(denominator.to(torch.float32))
        encoder_h_raw = ((encoder_h_raw + encoder_h_raw.T) * 0.5).contiguous()
        identity_scale = canonical_h.diagonal().double().mean()
        fitted = shrink_cross_term_objective(
            CrossTermStatistics(
                hessian=canonical_h,
                cross_term=cross_term,
                weight_sum=denominator,
                rows=int(rows.size),
                validate_values=False,
            ),
            official_down_exl,
            local_alpha=beta,
            identity_scale=identity_scale,
        )
        target, solver = solve_with_minimal_official_prior(
            fitted,
            official_down_exl,
            identity_scale=identity_scale,
        )
        encoder_h, encoder_shrinkage = apply_frozen_h2_shrinkage(
            encoder_h_raw, importance_vector
        )
        encoder_h = encoder_h.contiguous()
        target = target.contiguous()
        execution = {
            "schema": "glm52-full-w4a8-native-profile-triplet-v1",
            "rates": {
                "gate_proj": gate_bits,
                "up_proj": up_bits,
                "down_proj": down_bits,
            },
            "h_a8": "mxfp8_e4m3_ue8m0_k32",
            "act_a8": "mxfp8_e4m3_ue8m0_k32",
            "activation": "torch.nn.functional.silu(gate) * up",
            "native_e4m3_weights": True,
            "caller_coordinate_corrected": True,
            "coupled_upstream_scale_choice_id": scale_choice_by_pair[
                (gate_bits, up_bits)
            ]["choice_id"],
            "fit_only": True,
        }
        execution_id = canonical_sha256(execution)
        objective: dict[str, Any] = {
            "layer": runtime.layer,
            "expert": int(weights.expert),
            "role": "fit",
            "subfold": "calibration",
            "beta": beta,
            "canonical_h_sha256": None,
            "cross_term_sha256": None,
            "encoder_h_sha256": None,
            "derived_target_sha256": None,
            "candidate_hashes_deferred": True,
            "execution_contract_id": execution_id,
            "coupled_upstream_scale_choice_id": scale_choice_by_pair[
                (gate_bits, up_bits)
            ]["choice_id"],
            "solver": solver,
            "encoder_h_shrinkage": {
                key: float(value) for key, value in encoder_shrinkage.items()
            },
            "selection_used": False,
            "holdout_used": False,
        }
        objective_id = canonical_sha256(objective)
        objective["evidence_id"] = objective_id
        binding = W4A8DerivedTensorBinding(
            official_bf16_parent=base_config.source_binding,
            execution_contract=execution,
            execution_contract_sha256=execution_id,
            fit_hb_evidence_id=objective_id,
            fit_hb_evidence_sha256=objective_id,
            fit_hessian_sha256=None,
            fit_cross_term_sha256=None,
            beta=beta,
            derived_tensor_sha256=None,
            candidate_hashes_deferred=True,
        )
        config = replace(
            base_config,
            bits=down_bits,
            source_binding=binding,
            anchored_input_residual_profile=anchors[down_bits],
        )
        dense = DenseHessian(
            matrix=encoder_h,
            evidence_id=objective_id,
            construction=DERIVED_H2_CONSTRUCTION,
            split_id="fit",
            normalization_count=1,
            routed_sample_count=int(rows.size),
        )
        encoded = encode_uniform_sqg(target, dense, config, runtime=kquant_runtime)
        if not torch.equal(encoded.suh, anchors[down_bits].expected_stored_fp16()):
            raise RuntimeError("profile down candidate changed anchored private suh")
        if not torch.equal(encoded.svh, down_profile.expected_stored_fp16()):
            raise RuntimeError("profile down candidate changed shared output svh")
        native[cell] = _native_projection(
            encoded, bits=down_bits, lut=lut_by_bits[down_bits], device=device
        )
        evidence[str(cell)] = {
            **objective,
            "payload_sha256": _projection_payload_sha256(encoded),
        }
    return native, evidence


def _score_role(
    runtime: Any,
    *,
    expert: int,
    role: str,
    upstream_by_pair: Mapping[
        tuple[int, int], Mapping[str, NativeProjection]
    ],
    native_down: Mapping[tuple[int, int, int], NativeProjection],
    source_gpu: Mapping[str, torch.Tensor],
    hadamard: torch.Tensor,
    chunk_rows: int,
    device: torch.device,
) -> dict[str, Any]:
    routed = runtime.capture.routed_rows(expert, role)
    rows = routed.row_indices
    gates = routed.applied_gates
    validate_route_gates_once(gates, rows=int(rows.size))
    losses = {
        cell: torch.zeros((), dtype=torch.float64, device=device)
        for cell in balanced_profile_triplets()
    }
    teacher_energy = torch.zeros((), dtype=torch.float64, device=device)
    for begin, end, hidden in runtime.capture.iter_hidden_prefetch(
        rows, chunk_rows=chunk_rows, device=device, dtype=torch.float32
    ):
        teacher = _teacher_output(hidden, source_gpu)
        route_gates = gates[begin:end].to(device)
        teacher_energy.add_(
            torch.sum(
                teacher.square().sum(dim=1) * route_gates.square(),
                dtype=torch.float64,
            )
        )
        activations = execute_full_w4a8_upstream_pairs(
            hidden,
            upstream_by_pair,
            hadamard,
        )
        for cell in balanced_profile_triplets():
            gate_bits, up_bits, _ = cell
            candidate = execute_full_w4a8_down(
                activations[(gate_bits, up_bits)], native_down[cell], hadamard
            )
            losses[cell].add_(
                gate_square_weighted_complete_expert_sse(
                    candidate, teacher, route_gates
                )
            )
            del candidate
        del hidden, teacher, route_gates, activations
    ordered_cells = balanced_profile_triplets()
    loss_stack = torch.stack([losses[cell] for cell in ordered_cells])
    mean_sse = loss_stack.mean()
    frozen = torch.cat(
        (
            loss_stack,
            torch.stack((mean_sse, teacher_energy, mean_sse / teacher_energy)),
        )
    ).detach().cpu().tolist()
    validate_frozen_sse_vector(frozen, expected=len(ordered_cells) + 3)
    frozen_losses = dict(zip(ordered_cells, frozen[: len(ordered_cells)], strict=True))
    frozen_mean, frozen_teacher, frozen_relative = frozen[-3:]
    return {
        "role": role,
        "rows": int(rows.size),
        "documents": int(torch.unique(routed.document_epochs).numel()),
        "triplet_sse": {
            str(cell): frozen_losses[cell] for cell in ordered_cells
        },
        "mean_balanced_triplet_sse": frozen_mean,
        "teacher_energy": frozen_teacher,
        "relative_error": frozen_relative,
    }


def _evaluate_expert(
    runtime: Any,
    *,
    expert: int,
    role: str,
    gate_profile: Any,
    down_profile: Any,
    global_h13: torch.Tensor,
    global_evidence: Mapping[str, Any],
    beta: float,
    hadamard: torch.Tensor,
    chunk_rows: int,
    device: torch.device,
    kquant_runtime: Any,
    lut_by_bits: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    from src.fresh_pipeline_common import canonical_sha256
    from src.fresh_pipeline_runner import _load_permutation

    weights = runtime.source.load_expert_bf16(runtime.layer, expert, device=device)
    permutation = _load_permutation(runtime, expert)
    h13, h13_evidence, routed, calibration_mask = _build_expert_h13(
        runtime,
        expert=expert,
        global_h13=global_h13,
        global_evidence=global_evidence,
        gate_profile=gate_profile,
        hadamard=hadamard,
        chunk_rows=chunk_rows,
        device=device,
    )
    coupled_upstream = build_coupled_upstream_candidates(
        runtime,
        weights,
        permutation,
        gate_profile,
        down_profile,
        h13,
        h13_evidence,
        routed,
        calibration_mask,
        hadamard=hadamard,
        device=device,
        chunk_rows=chunk_rows,
        kquant_runtime=kquant_runtime,
        lut_by_bits=lut_by_bits,
    )
    source_gpu = {
        "gate_proj": weights.gate_hf.T.float().to(device),
        "up_proj": weights.up_hf.T.float().to(device),
        "down_proj": weights.down_hf.T.float().to(device),
    }
    preliminary, preliminary_evidence = _preliminary_down_anchors(
        runtime,
        weights,
        permutation,
        gate_profile,
        down_profile,
        routed,
        calibration_mask,
        source_gpu,
        device=device,
        chunk_rows=chunk_rows,
        kquant_runtime=kquant_runtime,
    )
    balanced_scale_choices = {
        pair: coupled_upstream.choice_receipt(pair)
        for pair in ((3, 3), (4, 4))
    }
    native_down, down_evidence = _fit_balanced_down(
        runtime,
        weights,
        permutation,
        down_profile,
        routed,
        calibration_mask,
        coupled_upstream.native_by_pair,
        balanced_scale_choices,
        preliminary,
        source_gpu,
        beta=beta,
        device=device,
        hadamard=hadamard,
        chunk_rows=chunk_rows,
        kquant_runtime=kquant_runtime,
        lut_by_bits=lut_by_bits,
    )
    score = _score_role(
        runtime,
        expert=expert,
        role=role,
        upstream_by_pair=coupled_upstream.native_by_pair,
        native_down=native_down,
        source_gpu=source_gpu,
        hadamard=hadamard,
        chunk_rows=chunk_rows,
        device=device,
    )
    record: dict[str, Any] = {
        "schema": EXPERT_SCORE_SCHEMA,
        "complete": True,
        "layer": runtime.layer,
        "expert": expert,
        "role": role,
        "score": score,
        "h13_evidence": h13_evidence,
        "preliminary_down_anchor_evidence": preliminary_evidence,
        "down_evidence": down_evidence,
        "coupled_scale_selection": coupled_upstream.scale_evidence,
        "coupled_scale_choices": {
            f"k{pair[0]}_k{pair[1]}": choice
            for pair, choice in balanced_scale_choices.items()
        },
        "coupled_raw_scale_centers": coupled_upstream.raw_center_receipts,
        "upstream_payload_sha256": {
            f"k{pair[0]}_k{pair[1]}": {
                projection: _projection_payload_sha256(
                    coupled_upstream.encoded_by_pair[pair][projection]
                )
                for projection in ("gate_proj", "up_proj")
            }
            for pair in ((3, 3), (4, 4))
        },
        "fit_constructed": True,
        "selection_used_for_construction": False,
        "holdout_used_for_construction_or_choice": False,
        "mcg_inputs": 0,
    }
    record["score_id"] = canonical_sha256(record)
    del (
        weights,
        h13,
        coupled_upstream,
        balanced_scale_choices,
        preliminary,
        native_down,
        source_gpu,
    )
    return record


def _run_cell(
    args: argparse.Namespace,
    runtime: Any,
    scales: Any,
    prereg: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    role: str,
    destination: Path,
) -> dict[str, Any]:
    from kquant.sqg_e4m3 import sqg_xor_cheb_t12_bytes
    from src.fresh_pipeline_common import atomic_json, canonical_sha256, load_json_object
    from src.fresh_pipeline_runner import _load_bound_kquant_runtime
    from src.glm52_fresh_sqg.reference import normalized_hadamard
    import src.glm52_fresh_sqg.codec as codec

    if destination.exists():
        existing = load_json_object(destination)
        validate_cell_score(
            existing,
            layer=args.layer,
            role=role,
            cell=cell,
            panel=prereg["selection_panel"],
        )
        return existing
    device = torch.device(args.device)
    hadamard = normalized_hadamard(device=device, dtype=torch.float32, size=HADAMARD_BLOCK)
    gate_profile, down_profile = _profile(runtime, scales, cell)
    global_h13, global_evidence = _profile_global_h13(
        runtime,
        gate_profile,
        hadamard,
        chunk_rows=args.chunk_rows,
        cell_id=str(cell["cell_id"]),
    )
    kquant_runtime = _load_bound_kquant_runtime(runtime)
    lut_by_bits = {
        bits: sqg_xor_cheb_t12_bytes(bits).to(device).contiguous() for bits in (3, 4)
    }
    codec.PRODUCTION_H13_CONSTRUCTION = H13_CONSTRUCTION
    codec.PRODUCTION_H2_CONSTRUCTION = PRELIMINARY_H2_CONSTRUCTION
    panel_scores: list[dict[str, Any]] = []
    for index, expert in enumerate(prereg["selection_panel"]):
        record = _evaluate_expert(
            runtime,
            expert=int(expert),
            role=role,
            gate_profile=gate_profile,
            down_profile=down_profile,
            global_h13=global_h13,
            global_evidence=global_evidence,
            beta=float(args.beta),
            hadamard=hadamard,
            chunk_rows=args.chunk_rows,
            device=device,
            kquant_runtime=kquant_runtime,
            lut_by_bits=lut_by_bits,
        )
        panel_scores.append(record)
        print(
            f"layer {args.layer} cell {cell['cell_id']} {role}: "
            f"expert {index + 1}/{len(prereg['selection_panel'])}",
            flush=True,
        )
    score_device = torch.device(args.device)
    total_sse_tensor = torch.tensor(
        [item["score"]["mean_balanced_triplet_sse"] for item in panel_scores],
        dtype=torch.float64,
        device=score_device,
    ).sum()
    teacher_energy_tensor = torch.tensor(
        [item["score"]["teacher_energy"] for item in panel_scores],
        dtype=torch.float64,
        device=score_device,
    ).sum()
    total_sse = float(total_sse_tensor)
    teacher_energy = float(teacher_energy_tensor)
    result: dict[str, Any] = {
        "schema": CELL_SCORE_SCHEMA,
        "complete": True,
        "layer": args.layer,
        "role": role,
        "cell": dict(cell),
        "selection_panel": list(prereg["selection_panel"]),
        "balanced_triplets": [list(value) for value in balanced_profile_triplets()],
        "aggregate_sse": total_sse,
        "teacher_energy": teacher_energy,
        "aggregate_relative_error": total_sse / teacher_energy,
        "global_h13_evidence": global_evidence,
        "expert_scores": panel_scores,
        "selection_used_for_construction": False,
        "holdout_used_for_choice": False,
        "mcg_inputs": 0,
    }
    result["score_id"] = canonical_sha256(result)
    validate_cell_score(
        result,
        layer=args.layer,
        role=role,
        cell=cell,
        panel=prereg["selection_panel"],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, result)
    return result


def _initialize(args: argparse.Namespace) -> None:
    runtime, scales, cells = _load_context(args)
    search, prereg_path, _ = _paths(args.output_root.resolve(), args.layer)
    prereg = _preregistration(
        runtime,
        scales,
        cells,
        beta=float(args.beta),
        beta_lineage=args.beta_lineage,
    )
    _write_or_check(prereg_path, prereg)
    print(json.dumps({"complete": True, "path": str(prereg_path), "cells": 16}, sort_keys=True))


def _worker(args: argparse.Namespace) -> None:
    from src.fresh_pipeline_common import load_json_object

    runtime, scales, cells = _load_context(args)
    search, prereg_path, _ = _paths(args.output_root.resolve(), args.layer)
    prereg = load_json_object(prereg_path)
    expected = _preregistration(
        runtime,
        scales,
        cells,
        beta=float(args.beta),
        beta_lineage=args.beta_lineage,
    )
    if prereg != expected:
        raise ValueError("W4A8 profile preregistration differs")
    by_id = {str(cell["cell_id"]): cell for cell in cells}
    for cell_id in assigned_cell_ids(cells, int(args.worker_index)):
        _run_cell(
            args,
            runtime,
            scales,
            prereg,
            by_id[cell_id],
            role="selection",
            destination=search / "cells" / cell_id / "score_selection.json",
        )


def _finalize(args: argparse.Namespace) -> None:
    from src.fresh_pipeline_common import atomic_json, canonical_sha256, load_json_object, sha256_file

    search, prereg_path, selection_path = _paths(args.output_root.resolve(), args.layer)
    if selection_path.exists():
        print(selection_path.read_text(encoding="utf-8"))
        return
    prereg = load_json_object(prereg_path)
    scores: dict[str, Any] = {}
    for cell in prereg["cells"]:
        cell_id = str(cell["cell_id"])
        path = search / "cells" / cell_id / "score_selection.json"
        value = load_json_object(path)
        validate_cell_score(
            value,
            layer=args.layer,
            role="selection",
            cell=cell,
            panel=prereg["selection_panel"],
        )
        scores[cell_id] = value
    ordered_ids = sorted(scores)
    objective_tensor = torch.tensor(
        [scores[cell_id]["aggregate_relative_error"] for cell_id in ordered_ids],
        dtype=torch.float64,
        device=torch.device(args.device),
    )
    if not bool(torch.isfinite(objective_tensor).all()) or bool(
        (objective_tensor < 0).any()
    ):
        raise ValueError("profile selection objective is invalid")
    ranked_indices = torch.argsort(objective_tensor, stable=True).cpu().tolist()
    ranked = [ordered_ids[index] for index in ranked_indices]
    selected_id = ranked[0]
    objective_values = objective_tensor.detach().cpu().tolist()
    objectives = dict(zip(ordered_ids, objective_values, strict=True))
    selected = next(cell for cell in prereg["cells"] if cell["cell_id"] == selected_id)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "complete": True,
        "layer": args.layer,
        "preregistration_id": prereg["preregistration_id"],
        "preregistration_sha256": sha256_file(prereg_path),
        "selected_cell_id": selected_id,
        "selected_cell": dict(selected),
        "selected_by": "lowest_selection_role_exact_full_w4a8_balanced_3p5bpw_panel_error",
        "ranked_cell_ids": ranked,
        "cell_metrics": {
            cell_id: {
                "aggregate_relative_error": objectives[cell_id],
                "score_id": scores[cell_id]["score_id"],
                "score_sha256": sha256_file(
                    search / "cells" / cell_id / "score_selection.json"
                ),
            }
            for cell_id in ranked
        },
        "selection_panel": list(prereg["selection_panel"]),
        "rate_objective": prereg["rate_objective"],
        "arithmetic": prereg["arithmetic"],
        "profile_selection_precedes_rate_allocation": True,
        "final_allocation_not_chosen_here": True,
        "selection_used_once_for_choice": True,
        "holdout_used_for_choice": False,
        "holdout_report_is_separate_immutable_artifact": True,
        "mcg_inputs": 0,
    }
    result["selection_id"] = canonical_sha256(result)
    atomic_json(selection_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _finalize_identity_rescue(args: argparse.Namespace) -> None:
    """Select honestly from the four already-scored identity cells only.

    This paid-node recovery never fabricates the twelve excluded factorial
    scores and never rewrites a completed score.  Each adopted score is
    validated against the original 16-cell preregistration before its exact
    file digest is incorporated into the rescue selection receipt.
    """

    from src.fresh_pipeline_common import atomic_json, canonical_sha256, load_json_object, sha256_file

    search, prereg_path, selection_path = _paths(args.output_root.resolve(), args.layer)
    if selection_path.exists():
        print(selection_path.read_text(encoding="utf-8"))
        return
    prereg = load_json_object(prereg_path)
    cells = tuple(prereg["cells"])
    if len(cells) != 16:
        raise ValueError("identity rescue requires the original 16-cell preregistration")
    identities = tuple(
        cell for cell in cells
        if cell.get("family") == "identity" and int(cell.get("draw", -1)) in range(4)
    )
    if len(identities) != 4 or len({cell["cell_id"] for cell in identities}) != 4:
        raise ValueError("identity rescue requires draw-00/01/02/03 identity exactly")
    scores: dict[str, dict[str, Any]] = {}
    score_bindings: dict[str, dict[str, str]] = {}
    for cell in identities:
        cell_id = str(cell["cell_id"])
        path = search / "cells" / cell_id / "score_selection.json"
        value = load_json_object(path)
        validate_cell_score(
            value,
            layer=args.layer,
            role="selection",
            cell=cell,
            panel=prereg["selection_panel"],
        )
        scores[cell_id] = value
        score_bindings[cell_id] = {
            "score_id": str(value["score_id"]),
            "sha256": sha256_file(path),
        }
    ranked = sorted(
        scores,
        key=lambda cell_id: (
            float(scores[cell_id]["aggregate_relative_error"]), cell_id
        ),
    )
    selected_id = ranked[0]
    selected = next(cell for cell in identities if cell["cell_id"] == selected_id)
    excluded = sorted(
        str(cell["cell_id"]) for cell in cells if cell.get("family") != "identity"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "complete": True,
        "layer": args.layer,
        "preregistration_id": prereg["preregistration_id"],
        "preregistration_sha256": sha256_file(prereg_path),
        "selected_cell_id": selected_id,
        "selected_cell": dict(selected),
        "selected_by": "lowest_selection_role_exact_full_w4a8_identity_only_owner_speed_rescue",
        "ranked_cell_ids": ranked,
        "cell_metrics": {
            cell_id: {
                "aggregate_relative_error": float(scores[cell_id]["aggregate_relative_error"]),
                **score_bindings[cell_id],
            }
            for cell_id in ranked
        },
        "selection_panel": list(prereg["selection_panel"]),
        "rate_objective": prereg["rate_objective"],
        "arithmetic": prereg["arithmetic"],
        "rescue_policy": {
            "schema": "glm52-w4a8-identity-only-owner-speed-rescue-v1",
            "original_factorial_cell_count": 16,
            "evaluated_identity_cell_ids": sorted(scores),
            "excluded_cell_ids": excluded,
            "excluded_families": ["aggregate_rms", "inverse_quarter_rms", "quarter_rms"],
            "existing_atomic_scores_adopted_without_rewrite": True,
            "fake_or_imputed_scores": 0,
            "owner_directed_speed_recovery": True,
        },
        "profile_selection_precedes_rate_allocation": True,
        "final_allocation_not_chosen_here": True,
        "selection_used_once_for_choice": True,
        "holdout_used_for_choice": False,
        "mcg_inputs": 0,
    }
    result["selection_id"] = canonical_sha256(result)
    atomic_json(selection_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _holdout(args: argparse.Namespace) -> None:
    from src.fresh_pipeline_common import atomic_json, canonical_sha256, load_json_object, sha256_file

    runtime, scales, cells = _load_context(args)
    search, prereg_path, selection_path = _paths(args.output_root.resolve(), args.layer)
    prereg = load_json_object(prereg_path)
    selection = load_json_object(selection_path)
    destination = search / "holdout_report.json"
    if destination.exists():
        print(destination.read_text(encoding="utf-8"))
        return
    selected = selection["selected_cell"]
    score = _run_cell(
        args,
        runtime,
        scales,
        prereg,
        selected,
        role="holdout",
        destination=search / "holdout_selected_score.json",
    )
    report: dict[str, Any] = {
        "schema": HOLDOUT_SCHEMA,
        "complete": True,
        "layer": args.layer,
        "selection_id": selection["selection_id"],
        "selection_sha256": sha256_file(selection_path),
        "preregistration_sha256": sha256_file(prereg_path),
        "selected_cell_id": selection["selected_cell_id"],
        "aggregate_relative_error": score["aggregate_relative_error"],
        "score_id": score["score_id"],
        "score_sha256": sha256_file(search / "holdout_selected_score.json"),
        "holdout_used_for_choice": False,
        "selection_artifact_was_not_modified": True,
        "mcg_inputs": 0,
    }
    report["report_id"] = canonical_sha256(report)
    atomic_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    args = _parser().parse_args()
    _configure(args)
    if args.initialize:
        _initialize(args)
    elif args.worker_index is not None:
        if args.worker_index not in range(4):
            raise ValueError("worker index must be 0, 1, 2, or 3")
        _worker(args)
    elif args.finalize:
        _finalize(args)
    elif args.finalize_identity_rescue:
        _finalize_identity_rescue(args)
    else:
        _holdout(args)


if __name__ == "__main__":
    started = time.monotonic()
    main()
    print(json.dumps({"elapsed_seconds": time.monotonic() - started}, sort_keys=True))
