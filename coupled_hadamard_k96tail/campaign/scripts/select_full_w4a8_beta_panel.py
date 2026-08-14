#!/usr/bin/env python3
"""Select the full-W4A8 down-objective beta on a fit-only expert panel.

The panel is deliberately independent of selection and holdout.  Sixteen
experts are chosen at deterministic gate-square-mass quantiles of the
``fit/calibration`` routes.  The eight K3/K4 gate/up/down triplets are each
assigned exactly twice, so the panel is a balanced 3.5-bpw microcosm without
depending on an MCG map or on a not-yet-selected SQG allocation.

For each expert, the exact full-W4A8 upstream candidate, canonical ``(H,B)``,
KQuant pre-finalize covariance, down input anchor, and fit/allocation tensors
are built once.  The frozen beta grid then changes only the jointly shrunk
derived target.  All beta targets reuse one finalized dense-H session through
the existing g-scale-aware anchored batch path.  Every candidate closes its
official-BF16 parent binding, ``g_scale == 1``, packed payload, anchored scales,
native finite-E4M3 labels, and exact decoded label operand before it is scored.

Modes are fail closed and resumable:

* ``--prepare-panel`` writes the deterministic fit-only panel manifest;
* worker mode (``--start-index/--end-index``) evaluates panel members;
* ``--finalize`` verifies all records and selects one global beta by summed
  gate-square-weighted complete-expert SSE on ``fit/allocation``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "kquant"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts.encode_full_w4a8_selected_layer import (  # noqa: E402
    _preliminary_selected_down_anchor,
)
from scripts.score_glm52_w4a8_activation_quality import (  # noqa: E402
    NativeProjection,
    apply_gate_up_output_transform_silu,
    prepare_down_operand,
    prepare_gate_up_operand,
    native_label_gemm,
)
from scripts.score_sqg_w4a8_triplet_candidates import (  # noqa: E402
    DERIVED_H2_CONSTRUCTION,
    FULL_W4A8_ENDPOINT,
    HIDDEN,
    H13_CONSTRUCTION,
    INTERMEDIATE,
    NUM_EXPERTS,
    PRELIMINARY_H2_CONSTRUCTION,
    _build_expert_h13,
    _config_at_rate,
    _load_global_h13,
    _native_projection,
    _profile_from_selection,
    _projection_payload_sha256,
    _teacher_output,
    build_coupled_upstream_candidates,
    execute_full_w4a8_down,
    fit_subfold_mask,
    gate_square_weighted_complete_expert_sse,
    subfold_contract,
    validate_coupled_scale_choice,
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
from src.sqg_k34_allocation import (  # noqa: E402
    TRIPLET_PROJECTIONS,
    canonical_json_bytes,
    enumerate_rate_triplets,
    sha256_file,
)
from src.fresh_pipeline_common import (  # noqa: E402
    canonical_sha256 as pipeline_canonical_sha256,
)


BETAS = (0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0)
PANEL_SIZE = 16
PANEL_SCHEMA = "glm52-sqg-full-w4a8-beta-panel-v2"
EXPERT_SCHEMA = "glm52-sqg-full-w4a8-beta-panel-expert-v2"
RESULT_SCHEMA = "glm52-sqg-full-w4a8-beta-selection-v2"
SHARED_STATS_SCHEMA = "glm52-sqg-full-w4a8-shared-hb-qpre-v1"
EXECUTION_SCHEMA = "glm52-full-w4a8-beta-panel-execution-v1"
WORKER_EXECUTION_SCHEMA = "glm52-full-w4a8-beta-panel-worker-execution-v1"
IMMUTABLE_INPUT_SCHEMA = "glm52-full-w4a8-beta-panel-immutable-inputs-v1"
PURPOSE = "fit_only_hyperparameter_selection"
ALLOWED_PANEL_TO_WORKER_CODE_DRIFT = frozenset(
    {"scripts/select_full_w4a8_beta_panel.py"}
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def execution_contract_sha256(value: Mapping[str, Any]) -> str:
    """Hash an execution contract exactly as its manifest binding validates it."""

    return pipeline_canonical_sha256(value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def immutable_runtime_input_binding(runtime: Any) -> dict[str, Any]:
    """Bind model/capture bytes independently of worker GPU placement."""

    preflight = runtime.preflight
    source_preflight = preflight.get("source_seal")
    source_seal = runtime.source_seal
    capture = runtime.capture.binding()
    if not isinstance(source_preflight, Mapping) or not isinstance(
        source_seal, Mapping
    ):
        raise ValueError("worker runtime lacks sealed official-BF16 provenance")
    if int(capture.get("layer", -1)) != int(runtime.layer):
        raise ValueError("worker capture layer differs from runtime layer")
    source_payload_contract = {
        key: source_seal.get(key)
        for key in ("repo", "revision", "index", "config", "shards")
    }
    result: dict[str, Any] = {
        "schema": IMMUTABLE_INPUT_SCHEMA,
        "layer": int(runtime.layer),
        "preflight_id": preflight.get("preflight_id"),
        "source_seal_sha256": source_preflight.get("sha256"),
        "official_bf16_payload_set_sha256": canonical_sha256(
            source_payload_contract
        ),
        "capture": capture,
        "gpu_placement_affects_inputs": False,
    }
    result["binding_id"] = canonical_sha256(result)
    return result


def _validate_immutable_runtime_input_binding(
    value: Mapping[str, Any], *, layer: int
) -> None:
    material = dict(value)
    binding_id = material.pop("binding_id", None)
    capture = value.get("capture")
    if (
        value.get("schema") != IMMUTABLE_INPUT_SCHEMA
        or int(value.get("layer", -1)) != layer
        or value.get("gpu_placement_affects_inputs") is not False
        or not isinstance(capture, Mapping)
        or int(capture.get("layer", -1)) != layer
        or not isinstance(value.get("preflight_id"), str)
        or not isinstance(value.get("source_seal_sha256"), str)
        or len(value["source_seal_sha256"]) != 64
        or not isinstance(value.get("official_bf16_payload_set_sha256"), str)
        or len(value["official_bf16_payload_set_sha256"]) != 64
        or any(
            not isinstance(capture.get(key), str) or len(capture[key]) != 64
            for key in ("capture_manifest_sha256", "layer_manifest_sha256")
        )
        or binding_id != canonical_sha256(material)
    ):
        raise ValueError("immutable layer/capture/source binding differs")


def worker_execution_binding(
    runtime: Any,
    *,
    physical_gpu_override: str,
    panel_code_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record an explicit scheduling override without rebinding any input."""

    from src.fresh_pipeline_common import SELECTED_LAYERS

    physical_gpu_override = str(physical_gpu_override)
    if physical_gpu_override not in {str(index) for index in range(8)}:
        raise ValueError(
            "worker physical GPU override must be one of 0,1,2,3,4,5,6,7"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != physical_gpu_override:
        raise ValueError("worker physical GPU override must match CUDA_VISIBLE_DEVICES")
    if str(torch.device(runtime.device)) != "cuda:0":
        raise ValueError("beta worker must address its one visible GPU as cuda:0")
    immutable_inputs = immutable_runtime_input_binding(runtime)
    code_transition = panel_to_worker_code_transition(
        code_binding() if panel_code_binding is None else panel_code_binding
    )
    result: dict[str, Any] = {
        "schema": WORKER_EXECUTION_SCHEMA,
        "layer": int(runtime.layer),
        "logical_device": "cuda:0",
        "sealed_layer_physical_gpu": str(SELECTED_LAYERS.index(int(runtime.layer))),
        "physical_gpu_override": physical_gpu_override,
        "cuda_visible_devices": physical_gpu_override,
        "override_scope": "visibility_assertion_only",
        "immutable_runtime_inputs": immutable_inputs,
        "immutable_runtime_inputs_id": immutable_inputs["binding_id"],
        "code_transition": code_transition,
    }
    result["execution_id"] = canonical_sha256(result)
    return result


def _validate_worker_execution_binding(
    value: Mapping[str, Any], *, layer: int
) -> None:
    from src.fresh_pipeline_common import SELECTED_LAYERS

    material = dict(value)
    execution_id = material.pop("execution_id", None)
    immutable_inputs = value.get("immutable_runtime_inputs")
    code_transition = value.get("code_transition")
    physical_gpu = value.get("physical_gpu_override")
    if (
        value.get("schema") != WORKER_EXECUTION_SCHEMA
        or int(value.get("layer", -1)) != layer
        or value.get("logical_device") != "cuda:0"
        or value.get("sealed_layer_physical_gpu")
        != str(SELECTED_LAYERS.index(layer))
        or physical_gpu not in {str(index) for index in range(8)}
        or value.get("cuda_visible_devices") != physical_gpu
        or value.get("override_scope") != "visibility_assertion_only"
        or not isinstance(immutable_inputs, Mapping)
        or not isinstance(code_transition, Mapping)
        or value.get("immutable_runtime_inputs_id")
        != immutable_inputs.get("binding_id")
        or execution_id != canonical_sha256(material)
    ):
        raise ValueError("beta worker execution binding differs")
    _validate_immutable_runtime_input_binding(immutable_inputs, layer=layer)
    transition_material = dict(code_transition)
    transition_id = transition_material.pop("transition_id", None)
    if (
        transition_id != canonical_sha256(transition_material)
        or code_transition.get("worker_code_binding") != code_binding()
        or code_transition.get("panel_artifact_rewritten") is not False
        or set(code_transition.get("changed_paths", ()))
        not in (set(), set(ALLOWED_PANEL_TO_WORKER_CODE_DRIFT))
    ):
        raise ValueError("beta worker code-transition binding differs")


def _open_beta_worker_runtime(open_runtime: Any, args: argparse.Namespace) -> Any:
    """Open the unchanged sealed layer while explicitly overriding placement."""

    if args.worker_physical_gpu is None:
        raise ValueError("worker mode requires --worker-physical-gpu")
    return open_runtime(
        args.preflight,
        layer=args.layer,
        device=args.device,
        expected_visible_gpu_override=args.worker_physical_gpu,
    )


def code_binding() -> dict[str, str]:
    """Bind every local arithmetic/provenance source imported by the panel."""

    relatives = (
        "scripts/select_full_w4a8_beta_panel.py",
        "scripts/coupled_gate_up_scale_selector.py",
        "scripts/score_sqg_w4a8_triplet_candidates.py",
        "scripts/encode_full_w4a8_selected_layer.py",
        "scripts/score_glm52_w4a8_activation_quality.py",
        "scripts/w4a8_cross_term.py",
        "scripts/w4a8_stable_solve.py",
        "src/glm52_fresh_sqg/codec.py",
        "src/glm52_fresh_sqg/manifest.py",
        "src/sqg_k34_allocation.py",
    )
    return {relative: sha256_file(PROJECT_ROOT / relative) for relative in relatives}


def panel_to_worker_code_transition(
    panel_code_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Permit only this reviewed worker-initialization repair after panel prep."""

    current = code_binding()
    if set(panel_code_binding) != set(current) or any(
        not isinstance(value, str) or len(value) != 64
        for value in panel_code_binding.values()
    ):
        raise ValueError("beta panel code-binding domain differs")
    changed = sorted(
        relative
        for relative in current
        if panel_code_binding[relative] != current[relative]
    )
    if changed and set(changed) != ALLOWED_PANEL_TO_WORKER_CODE_DRIFT:
        raise ValueError(f"unapproved beta panel-to-worker code drift: {changed}")
    result: dict[str, Any] = {
        "panel_code_binding": dict(panel_code_binding),
        "worker_code_binding": current,
        "changed_paths": changed,
        "allowed_changed_paths": sorted(ALLOWED_PANEL_TO_WORKER_CODE_DRIFT),
        "panel_artifact_rewritten": False,
        "semantic_scope": "initialize_preliminary_down_anchor_h2_contract",
    }
    result["transition_id"] = canonical_sha256(result)
    return result


def configure_worker_codec(codec_module: Any) -> None:
    """Select the exact production contracts used by the beta worker stages."""

    codec_module.PRODUCTION_H13_CONSTRUCTION = H13_CONSTRUCTION
    codec_module.PRODUCTION_H2_CONSTRUCTION = PRELIMINARY_H2_CONSTRUCTION


def _validate_metric(metric: Mapping[str, Any]) -> dict[str, Any]:
    expert = int(metric.get("expert", -1))
    rows = int(metric.get("rows", 0))
    documents = int(metric.get("documents", 0))
    mass = float(metric.get("gate_square_sum", math.nan))
    if not 0 <= expert < NUM_EXPERTS:
        raise ValueError("expert metric ID differs")
    if rows <= 0 or documents <= 0 or not math.isfinite(mass) or mass <= 0:
        raise ValueError(f"expert {expert}: fit metric is empty or invalid")
    return {
        "expert": expert,
        "rows": rows,
        "documents": documents,
        "gate_square_sum": mass,
    }


def build_balanced_fit_panel(
    *,
    layer: int,
    metrics: Sequence[Mapping[str, Any]],
    panel_size: int = PANEL_SIZE,
) -> list[dict[str, Any]]:
    """Return mass-stratified experts with a balanced mixed-rate assignment."""

    if layer < 0:
        raise ValueError("layer must be nonnegative")
    if panel_size <= 0 or panel_size > NUM_EXPERTS or panel_size % 8:
        raise ValueError("panel size must be a positive multiple of eight <= 256")
    normalized = [_validate_metric(metric) for metric in metrics]
    if sorted(metric["expert"] for metric in normalized) != list(range(NUM_EXPERTS)):
        raise ValueError("fit metrics must cover all 256 experts exactly once")
    ranked = sorted(normalized, key=lambda item: (item["gate_square_sum"], item["expert"]))
    positions = [int((index + 0.5) * NUM_EXPERTS / panel_size) for index in range(panel_size)]
    selected = [dict(ranked[position]) for position in positions]

    # Pair rate cells with experts in a hash order rather than mass order.  This
    # preserves exact K3/K4 balance while avoiding a systematic rate/mass trend.
    # Attach an explicit replica ordinal so repeated cells receive independent
    # but deterministic positions in the hash ordering.
    indexed_rates = [
        (replica, rates)
        for replica in range(panel_size // 8)
        for rates in enumerate_rate_triplets()
    ]
    indexed_rates.sort(
        key=lambda item: hashlib.sha256(
            f"glm52-w4a8-beta-rate-v1/{layer}/{item[0]}/{item[1]}".encode()
        ).digest()
    )
    expert_order = sorted(
        range(panel_size),
        key=lambda index: hashlib.sha256(
            f"glm52-w4a8-beta-expert-v1/{layer}/{selected[index]['expert']}".encode()
        ).digest(),
    )
    assignments: list[dict[str, Any]] = []
    for panel_index, expert_index in enumerate(expert_order):
        metric = selected[expert_index]
        rates = indexed_rates[panel_index][1]
        assignments.append(
            {
                **metric,
                "mass_rank": positions[expert_index],
                "stratum": expert_index,
                "rates": dict(zip(TRIPLET_PROJECTIONS, rates, strict=True)),
            }
        )
    assignments.sort(key=lambda item: item["expert"])
    for panel_index, assignment in enumerate(assignments):
        assignment["panel_index"] = panel_index

    values = [
        int(assignment["rates"][projection])
        for assignment in assignments
        for projection in TRIPLET_PROJECTIONS
    ]
    if values.count(3) != values.count(4) or sum(values) / len(values) != 3.5:
        raise AssertionError("balanced beta panel did not preserve 3.5 bpw")
    observed = sorted(
        tuple(int(assignment["rates"][name]) for name in TRIPLET_PROJECTIONS)
        for assignment in assignments
    )
    expected = sorted(list(enumerate_rate_triplets()) * (panel_size // 8))
    if observed != expected:
        raise AssertionError("balanced beta panel lost a mixed-rate triplet")
    return assignments


def choose_beta(
    records: Iterable[Mapping[str, Any]],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Aggregate raw fit/allocation loss and deterministically choose beta."""

    records = list(records)
    if not records:
        raise ValueError("beta selection requires expert records")
    numerical_device = torch.device(device)
    rows_sse: list[list[float]] = []
    rows_energy: list[list[float]] = []
    for record in records:
        candidates = record.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(BETAS):
            raise ValueError("expert beta candidate census differs")
        by_beta = {float(candidate["beta"]): candidate for candidate in candidates}
        if set(by_beta) != set(BETAS):
            raise ValueError("expert beta grid differs")
        row_sse: list[float] = []
        row_energy: list[float] = []
        for beta in BETAS:
            sse = float(by_beta[beta]["fit_allocation_sse"])
            energy = float(by_beta[beta]["fit_allocation_teacher_energy"])
            if not math.isfinite(sse) or sse < 0 or not math.isfinite(energy) or energy <= 0:
                raise ValueError("expert beta score is invalid")
            row_sse.append(sse)
            row_energy.append(energy)
        rows_sse.append(row_sse)
        rows_energy.append(row_energy)
    sse_tensor = torch.tensor(rows_sse, dtype=torch.float64, device=numerical_device)
    energy_tensor = torch.tensor(
        rows_energy, dtype=torch.float64, device=numerical_device
    )
    winner_indices = torch.argmin(sse_tensor, dim=1)
    expert_wins = torch.bincount(winner_indices, minlength=len(BETAS))
    total_sse = sse_tensor.sum(dim=0)
    total_energy = energy_tensor.sum(dim=0)
    winner_index = int(torch.argmin(total_sse))
    rows = []
    for index, beta in enumerate(BETAS):
        sse = float(total_sse[index])
        energy = float(total_energy[index])
        rows.append(
            {
                "beta": beta,
                "fit_allocation_sse": sse,
                "fit_allocation_teacher_energy": energy,
                "fit_allocation_nmse": sse / energy,
                "expert_wins": int(expert_wins[index]),
            }
        )
    winner = rows[winner_index]
    baseline = next(item for item in rows if item["beta"] == 0.0)
    return {
        "winner_beta": winner["beta"],
        "winner_fit_allocation_sse": winner["fit_allocation_sse"],
        "winner_fit_allocation_nmse": winner["fit_allocation_nmse"],
        "winner_delta_sse_vs_beta0": (
            winner["fit_allocation_sse"] - baseline["fit_allocation_sse"]
        ),
        "winner_relative_sse_vs_beta0": (
            winner["fit_allocation_sse"] / baseline["fit_allocation_sse"] - 1.0
            if baseline["fit_allocation_sse"] > 0
            else 0.0
        ),
        "aggregate": rows,
        "tie_break": "minimum_raw_sse_then_smaller_beta",
    }


@dataclass(frozen=True)
class SharedStatistics:
    canonical_h: torch.Tensor
    cross_term: torch.Tensor
    encoder_h: torch.Tensor
    importance: torch.Tensor
    weight_sum: float
    rows: int
    teacher_energy: float
    evidence: Mapping[str, Any]


def _validate_panel_manifest(value: Mapping[str, Any], *, layer: int) -> None:
    if (
        value.get("schema") != PANEL_SCHEMA
        or value.get("complete") is not True
        or int(value.get("layer", -1)) != layer
        or value.get("role") != "fit"
        or value.get("calibration_subfold") != "fit/calibration"
        or value.get("selection_used") is not False
        or value.get("holdout_used") is not False
        or value.get("mcg_inputs") != 0
        or tuple(float(beta) for beta in value.get("betas", ())) != BETAS
        or int(value.get("panel_size", -1)) != PANEL_SIZE
        or value.get("activation_endpoint") != FULL_W4A8_ENDPOINT
    ):
        raise ValueError("fit-only beta panel contract differs")
    assignments = value.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != int(value["panel_size"]):
        raise ValueError("beta panel assignment census differs")
    material = dict(value)
    panel_id = material.pop("panel_id", None)
    if panel_id != canonical_sha256(material):
        raise ValueError("beta panel ID differs")
    panel_to_worker_code_transition(value.get("code_binding", {}))
    immutable_inputs = value.get("immutable_runtime_inputs")
    if not isinstance(immutable_inputs, Mapping):
        raise ValueError("beta panel immutable input binding is absent")
    _validate_immutable_runtime_input_binding(immutable_inputs, layer=layer)


def _prepare_panel(args: argparse.Namespace) -> None:
    from scripts.encode_final_shard import _open_fast_sealed_runtime
    from src.fresh_pipeline_common import atomic_json

    output = args.output_root.resolve() / f"layer_{args.layer:03d}" / "beta_panel.json"
    if output.exists():
        raise ValueError("beta panel manifest already exists")
    if args.panel_size != PANEL_SIZE:
        raise ValueError(f"the preregistered beta panel size is exactly {PANEL_SIZE}")
    runtime = _open_fast_sealed_runtime(args.preflight, layer=args.layer, device=args.device)
    _gate, _down, profile_evidence = _profile_from_selection(
        runtime, args.profile_selection.resolve()
    )
    _global_h13, global_evidence = _load_global_h13(
        args.output_root.resolve(), args.layer, device=torch.device(args.device)
    )
    if global_evidence.get("profile_selection") != profile_evidence:
        raise ValueError("global H13/profile selection differs")
    metrics = []
    for expert in range(NUM_EXPERTS):
        routed = runtime.capture.routed_rows(expert, "fit")
        mask = fit_subfold_mask(
            routed.document_epochs, layer=args.layer, subfold="calibration"
        )
        gates = routed.applied_gates[torch.from_numpy(mask)]
        docs = routed.document_epochs[torch.from_numpy(mask)]
        metrics.append(
            {
                "expert": expert,
                "rows": int(mask.sum()),
                "documents": int(torch.unique(docs).numel()),
                "gate_square_sum": float(gates.double().square().sum()),
            }
        )
    assignments = build_balanced_fit_panel(
        layer=args.layer, metrics=metrics, panel_size=args.panel_size
    )
    result: dict[str, Any] = {
        "schema": PANEL_SCHEMA,
        "complete": True,
        "layer": args.layer,
        "purpose": PURPOSE,
        "activation_endpoint": FULL_W4A8_ENDPOINT,
        "role": "fit",
        "calibration_subfold": "fit/calibration",
        "allocation_subfold": "fit/allocation",
        "subfold_contract": subfold_contract(args.layer),
        "selection_rule": (
            "one center-rank expert per equal-count gate-square-mass stratum; "
            "triplets hash-paired independently of mass"
        ),
        "panel_size": args.panel_size,
        "rate_balance": {
            "independent_per_tensor_k3_k4": True,
            "k3_tensors": args.panel_size * 3 // 2,
            "k4_tensors": args.panel_size * 3 // 2,
            "each_triplet_repetitions": args.panel_size // 8,
            "realized_bpw": 3.5,
            "uniform_k3": False,
        },
        "betas": list(BETAS),
        "beta_choice_objective": (
            "summed gate-square-weighted complete-expert raw SSE on fit/allocation"
        ),
        "assignments": assignments,
        "profile_selection": profile_evidence,
        "global_h13_evidence_id": global_evidence["evidence_id"],
        "capture": runtime.capture.binding(),
        "immutable_runtime_inputs": immutable_runtime_input_binding(runtime),
        "code_binding": code_binding(),
        "selection_used": False,
        "holdout_used": False,
        "mcg_inputs": 0,
    }
    result["panel_id"] = canonical_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)


def _shared_statistics(
    runtime: Any,
    weights: Any,
    permutation: Any,
    routed: Any,
    calibration_mask: np.ndarray,
    gate: NativeProjection,
    up: NativeProjection,
    preliminary_down: Any,
    down_profile: Any,
    coupled_scale_choice: Mapping[str, Any],
    *,
    rates: Mapping[str, int],
    device: torch.device,
    hadamard: torch.Tensor,
    chunk_rows: int,
) -> tuple[SharedStatistics, Any, Mapping[str, torch.Tensor]]:
    """Capture canonical H/B and q-pre covariance exactly once per expert."""

    from src.calibration_hessian import apply_frozen_h2_shrinkage
    from src.glm52_fresh_sqg import SharedResidualProfile
    from src.glm52_fresh_sqg.codec import realize_shared_residual_profile
    from src.glm52_fresh_sqg.reference import tensor_sha256

    rows = routed.row_indices[calibration_mask]
    route_gates = routed.applied_gates[torch.from_numpy(calibration_mask)].to(device)
    source_gpu = {
        "gate_proj": weights.gate_hf.T.float().to(device),
        "up_proj": weights.up_hf.T.float().to(device),
        "down_proj": weights.down_hf.T.float().to(device),
    }
    anchor = SharedResidualProfile.from_stored_vector(
        preliminary_down.suh,
        side="input",
        profile_id=(
            f"layer-{runtime.layer:03d}/expert-{int(weights.expert):03d}/"
            f"down-k{rates['down_proj']}-w4a8-beta-panel-anchor-v1"
        ),
        derivation="fit_calibration_official_bf16_preliminary_down_encode",
    )
    anchor = realize_shared_residual_profile(anchor, runtime.device)
    anchor_vector = anchor.expected_stored_fp16().to(device)
    raw_h = torch.zeros((INTERMEDIATE, INTERMEDIATE), dtype=torch.float32, device=device)
    raw_qpre = torch.zeros_like(raw_h)
    raw_b = torch.zeros((INTERMEDIATE, HIDDEN), dtype=torch.float32, device=device)
    importance_chunks: list[torch.Tensor] = []
    denominator = torch.zeros((), dtype=torch.float64, device=device)
    teacher_energy = torch.zeros((), dtype=torch.float64, device=device)
    for begin, end, hidden in runtime.capture.iter_hidden_prefetch(
        rows, chunk_rows=chunk_rows, device=device, dtype=torch.float32
    ):
        h_a8, h_observation, _ = prepare_gate_up_operand(
            hidden, gate.suh, hadamard, quantize_a8=True
        )
        if h_observation is None or bool(h_observation.preclamp_overflow.any()):
            raise RuntimeError("beta panel h-A8 path overflowed")
        _, _, activation = apply_gate_up_output_transform_silu(
            native_label_gemm(h_a8, gate.weight),
            native_label_gemm(h_a8, up.weight),
            gate.svh,
            up.svh,
            hadamard,
        )
        teacher = _teacher_output(hidden, source_gpu)
        importance = route_gates[begin:end].square()
        q_label, observation, _ = prepare_down_operand(
            activation, anchor_vector, hadamard, quantize_a8=True
        )
        if observation is None or bool(observation.preclamp_overflow.any()):
            raise RuntimeError("beta panel act-A8 path overflowed")
        q_eff = effective_canonical_operand_from_quantized_transform(
            q_label, anchor_vector, hadamard, validate_values=False
        )
        q_pre = kquant_prefinalize_operand_from_label_operand(
            q_label, anchor.signs, hadamard, validate_values=False
        )
        raw_h.addmm_(q_eff.T, q_eff * importance[:, None])
        raw_b.addmm_(q_eff.T, teacher * importance[:, None])
        raw_qpre.addmm_(q_pre.T, q_pre * importance[:, None])
        importance_chunks.append(importance)
        denominator.add_(importance.double().sum())
        teacher_energy.add_(
            torch.sum(teacher.square().sum(dim=1) * importance, dtype=torch.float64)
        )
        del hidden, h_a8, activation, teacher, importance, q_label, q_eff, q_pre
    if not bool(torch.isfinite(denominator)) or not bool(denominator > 0):
        raise ValueError("beta panel calibration has no routed mass")
    raw_h.div_(denominator.to(torch.float32))
    raw_b.div_(denominator.to(torch.float32))
    raw_qpre.div_(denominator.to(torch.float32))
    raw_h = ((raw_h + raw_h.T) * 0.5).contiguous()
    raw_qpre = ((raw_qpre + raw_qpre.T) * 0.5).contiguous()
    importance_vector = torch.cat(importance_chunks)
    encoder_h, shrinkage = apply_frozen_h2_shrinkage(raw_qpre, importance_vector)
    encoder_h = encoder_h.contiguous()
    evidence: dict[str, Any] = {
        "schema": SHARED_STATS_SCHEMA,
        "layer": runtime.layer,
        "expert": int(weights.expert),
        "rates": {name: int(rates[name]) for name in TRIPLET_PROJECTIONS},
        "role": "fit",
        "subfold": "calibration",
        "rows": int(rows.size),
        "gate_square_sum": float(denominator),
        "teacher_energy": float(teacher_energy),
        "canonical_h_sha256": None,
        "cross_term_sha256": None,
        "encoder_h_sha256": None,
        "candidate_hashes_deferred": True,
        "encoder_h_shrinkage": {key: float(value) for key, value in shrinkage.items()},
        "down_input_anchor_sha256": tensor_sha256(anchor.expected_stored_fp16()),
        "down_output_profile": down_profile.manifest(),
        "coupled_upstream_scale_choice_id": coupled_scale_choice["choice_id"],
        "computed_once_for_all_betas": True,
        "selection_used": False,
        "holdout_used": False,
    }
    evidence["evidence_id"] = canonical_sha256(evidence)
    return (
        SharedStatistics(
            canonical_h=raw_h,
            cross_term=raw_b,
            encoder_h=encoder_h,
            importance=importance_vector,
            weight_sum=denominator,
            rows=int(rows.size),
            teacher_energy=teacher_energy,
            evidence=evidence,
        ),
        anchor,
        source_gpu,
    )


def _cache_fit_allocation(
    runtime: Any,
    routed: Any,
    gate: NativeProjection,
    up: NativeProjection,
    source_gpu: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    hadamard: torch.Tensor,
    chunk_rows: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], dict[str, Any]]:
    """Build the beta-invariant allocation tensors once, retaining no text holdout."""

    allocation_mask = fit_subfold_mask(
        routed.document_epochs, layer=runtime.layer, subfold="allocation"
    )
    rows = routed.row_indices[allocation_mask]
    route_gates = routed.applied_gates[torch.from_numpy(allocation_mask)]
    if rows.size == 0:
        raise ValueError("beta panel fit/allocation has no routed rows")
    validate_route_gates_once(route_gates, rows=int(rows.size))
    cached = []
    teacher_energy = torch.zeros((), dtype=torch.float64, device=device)
    for begin, end, hidden in runtime.capture.iter_hidden_prefetch(
        rows, chunk_rows=chunk_rows, device=device, dtype=torch.float32
    ):
        h_a8, observation, _ = prepare_gate_up_operand(
            hidden, gate.suh, hadamard, quantize_a8=True
        )
        if observation is None or bool(observation.preclamp_overflow.any()):
            raise RuntimeError("beta panel allocation h-A8 path overflowed")
        _, _, activation = apply_gate_up_output_transform_silu(
            native_label_gemm(h_a8, gate.weight),
            native_label_gemm(h_a8, up.weight),
            gate.svh,
            up.svh,
            hadamard,
        )
        teacher = _teacher_output(hidden, source_gpu)
        gates = route_gates[begin:end].to(device)
        teacher_energy.add_(
            torch.sum(
                teacher.square().sum(dim=1) * gates.square(), dtype=torch.float64
            )
        )
        cached.append((activation, teacher, gates))
        del hidden, h_a8, observation
    return cached, {
        "rows": int(rows.size),
        "documents": int(torch.unique(routed.document_epochs[torch.from_numpy(allocation_mask)]).numel()),
        "gate_square_sum": float(route_gates.double().square().sum()),
        "teacher_energy": float(teacher_energy),
        "computed_once_for_all_betas": True,
    }


def validate_anchored_beta_candidate(
    encoded: Any,
    native: NativeProjection,
    *,
    beta: float,
    bits: int,
    anchor: Any,
    down_profile: Any,
    official_parent: Any,
) -> None:
    """Close exact lineage, g=1, scales, payload and native-label endpoint."""

    source = encoded.manifest.get("source", {})
    transform = encoded.manifest.get("transform", {})
    closure = encoded.manifest.get("decoded_closure", {})
    if source.get("kind") != "official_bf16_w4a8_derived_fit_target":
        raise RuntimeError("beta candidate lost W4A8-derived source kind")
    if source.get("official_bf16_parent") != official_parent.manifest():
        raise RuntimeError("beta candidate official BF16 parent differs")
    if float(source.get("beta", math.nan)) != beta:
        raise RuntimeError("beta candidate source beta differs")
    if int(encoded.bits) != bits or int(native.bits) != bits:
        raise RuntimeError("beta candidate rate differs")
    if float(transform.get("global_scale", math.nan)) != 1.0:
        raise RuntimeError("beta candidate did not preserve anchored g_scale=1")
    if not torch.equal(encoded.suh, anchor.expected_stored_fp16()):
        raise RuntimeError("beta candidate changed anchored private suh")
    if not torch.equal(encoded.svh, down_profile.expected_stored_fp16()):
        raise RuntimeError("beta candidate changed shared output svh")
    full_packed_closure = closure.get("packed_states_exact") is True
    deferred_candidate_closure = (
        closure.get("mode") == "candidate_sweep_fast"
        and closure.get("full_decode_deferred") is True
        and closure.get("selected_model_eligible") is False
        and encoded.manifest.get("encoder", {}).get(
            "candidate_sweep_fast_requested"
        ) is True
    )
    if closure.get("passed") is not True or not (
        full_packed_closure or deferred_candidate_closure
    ):
        raise RuntimeError("beta candidate packed/decode closure failed")
    if not torch.equal(native.weight, native.weight.to(torch.float8_e4m3fn).float()):
        raise RuntimeError("beta candidate labels are not exact finite E4M3")


def _encode_beta_grid(
    runtime: Any,
    weights: Any,
    permutation: Any,
    down_profile: Any,
    stats: SharedStatistics,
    anchor: Any,
    official_down_exl: torch.Tensor,
    cached_allocation: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    allocation_evidence: Mapping[str, Any],
    *,
    bits: int,
    device: torch.device,
    hadamard: torch.Tensor,
    kquant_runtime: Any,
    lut: torch.Tensor,
) -> list[dict[str, Any]]:
    from src.glm52_fresh_sqg import (
        DenseHessian,
        W4A8DerivedTensorBinding,
        encode_uniform_sqg,
        prepare_dense_h_session,
    )
    from src.glm52_fresh_sqg.reference import tensor_sha256

    base = _config_at_rate(
        runtime,
        weights,
        "down_proj",
        permutation,
        None,
        down_profile,
        bits=bits,
        candidate_sweep=True,
    )
    official_parent = base.source_binding
    dense = DenseHessian(
        matrix=stats.encoder_h,
        evidence_id=str(stats.evidence["evidence_id"]),
        construction=DERIVED_H2_CONSTRUCTION,
        split_id="fit",
        normalization_count=1,
        routed_sample_count=stats.rows,
    )
    session = None
    candidates = []
    identity_scale = stats.canonical_h.diagonal().double().mean()
    statistics = CrossTermStatistics(
        hessian=stats.canonical_h,
        cross_term=stats.cross_term,
        weight_sum=stats.weight_sum,
        rows=stats.rows,
        validate_values=False,
    )
    for beta in BETAS:
        fitted = shrink_cross_term_objective(
            statistics,
            official_down_exl,
            local_alpha=beta,
            identity_scale=identity_scale,
        )
        target, solver = solve_with_minimal_official_prior(
            fitted,
            official_down_exl,
            identity_scale=identity_scale,
        )
        target = target.contiguous()
        contract = {
            "schema": EXECUTION_SCHEMA,
            "activation_endpoint": FULL_W4A8_ENDPOINT,
            "h_a8": "mxfp8_e4m3_ue8m0_k32",
            "weight_endpoint": "native_finite_e4m3_sqg_labels",
            "nonlinearity": "torch_silu_gate_times_up_with_fp16_inter_gemm_boundary",
            "act_a8": "mxfp8_e4m3_ue8m0_k32",
            "down_input_anchor_sha256": stats.evidence["down_input_anchor_sha256"],
            "down_output_profile": down_profile.manifest(),
            "h_b_beta": beta,
            "shared_h_b_qpre_evidence_id": stats.evidence["evidence_id"],
            "selection_rows_used": False,
            "holdout_used": False,
        }
        contract_id = execution_contract_sha256(contract)
        binding = W4A8DerivedTensorBinding(
            official_bf16_parent=official_parent,
            execution_contract=contract,
            execution_contract_sha256=contract_id,
            fit_hb_evidence_id=str(stats.evidence["evidence_id"]),
            fit_hb_evidence_sha256=str(stats.evidence["evidence_id"]),
            fit_hessian_sha256=None,
            fit_cross_term_sha256=None,
            beta=beta,
            derived_tensor_sha256=None,
            candidate_hashes_deferred=True,
        )
        config = replace(
            base,
            source_binding=binding,
            anchored_input_residual_profile=anchor,
        )
        if session is None:
            session = prepare_dense_h_session(dense, config)
        encoded = encode_uniform_sqg(
            target,
            dense,
            config,
            runtime=kquant_runtime,
            dense_h_session=session,
        )
        native = _native_projection(encoded, bits=bits, lut=lut, device=device)
        validate_anchored_beta_candidate(
            encoded,
            native,
            beta=beta,
            bits=bits,
            anchor=anchor,
            down_profile=down_profile,
            official_parent=official_parent,
        )
        loss = torch.zeros((), dtype=torch.float64, device=device)
        for activation, teacher, route_gates in cached_allocation:
            output = execute_full_w4a8_down(activation, native, hadamard)
            loss.add_(
                gate_square_weighted_complete_expert_sse(
                    output, teacher, route_gates
                )
            )
            del output
        candidates.append(
            {
                "beta": beta,
                "fit_allocation_sse": loss,
                "fit_allocation_teacher_energy": float(
                    allocation_evidence["teacher_energy"]
                ),
                "fit_allocation_nmse": None,
                "derived_target_sha256": None,
                "candidate_hashes_deferred": True,
                "payload_sha256": _projection_payload_sha256(encoded),
                "execution_contract_id": contract_id,
                "shared_h_b_qpre_evidence_id": stats.evidence["evidence_id"],
                "solver": solver,
                "dense_h_session_id": session.session_id,
                "dense_h_use_ordinal": int(session.use_count),
                "g_scale": float(encoded.manifest["transform"]["global_scale"]),
                "packed_closure": True,
                "native_e4m3_exact": True,
                "selection_used": False,
                "holdout_used": False,
            }
        )
        del target, encoded, native
    if session is None or session.use_count != len(BETAS):
        raise RuntimeError("beta grid did not reuse one dense-H session exactly")
    frozen_losses = torch.stack(
        [candidate["fit_allocation_sse"] for candidate in candidates]
    ).detach().cpu().tolist()
    validate_frozen_sse_vector(frozen_losses, expected=len(BETAS))
    allocation_teacher = float(allocation_evidence["teacher_energy"])
    for candidate, loss in zip(candidates, frozen_losses, strict=True):
        candidate["fit_allocation_sse"] = loss
        candidate["fit_allocation_nmse"] = loss / allocation_teacher
    if len({candidate["dense_h_session_id"] for candidate in candidates}) != 1:
        raise RuntimeError("beta grid dense-H session identity drifted")
    return candidates


def _expert_path(root: Path, layer: int, expert: int) -> Path:
    return root / f"layer_{layer:03d}" / "beta_experts" / f"expert_{expert:03d}.json"


def _validate_expert_record(
    value: Mapping[str, Any],
    *,
    layer: int,
    assignment: Mapping[str, Any],
    expected_worker_execution: Mapping[str, Any] | None = None,
    expected_immutable_inputs: Mapping[str, Any] | None = None,
) -> None:
    if (
        value.get("schema") != EXPERT_SCHEMA
        or value.get("complete") is not True
        or int(value.get("layer", -1)) != layer
        or int(value.get("expert", -1)) != int(assignment["expert"])
        or value.get("rates") != assignment.get("rates")
        or value.get("role") != "fit"
        or value.get("selection_used") is not False
        or value.get("holdout_used") is not False
        or value.get("mcg_inputs") != 0
    ):
        raise ValueError("beta expert record contract differs")
    candidates = value.get("candidates")
    rates = assignment["rates"]
    scale_selection = value.get("coupled_scale_selection")
    scale_choice = validate_coupled_scale_choice(
        value.get("coupled_scale_choice", {}),
        gate_bits=int(rates["gate_proj"]),
        up_bits=int(rates["up_proj"]),
    )
    if (
        not isinstance(scale_selection, Mapping)
        or scale_selection.get("evidence_id") != scale_choice["scale_evidence_id"]
        or scale_selection.get("selection_used") is not False
        or scale_selection.get("holdout_used") is not False
        or scale_selection.get("fit_allocation_used") is not False
        or value.get("upstream_payload_sha256")
        != {
            "gate_proj": scale_choice["gate"]["payload_sha256"],
            "up_proj": scale_choice["up"]["payload_sha256"],
        }
    ):
        raise ValueError("beta expert coupled scale binding differs")
    if not isinstance(candidates, list) or len(candidates) != len(BETAS):
        raise ValueError("beta expert candidate census differs")
    if tuple(float(candidate.get("beta", math.nan)) for candidate in candidates) != BETAS:
        raise ValueError("beta expert grid/order differs")
    shared_ids = {candidate.get("shared_h_b_qpre_evidence_id") for candidate in candidates}
    session_ids = {candidate.get("dense_h_session_id") for candidate in candidates}
    if shared_ids != {value.get("shared_statistics", {}).get("evidence_id")}:
        raise ValueError("beta candidates did not share H/B/q-pre evidence")
    if len(session_ids) != 1:
        raise ValueError("beta candidates did not share one dense-H session")
    worker_execution = value.get("worker_execution")
    if not isinstance(worker_execution, Mapping):
        raise ValueError("beta expert worker execution binding is absent")
    _validate_worker_execution_binding(worker_execution, layer=layer)
    if (
        expected_worker_execution is not None
        and worker_execution != expected_worker_execution
    ):
        raise ValueError("beta expert worker execution placement differs")
    immutable_inputs = worker_execution["immutable_runtime_inputs"]
    if (
        expected_immutable_inputs is not None
        and immutable_inputs != expected_immutable_inputs
    ):
        raise ValueError("beta expert layer/capture/source bytes differ from panel")
    source_tensors = value.get("official_bf16_tensor_sha256")
    source_shards = value.get("official_bf16_shards")
    if (
        not isinstance(source_tensors, Mapping)
        or set(source_tensors) != set(TRIPLET_PROJECTIONS)
        or any(
            not isinstance(digest, str) or len(digest) != 64
            for digest in source_tensors.values()
        )
        or not isinstance(source_shards, Mapping)
        or set(source_shards) != set(TRIPLET_PROJECTIONS)
        or any(not isinstance(name, str) or not name for name in source_shards.values())
    ):
        raise ValueError("beta expert official-BF16 source binding differs")
    material = dict(value)
    record_id = material.pop("record_id", None)
    if record_id != canonical_sha256(material):
        raise ValueError("beta expert record ID differs")


def _worker(args: argparse.Namespace) -> None:
    from kquant.sqg_e4m3 import sqg_xor_cheb_t12_bytes
    from scripts.encode_final_shard import _open_fast_sealed_runtime
    from src.fresh_pipeline_common import atomic_json
    from src.fresh_pipeline_runner import _load_bound_kquant_runtime, _load_permutation
    from src.glm52_fresh_sqg.reference import normalized_hadamard
    import src.glm52_fresh_sqg.codec as codec

    panel_path = args.output_root.resolve() / f"layer_{args.layer:03d}" / "beta_panel.json"
    panel = _read_json(panel_path)
    _validate_panel_manifest(panel, layer=args.layer)
    assignments = panel["assignments"]
    if args.panel_size != PANEL_SIZE:
        raise ValueError(f"the preregistered beta panel size is exactly {PANEL_SIZE}")
    if (
        args.start_index is None
        or args.end_index is None
        or not 0 <= args.start_index < args.end_index <= len(assignments)
    ):
        raise ValueError("worker requires 0 <= start-index < end-index <= panel size")
    runtime = _open_beta_worker_runtime(_open_fast_sealed_runtime, args)
    worker_execution = worker_execution_binding(
        runtime,
        physical_gpu_override=args.worker_physical_gpu,
        panel_code_binding=panel["code_binding"],
    )
    if worker_execution["immutable_runtime_inputs"] != panel.get(
        "immutable_runtime_inputs"
    ):
        raise ValueError("worker GPU override changed layer/capture/source binding")
    gate_profile, down_profile, profile_evidence = _profile_from_selection(
        runtime, args.profile_selection.resolve()
    )
    if panel.get("profile_selection") != profile_evidence:
        raise ValueError("worker profile selection differs from beta panel")
    device = torch.device(args.device)
    global_h13, global_evidence = _load_global_h13(
        args.output_root.resolve(), args.layer, device=device
    )
    if global_evidence["evidence_id"] != panel["global_h13_evidence_id"]:
        raise ValueError("worker global H13 differs from beta panel")
    hadamard = normalized_hadamard(device=device, dtype=torch.float32, size=128)
    kquant_runtime = _load_bound_kquant_runtime(runtime)
    lut_by_bits = {
        bits: sqg_xor_cheb_t12_bytes(bits).to(device).contiguous() for bits in (3, 4)
    }
    configure_worker_codec(codec)

    for assignment in assignments[args.start_index : args.end_index]:
        expert = int(assignment["expert"])
        output = _expert_path(args.output_root.resolve(), args.layer, expert)
        if output.exists():
            _validate_expert_record(
                _read_json(output),
                layer=args.layer,
                assignment=assignment,
                expected_worker_execution=worker_execution,
                expected_immutable_inputs=panel["immutable_runtime_inputs"],
            )
            continue
        rates = {name: int(assignment["rates"][name]) for name in TRIPLET_PROJECTIONS}
        weights = runtime.source.load_expert_bf16(args.layer, expert, device=device)
        permutation = _load_permutation(runtime, expert)
        h13, h13_evidence, routed, calibration_mask = _build_expert_h13(
            runtime,
            expert=expert,
            global_h13=global_h13,
            global_evidence=global_evidence,
            gate_profile=gate_profile,
            hadamard=hadamard,
            chunk_rows=args.chunk_rows,
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
            chunk_rows=args.chunk_rows,
            kquant_runtime=kquant_runtime,
            lut_by_bits=lut_by_bits,
        )
        pair = (rates["gate_proj"], rates["up_proj"])
        coupled_scale_choice = coupled_upstream.choice_receipt(pair)
        gate_encoded = coupled_upstream.encoded_by_pair[pair]["gate_proj"]
        up_encoded = coupled_upstream.encoded_by_pair[pair]["up_proj"]
        gate = coupled_upstream.native_by_pair[pair]["gate_proj"]
        up = coupled_upstream.native_by_pair[pair]["up_proj"]
        preliminary, preliminary_evidence = _preliminary_selected_down_anchor(
            runtime,
            weights,
            permutation,
            gate_profile,
            down_profile,
            routed,
            calibration_mask,
            bits=rates["down_proj"],
            device=device,
            chunk_rows=args.chunk_rows,
            kquant_runtime=kquant_runtime,
        )
        stats, anchor, source_gpu = _shared_statistics(
            runtime,
            weights,
            permutation,
            routed,
            calibration_mask,
            gate,
            up,
            preliminary,
            down_profile,
            coupled_scale_choice,
            rates=rates,
            device=device,
            hadamard=hadamard,
            chunk_rows=args.chunk_rows,
        )
        cached_allocation, allocation_evidence = _cache_fit_allocation(
            runtime,
            routed,
            gate,
            up,
            source_gpu,
            device=device,
            hadamard=hadamard,
            chunk_rows=args.chunk_rows,
        )
        official_down_exl = (
            weights.down_hf.index_select(
                1, permutation.new_to_old.to(device=weights.down_hf.device)
            ).T.float().to(device)
        ).contiguous()
        candidates = _encode_beta_grid(
            runtime,
            weights,
            permutation,
            down_profile,
            stats,
            anchor,
            official_down_exl,
            cached_allocation,
            allocation_evidence,
            bits=rates["down_proj"],
            device=device,
            hadamard=hadamard,
            kquant_runtime=kquant_runtime,
            lut=lut_by_bits[rates["down_proj"]],
        )
        record: dict[str, Any] = {
            "schema": EXPERT_SCHEMA,
            "complete": True,
            "layer": args.layer,
            "expert": expert,
            "panel_id": panel["panel_id"],
            "panel_assignment": assignment,
            "rates": rates,
            "role": "fit",
            "calibration_subfold": "fit/calibration",
            "allocation_subfold": "fit/allocation",
            "worker_execution": worker_execution,
            "official_bf16_tensor_sha256": dict(weights.tensor_sha256),
            "official_bf16_shards": dict(weights.shard_names),
            "h13_evidence": h13_evidence,
            "coupled_scale_selection": coupled_upstream.scale_evidence,
            "coupled_raw_scale_centers": coupled_upstream.raw_center_receipts,
            "coupled_scale_choice": coupled_scale_choice,
            "preliminary_down_anchor_evidence": preliminary_evidence,
            "upstream_payload_sha256": {
                "gate_proj": _projection_payload_sha256(gate_encoded),
                "up_proj": _projection_payload_sha256(up_encoded),
            },
            "shared_statistics": stats.evidence,
            "fit_allocation": allocation_evidence,
            "candidates": candidates,
            "selection_used": False,
            "holdout_used": False,
            "mcg_inputs": 0,
        }
        record["record_id"] = canonical_sha256(record)
        _validate_expert_record(
            record,
            layer=args.layer,
            assignment=assignment,
            expected_worker_execution=worker_execution,
            expected_immutable_inputs=panel["immutable_runtime_inputs"],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, record)
        del (
            weights,
            h13,
            coupled_upstream,
            gate_encoded,
            up_encoded,
            gate,
            up,
            coupled_scale_choice,
            preliminary,
            stats,
            anchor,
            source_gpu,
            cached_allocation,
            official_down_exl,
            candidates,
        )


def finalize_beta_selection(
    root: Path,
    *,
    layer: int,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    panel_path = root / f"layer_{layer:03d}" / "beta_panel.json"
    panel = _read_json(panel_path)
    _validate_panel_manifest(panel, layer=layer)
    records = []
    receipts = []
    for assignment in panel["assignments"]:
        expert = int(assignment["expert"])
        path = _expert_path(root, layer, expert)
        value = _read_json(path)
        _validate_expert_record(
            value,
            layer=layer,
            assignment=assignment,
            expected_immutable_inputs=panel["immutable_runtime_inputs"],
        )
        if value.get("panel_id") != panel["panel_id"]:
            raise ValueError("beta expert panel binding differs")
        records.append(value)
        receipts.append(
            {
                "expert": expert,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "record_id": value["record_id"],
                "worker_execution_id": value["worker_execution"]["execution_id"],
                "physical_gpu_override": value["worker_execution"][
                    "physical_gpu_override"
                ],
                "immutable_runtime_inputs_id": value["worker_execution"][
                    "immutable_runtime_inputs_id"
                ],
            }
        )
    decision = choose_beta(records, device=device)
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "complete": True,
        "layer": layer,
        "purpose": PURPOSE,
        "activation_endpoint": FULL_W4A8_ENDPOINT,
        "panel_binding": {
            "path": str(panel_path.resolve()),
            "sha256": sha256_file(panel_path),
            "panel_id": panel["panel_id"],
        },
        "panel_size": len(records),
        "betas": list(BETAS),
        **decision,
        "expert_receipts": receipts,
        "shared_work_contract": {
            "canonical_h_b_qpre_once_per_expert": True,
            "fit_allocation_upstream_and_teacher_once_per_expert": True,
            "one_dense_h_session_reused_for_all_betas": True,
            "same_fixed_mixed_rate_triplet_for_all_betas": True,
        },
        "selection_used": False,
        "holdout_used": False,
        "mcg_inputs": 0,
    }
    result["selection_id"] = canonical_sha256(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight")
    parser.add_argument("--profile-selection", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--panel-size", type=int, default=PANEL_SIZE)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument(
        "--worker-physical-gpu",
        choices=tuple(str(index) for index in range(8)),
        help=(
            "required in worker mode; explicitly opts the worker into a "
            "physical GPU different from the layer's historical sealed slot"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-panel", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.threads <= 0 or args.chunk_rows <= 0:
        raise ValueError("threads and chunk rows must be positive")
    if args.finalize:
        if args.worker_physical_gpu is not None:
            raise ValueError("finalize mode does not accept a worker GPU override")
        output = args.output_root.resolve() / f"layer_{args.layer:03d}" / "beta_selection.json"
        if output.exists():
            raise ValueError("beta selection already exists")
        from src.fresh_pipeline_common import atomic_json

        if not args.device.startswith("cuda:") or not torch.cuda.is_available():
            raise ValueError("production beta selection ranking requires CUDA")
        result = finalize_beta_selection(
            args.output_root.resolve(), layer=args.layer, device=args.device
        )
        atomic_json(output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.preflight is None or args.profile_selection is None:
        raise ValueError("prepare/worker mode requires preflight and profile selection")
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise ValueError("exact SQG beta panel requires an explicit CUDA device")
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
    started = time.monotonic()
    if args.prepare_panel:
        if args.worker_physical_gpu is not None:
            raise ValueError(
                "prepare-panel must use the layer's sealed physical GPU"
            )
        _prepare_panel(args)
    else:
        if args.worker_physical_gpu is None:
            raise ValueError("worker mode requires --worker-physical-gpu")
        _worker(args)
    print(json.dumps({"complete": True, "elapsed_seconds": time.monotonic() - started}))


if __name__ == "__main__":
    main()
