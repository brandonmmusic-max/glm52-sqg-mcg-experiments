#!/usr/bin/env python3
"""Seal one layer's no-shortcut coupled profile and beta recipe.

The production B300 artifact used an owner-fixed beta=0.25 identity-only
rescue.  The documented full-build DAG instead requires:

1. a 16-cell W4A8-native bootstrap profile search at beta=0.0625;
2. one layer-local seven-beta decision on fit/allocation;
3. exact bootstrap reuse if beta remains 0.0625, otherwise one final 16-cell
   profile search at the frozen selected beta; and
4. holdout only after each profile selection is sealed.

This implementation performs that DAG from the frozen SQG checkpoint and the
saved capture/Hessian inputs, with updated-QSRT H512/H128/H128 coupled GLM
semantics throughout.  Every expensive cell/expert result is atomic and
resumable.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "kquant"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts.coupled_recipe_core import (  # noqa: E402
    build_profile_global_h13,
    candidate_payloads,
    fit_coupled_down,
    prepare_coupled_expert,
    profile_cells,
    realize_profile_cell,
    score_coupled_candidates,
)
from scripts.profile_search_full_w4a8_native import (  # noqa: E402
    balanced_profile_triplets,
    choose_profile_cell,
    reduced_mass_panel,
)
from scripts.select_full_w4a8_beta_panel import (  # noqa: E402
    BETAS,
    PANEL_SIZE,
    build_balanced_fit_panel,
)


BOOTSTRAP_BETA = 0.0625
PROFILE_PREREG_SCHEMA = "glm52-updated-qsrt-coupled-profile-preregistration-v1"
PROFILE_CELL_SCHEMA = "glm52-updated-qsrt-coupled-profile-cell-score-v1"
PROFILE_SELECTION_SCHEMA = "glm52-updated-qsrt-coupled-profile-selection-v1"
PROFILE_HOLDOUT_SCHEMA = "glm52-updated-qsrt-coupled-profile-holdout-v1"
BETA_PANEL_SCHEMA = "glm52-updated-qsrt-coupled-beta-panel-v1"
BETA_EXPERT_SCHEMA = "glm52-updated-qsrt-coupled-beta-expert-v1"
BETA_CHOICE_SCHEMA = "glm52-updated-qsrt-coupled-beta-choice-v1"
FINAL_BINDING_SCHEMA = "glm52-updated-qsrt-coupled-final-profile-binding-v1"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--preflight", type=Path, required=True)
    value.add_argument("--source-sqg-root", type=Path, required=True)
    value.add_argument("--qsrt-root", type=Path, required=True)
    value.add_argument("--recipe-root", type=Path, required=True)
    value.add_argument("--final-profile-root", type=Path, required=True)
    value.add_argument("--layer", type=int, required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--chunk-rows", type=int, default=256)
    value.add_argument("--threads", type=int, default=6)
    value.add_argument("--smoke-cell")
    value.add_argument("--smoke-expert", type=int)
    return value


def configure(args: argparse.Namespace) -> None:
    if not 3 <= args.layer <= 78:
        raise ValueError("layer must lie in [3,78]")
    if args.chunk_rows <= 0 or args.threads <= 0:
        raise ValueError("chunk rows and threads must be positive")
    if not (args.qsrt_root.resolve() / "qsrt/qsrt_coupled.py").is_file():
        raise FileNotFoundError("updated QSRT coupled implementation is absent")
    if (args.smoke_cell is None) != (args.smoke_expert is None):
        raise ValueError("smoke-cell and smoke-expert must be supplied together")
    if args.smoke_expert is not None and not 0 <= args.smoke_expert < 256:
        raise ValueError("smoke expert must lie in [0,256)")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(args.threads)
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False


def runtime_binding(runtime: Any, args: argparse.Namespace) -> dict[str, Any]:
    from src.fresh_pipeline_runner import _load_scale_evidence

    revision = subprocess.check_output(
        ["git", "-C", str(args.qsrt_root.resolve()), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    diff = subprocess.check_output(
        ["git", "-C", str(args.qsrt_root.resolve()), "diff", "--binary", "HEAD"]
    )
    scales = _load_scale_evidence(runtime)
    result: dict[str, Any] = {
        "schema": "glm52-updated-qsrt-coupled-recipe-runtime-binding-v1",
        "layer": args.layer,
        "capture": runtime.capture.binding(),
        "preflight_id": runtime.preflight.get("preflight_id"),
        "source": runtime.source.validation.manifest(),
        "profile_scale_evidence_id": scales.evidence["evidence_id"],
        "canonical_hessian_dataset": {
            "repo": "brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians",
            "revision": "a05b3b92d749f6a641af5cfd52de2b4720380dfd",
        },
        "qsrt": {
            "root": str(args.qsrt_root.resolve()),
            "revision": revision,
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
        },
        "source_is_frozen_sqg_checkpoint": True,
        "official_bf16_weight_shards_read": False,
        "coupled_transform": {
            "residual_hadamard": 512,
            "preactivation_hadamard": 128,
            "postactivation_hadamard": 128,
            "residual_draw": 0,
            "intermediate_draw": 0,
            "activation": "silu",
        },
        "h13": {
            "global_alpha": 0.75,
            "local_alpha": 0.25,
            "profile_specific_h_a8": True,
            "fit_calibration_only": True,
        },
        "down_objective": {
            "candidate_conditioned_h_b": True,
            "fit_calibration_only": True,
        },
    }
    result["binding_id"] = canonical_sha256(result)
    return result


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    from src.fresh_pipeline_common import atomic_json as write

    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, dict(value))


def profile_paths(
    layer_root: Path, stage: str
) -> tuple[Path, Path, Path, Path]:
    search = layer_root / stage / "w4a8_native_profile_search"
    return (
        search,
        search / "preregistration.json",
        search / "selection.json",
        search / "holdout_selected_score.json",
    )


def profile_preregistration(
    runtime: Any,
    binding: Mapping[str, Any],
    *,
    beta: float,
    stage: str,
) -> dict[str, Any]:
    from bmmlaw_r7_encoder.search import mass_stratified_experts

    cells = profile_cells(runtime)
    full_panel = mass_stratified_experts(
        runtime.capture.role_gate_square_mass_by_expert("fit"), 16
    )
    panel = reduced_mass_panel(full_panel)
    result: dict[str, Any] = {
        "schema": PROFILE_PREREG_SCHEMA,
        "complete": True,
        "layer": runtime.layer,
        "stage": stage,
        "beta": float(beta),
        "cells": list(cells),
        "cell_count": 16,
        "selection_panel": list(panel),
        "full_fit_mass_panel": list(full_panel),
        "selection_panel_construction": (
            "rounded_octile_positions_of_fit_gate_square_mass_stratified_16_v1"
        ),
        "rate_objective": {
            "triplets": [list(item) for item in balanced_profile_triplets()],
            "weights": [0.5, 0.5],
            "mean_bits_per_projection": 3.5,
            "not_final_rate_allocation": True,
        },
        "roles": {
            "fit/calibration": "construction_only",
            "selection": "profile_choice_only",
            "holdout": "report_only_after_selection_seal",
        },
        "runtime_binding": dict(binding),
        "no_b300_identity_only_rescue": True,
        "profile_selection_precedes_rate_allocation": True,
        "holdout_used_for_choice": False,
        "mcg_inputs": 0,
    }
    result["preregistration_id"] = canonical_sha256(result)
    return result


def validate_profile_preregistration(
    value: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if value != expected:
        raise ValueError("coupled profile preregistration differs")


def validate_profile_cell(
    value: Mapping[str, Any],
    *,
    prereg: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> None:
    material = dict(value)
    score_id = material.pop("score_id", None)
    if (
        value.get("schema") != PROFILE_CELL_SCHEMA
        or value.get("complete") is not True
        or int(value.get("layer", -1)) != int(prereg["layer"])
        or float(value.get("beta", -1.0)) != float(prereg["beta"])
        or value.get("cell") != cell
        or value.get("selection_panel") != prereg["selection_panel"]
        or value.get("preregistration_id") != prereg["preregistration_id"]
        or value.get("selection_used_for_construction") is not False
        or value.get("holdout_used_for_choice") is not False
        or not math.isfinite(float(value.get("aggregate_sse", math.nan)))
        or not float(value["aggregate_sse"]) >= 0
        or not math.isfinite(float(value.get("teacher_energy", math.nan)))
        or not float(value["teacher_energy"]) > 0
        or score_id != canonical_sha256(material)
    ):
        raise ValueError("coupled profile cell score differs")


def run_profile_cell(
    args: argparse.Namespace,
    runtime: Any,
    prereg: Mapping[str, Any],
    cell: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    if output.is_file():
        value = load_json(output)
        validate_profile_cell(value, prereg=prereg, cell=cell)
        return value
    if output.exists():
        raise ValueError(f"profile cell output is unsafe: {output}")
    started = time.monotonic()
    gate_profile, down_profile = realize_profile_cell(runtime, cell)
    global_h13, global_evidence = build_profile_global_h13(
        runtime,
        gate_profile,
        cell_id=str(cell["cell_id"]),
        chunk_rows=args.chunk_rows,
    )
    expert_scores = []
    for index, expert in enumerate(prereg["selection_panel"]):
        prepared = prepare_coupled_expert(
            runtime,
            expert=int(expert),
            gate_profile=gate_profile,
            down_profile=down_profile,
            global_h13=global_h13,
            global_evidence=global_evidence,
            triplets=balanced_profile_triplets(),
            qsrt_root=args.qsrt_root,
            chunk_rows=args.chunk_rows,
        )
        encoded_down, native_down, down_evidence = fit_coupled_down(
            prepared, beta=float(prereg["beta"])
        )
        score = score_coupled_candidates(
            prepared, native_down, score_role="selection"
        )
        expert_scores.append(
            {
                "expert": int(expert),
                "score": score,
                "payload_sha256": candidate_payloads(prepared, encoded_down),
                "coupled_h13_evidence": prepared.coupled_h13_evidence,
                "preliminary_h2_evidence": prepared.preliminary_h2_evidence,
                "down_objective_evidence_id": {
                    str((*pair, bits)): evidence["evidence_id"]
                    for pair, by_bits in down_evidence.items()
                    for bits, evidence in by_bits.items()
                    if (*pair, bits) in balanced_profile_triplets()
                },
            }
        )
        del encoded_down, native_down, down_evidence
        prepared.release()
        print(
            f"layer {runtime.layer} {cell['cell_id']}: "
            f"selection expert {index + 1}/{len(prereg['selection_panel'])}",
            flush=True,
        )
    total_sse = sum(
        float(item["score"]["mean_balanced_triplet_sse"])
        for item in expert_scores
    )
    teacher_energy = sum(
        float(item["score"]["teacher_energy"]) for item in expert_scores
    )
    result: dict[str, Any] = {
        "schema": PROFILE_CELL_SCHEMA,
        "complete": True,
        "layer": runtime.layer,
        "beta": float(prereg["beta"]),
        "cell": dict(cell),
        "selection_panel": list(prereg["selection_panel"]),
        "preregistration_id": prereg["preregistration_id"],
        "balanced_triplets": [list(item) for item in balanced_profile_triplets()],
        "aggregate_sse": total_sse,
        "teacher_energy": teacher_energy,
        "aggregate_relative_error": total_sse / teacher_energy,
        "global_h13_evidence": global_evidence,
        "expert_scores": expert_scores,
        "elapsed_seconds": time.monotonic() - started,
        "selection_used_for_construction": False,
        "holdout_used_for_choice": False,
        "mcg_inputs": 0,
    }
    result["score_id"] = canonical_sha256(result)
    validate_profile_cell(result, prereg=prereg, cell=cell)
    atomic_json(output, result)
    return result


def profile_selection_arithmetic(beta: float) -> dict[str, Any]:
    return {
        "h_a8": "mxfp8_e4m3_ue8m0_k32",
        "weight_labels": "native_exact_e4m3",
        "activation": "torch.nn.functional.silu(gate) * up",
        "act_a8": "mxfp8_e4m3_ue8m0_k32",
        "accumulation": "fp32",
        "candidate_specific_down_h_b": True,
        "caller_coordinate_corrected": True,
        "private_down_suh_anchored_by_rate": True,
        "shared_output_svh_anchored": True,
        "beta": float(beta),
        "updated_qsrt_coupled_hadamard": True,
    }


def run_profile_holdout(
    args: argparse.Namespace,
    runtime: Any,
    prereg: Mapping[str, Any],
    selection: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    if output.is_file():
        value = load_json(output)
        if (
            value.get("schema") != PROFILE_HOLDOUT_SCHEMA
            or value.get("complete") is not True
            or value.get("selection_id") != selection["selection_id"]
            or value.get("holdout_used_for_choice") is not False
        ):
            raise ValueError("coupled profile holdout differs")
        return value
    cell = selection["selected_cell"]
    gate_profile, down_profile = realize_profile_cell(runtime, cell)
    global_h13, global_evidence = build_profile_global_h13(
        runtime,
        gate_profile,
        cell_id=str(cell["cell_id"]),
        chunk_rows=args.chunk_rows,
    )
    records = []
    for index, expert in enumerate(prereg["selection_panel"]):
        prepared = prepare_coupled_expert(
            runtime,
            expert=int(expert),
            gate_profile=gate_profile,
            down_profile=down_profile,
            global_h13=global_h13,
            global_evidence=global_evidence,
            triplets=balanced_profile_triplets(),
            qsrt_root=args.qsrt_root,
            chunk_rows=args.chunk_rows,
        )
        encoded_down, native_down, down_evidence = fit_coupled_down(
            prepared, beta=float(prereg["beta"])
        )
        records.append(
            {
                "expert": int(expert),
                "score": score_coupled_candidates(
                    prepared, native_down, score_role="holdout"
                ),
            }
        )
        del encoded_down, native_down, down_evidence
        prepared.release()
        print(
            f"layer {runtime.layer} {prereg['stage']}: "
            f"holdout expert {index + 1}/{len(prereg['selection_panel'])}",
            flush=True,
        )
    total_sse = sum(
        float(item["score"]["mean_balanced_triplet_sse"]) for item in records
    )
    energy = sum(float(item["score"]["teacher_energy"]) for item in records)
    result = {
        "schema": PROFILE_HOLDOUT_SCHEMA,
        "complete": True,
        "layer": runtime.layer,
        "beta": float(prereg["beta"]),
        "selection_id": selection["selection_id"],
        "selected_cell_id": selection["selected_cell_id"],
        "records": records,
        "aggregate_sse": total_sse,
        "teacher_energy": energy,
        "aggregate_relative_error": total_sse / energy,
        "selection_sealed_before_holdout": True,
        "holdout_used_for_choice": False,
    }
    result["holdout_id"] = canonical_sha256(result)
    atomic_json(output, result)
    return result


def run_profile_search(
    args: argparse.Namespace,
    runtime: Any,
    binding: Mapping[str, Any],
    *,
    layer_root: Path,
    stage: str,
    beta: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    search, prereg_path, selection_path, holdout_path = profile_paths(
        layer_root, stage
    )
    expected = profile_preregistration(
        runtime, binding, beta=float(beta), stage=stage
    )
    if prereg_path.is_file():
        prereg = load_json(prereg_path)
        validate_profile_preregistration(prereg, expected)
    elif prereg_path.exists():
        raise ValueError(f"unsafe preregistration path: {prereg_path}")
    else:
        atomic_json(prereg_path, expected)
        prereg = expected
    scores: dict[str, dict[str, Any]] = {}
    for cell in prereg["cells"]:
        cell_id = str(cell["cell_id"])
        scores[cell_id] = run_profile_cell(
            args,
            runtime,
            prereg,
            cell,
            search / "cells" / cell_id / "score_selection.json",
        )
    objectives = {
        cell_id: float(value["aggregate_sse"])
        for cell_id, value in scores.items()
    }
    selected_id = choose_profile_cell(objectives)
    selected_cell = next(
        dict(cell) for cell in prereg["cells"] if cell["cell_id"] == selected_id
    )
    expected_selection: dict[str, Any] = {
        "schema": PROFILE_SELECTION_SCHEMA,
        "complete": True,
        "layer": runtime.layer,
        "stage": stage,
        "arithmetic": profile_selection_arithmetic(beta),
        "selected_cell": selected_cell,
        "selected_cell_id": selected_id,
        "selected_by": "lowest_selection_role_updated_qsrt_coupled_full_w4a8_sse",
        "ranked_cell_ids": sorted(objectives, key=lambda key: (objectives[key], key)),
        "cell_metrics": {
            key: {
                "aggregate_sse": float(scores[key]["aggregate_sse"]),
                "aggregate_relative_error": float(
                    scores[key]["aggregate_relative_error"]
                ),
                "score_id": scores[key]["score_id"],
            }
            for key in scores
        },
        "selection_panel": list(prereg["selection_panel"]),
        "preregistration_id": prereg["preregistration_id"],
        "preregistration_sha256": sha256_file(prereg_path),
        "profile_selection_precedes_rate_allocation": True,
        "selection_used_once_for_choice": True,
        "holdout_used_for_choice": False,
        "final_allocation_not_chosen_here": True,
        "no_b300_identity_only_rescue": True,
        "mcg_inputs": 0,
    }
    expected_selection["selection_id"] = canonical_sha256(expected_selection)
    if selection_path.is_file():
        selection = load_json(selection_path)
        if selection != expected_selection:
            raise ValueError("coupled profile selection differs")
    elif selection_path.exists():
        raise ValueError(f"unsafe selection path: {selection_path}")
    else:
        atomic_json(selection_path, expected_selection)
        selection = expected_selection
    holdout = run_profile_holdout(
        args, runtime, prereg, selection, holdout_path
    )
    return selection, holdout


def beta_paths(layer_root: Path) -> tuple[Path, Path]:
    root = layer_root / "beta"
    return root / "beta_panel.json", root / "beta_choice.json"


def beta_panel_manifest(
    runtime: Any,
    binding: Mapping[str, Any],
    bootstrap_selection: Mapping[str, Any],
) -> dict[str, Any]:
    import torch
    from scripts.score_sqg_w4a8_triplet_candidates import fit_subfold_mask

    metrics = []
    for expert in range(256):
        routed = runtime.capture.routed_rows(expert, "fit")
        mask = fit_subfold_mask(
            routed.document_epochs, layer=runtime.layer, subfold="calibration"
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
        layer=runtime.layer, metrics=metrics, panel_size=PANEL_SIZE
    )
    result: dict[str, Any] = {
        "schema": BETA_PANEL_SCHEMA,
        "complete": True,
        "layer": runtime.layer,
        "purpose": "fit_only_layer_local_beta_selection",
        "calibration_subfold": "fit/calibration",
        "allocation_subfold": "fit/allocation",
        "betas": list(BETAS),
        "panel_size": PANEL_SIZE,
        "assignments": assignments,
        "rate_balance": {
            "each_triplet_repetitions": 2,
            "k3_tensors": 24,
            "k4_tensors": 24,
            "realized_bpw": 3.5,
            "not_final_rate_allocation": True,
        },
        "bootstrap_profile": {
            "selection_id": bootstrap_selection["selection_id"],
            "selected_cell_id": bootstrap_selection["selected_cell_id"],
            "beta": BOOTSTRAP_BETA,
        },
        "runtime_binding": dict(binding),
        "selection_used": False,
        "holdout_used": False,
        "no_owner_fixed_beta": True,
        "mcg_inputs": 0,
    }
    result["panel_id"] = canonical_sha256(result)
    return result


def validate_beta_expert(
    value: Mapping[str, Any],
    *,
    panel: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> None:
    material = dict(value)
    record_id = material.pop("record_id", None)
    if (
        value.get("schema") != BETA_EXPERT_SCHEMA
        or value.get("complete") is not True
        or int(value.get("layer", -1)) != int(panel["layer"])
        or value.get("assignment") != assignment
        or value.get("panel_id") != panel["panel_id"]
        or tuple(float(item["beta"]) for item in value.get("candidates", []))
        != tuple(BETAS)
        or any(
            not math.isfinite(float(item.get("fit_allocation_sse", math.nan)))
            or float(item["fit_allocation_sse"]) < 0
            for item in value.get("candidates", [])
        )
        or record_id != canonical_sha256(material)
    ):
        raise ValueError("coupled beta expert record differs")


def select_layer_beta(
    records: list[Mapping[str, Any]],
) -> tuple[dict[float, float], float]:
    """Select the layer beta from fit/allocation SSE only.

    The ordering is explicit so a numeric tie always chooses the smaller beta,
    matching the preregistered panel rule without consulting holdout scores.
    """

    totals = {
        float(beta): sum(
            float(
                next(
                    item
                    for item in record["candidates"]
                    if float(item["beta"]) == float(beta)
                )["fit_allocation_sse"]
            )
            for record in records
        )
        for beta in BETAS
    }
    selected = min(totals, key=lambda beta: (totals[beta], beta))
    return totals, float(selected)


def run_beta_panel(
    args: argparse.Namespace,
    runtime: Any,
    binding: Mapping[str, Any],
    *,
    layer_root: Path,
    bootstrap_selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    panel_path, choice_path = beta_paths(layer_root)
    expected_panel = beta_panel_manifest(runtime, binding, bootstrap_selection)
    if panel_path.is_file():
        panel = load_json(panel_path)
        if panel != expected_panel:
            raise ValueError("coupled beta panel differs")
    elif panel_path.exists():
        raise ValueError(f"unsafe beta panel path: {panel_path}")
    else:
        atomic_json(panel_path, expected_panel)
        panel = expected_panel
    cell = bootstrap_selection["selected_cell"]
    gate_profile, down_profile = realize_profile_cell(runtime, cell)
    global_h13, global_evidence = build_profile_global_h13(
        runtime,
        gate_profile,
        cell_id=str(cell["cell_id"]),
        chunk_rows=args.chunk_rows,
    )
    records = []
    expert_root = panel_path.parent / "experts"
    for index, assignment in enumerate(panel["assignments"]):
        expert = int(assignment["expert"])
        output = expert_root / f"expert_{expert:03d}.json"
        if output.is_file():
            record = load_json(output)
            validate_beta_expert(record, panel=panel, assignment=assignment)
            records.append(record)
            continue
        rates = tuple(
            int(assignment["rates"][name])
            for name in ("gate_proj", "up_proj", "down_proj")
        )
        prepared = prepare_coupled_expert(
            runtime,
            expert=expert,
            gate_profile=gate_profile,
            down_profile=down_profile,
            global_h13=global_h13,
            global_evidence=global_evidence,
            triplets=(rates,),
            qsrt_root=args.qsrt_root,
            chunk_rows=args.chunk_rows,
        )
        candidates = []
        for beta in BETAS:
            encoded_down, native_down, evidence = fit_coupled_down(
                prepared, beta=float(beta)
            )
            score = score_coupled_candidates(
                prepared, native_down, score_role="fit/allocation"
            )
            pair = rates[:2]
            candidates.append(
                {
                    "beta": float(beta),
                    "fit_allocation_sse": float(
                        score["mean_balanced_triplet_sse"]
                    ),
                    "fit_allocation_teacher_energy": float(
                        score["teacher_energy"]
                    ),
                    "fit_allocation_relative_error": float(
                        score["relative_error"]
                    ),
                    "payload_sha256": candidate_payloads(
                        prepared, encoded_down
                    )[str(rates)],
                    "down_objective_evidence_id": evidence[pair][rates[2]][
                        "evidence_id"
                    ],
                    "selection_used": False,
                    "holdout_used": False,
                }
            )
            del encoded_down, native_down, evidence
            gc.collect()
        record: dict[str, Any] = {
            "schema": BETA_EXPERT_SCHEMA,
            "complete": True,
            "layer": runtime.layer,
            "expert": expert,
            "assignment": dict(assignment),
            "panel_id": panel["panel_id"],
            "candidates": candidates,
            "coupled_h13_evidence": prepared.coupled_h13_evidence,
            "preliminary_h2_evidence": prepared.preliminary_h2_evidence,
            "selection_used": False,
            "holdout_used": False,
        }
        record["record_id"] = canonical_sha256(record)
        validate_beta_expert(record, panel=panel, assignment=assignment)
        atomic_json(output, record)
        records.append(record)
        prepared.release()
        print(
            f"layer {runtime.layer} beta panel: "
            f"expert {index + 1}/{len(panel['assignments'])}",
            flush=True,
        )
    totals, selected_beta = select_layer_beta(records)
    expected_choice: dict[str, Any] = {
        "schema": BETA_CHOICE_SCHEMA,
        "complete": True,
        "layer": runtime.layer,
        "selected_beta": float(selected_beta),
        "selection_rule": "minimum_summed_fit_allocation_raw_sse_smaller_beta_tie_break",
        "summed_fit_allocation_sse": {
            str(beta): totals[float(beta)] for beta in BETAS
        },
        "panel_id": panel["panel_id"],
        "panel_sha256": sha256_file(panel_path),
        "expert_record_ids": [record["record_id"] for record in records],
        "selection_used": False,
        "holdout_used": False,
        "owner_fixed_beta_bypassed": True,
        "seven_beta_panel_bypassed": False,
        "fleet_wide_inference_from_layer77": False,
        "mcg_inputs": 0,
    }
    expected_choice["choice_id"] = canonical_sha256(expected_choice)
    if choice_path.is_file():
        choice = load_json(choice_path)
        if choice != expected_choice:
            raise ValueError("coupled beta choice differs")
    elif choice_path.exists():
        raise ValueError(f"unsafe beta choice path: {choice_path}")
    else:
        atomic_json(choice_path, expected_choice)
        choice = expected_choice
    return panel, choice


def publish_final_profile(
    args: argparse.Namespace,
    *,
    layer_root: Path,
    selection: Mapping[str, Any],
    beta_choice: Mapping[str, Any],
) -> dict[str, Any]:
    source = (
        layer_root
        / ("bootstrap_profile" if float(beta_choice["selected_beta"]) == BOOTSTRAP_BETA else "final_profile")
        / "w4a8_native_profile_search"
        / "selection.json"
    )
    destination = (
        args.final_profile_root.resolve()
        / f"layer_{args.layer:03d}"
        / "w4a8_native_profile_search"
        / "selection.json"
    )
    if sha256_file(source) != hashlib.sha256(canonical_bytes(selection)).hexdigest():
        # atomic_json writes pretty JSON; compare the loaded object as authority.
        if load_json(source) != selection:
            raise ValueError("selected final profile source differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if load_json(destination) != selection:
            raise ValueError("published final profile differs")
    elif destination.exists():
        raise ValueError(f"unsafe published profile path: {destination}")
    else:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
    binding_path = destination.parent / "final_profile_binding.json"
    expected: dict[str, Any] = {
        "schema": FINAL_BINDING_SCHEMA,
        "complete": True,
        "layer": args.layer,
        "selected_beta": float(beta_choice["selected_beta"]),
        "beta_choice_id": beta_choice["choice_id"],
        "beta_choice_sha256": sha256_file(layer_root / "beta/beta_choice.json"),
        "profile_selection_id": selection["selection_id"],
        "profile_selection_sha256": sha256_file(destination),
        "bootstrap_reused_byte_for_byte": (
            float(beta_choice["selected_beta"]) == BOOTSTRAP_BETA
        ),
        "profile_reselected_exactly_once": (
            float(beta_choice["selected_beta"]) != BOOTSTRAP_BETA
        ),
        "no_b300_owner_speed_rescue": True,
    }
    expected["binding_id"] = canonical_sha256(expected)
    if binding_path.is_file():
        if load_json(binding_path) != expected:
            raise ValueError("final profile binding differs")
    else:
        atomic_json(binding_path, expected)
    return expected


def main() -> None:
    args = parser().parse_args()
    configure(args)
    if str(args.qsrt_root.resolve()) not in sys.path:
        sys.path.insert(0, str(args.qsrt_root.resolve()))
    from scripts.score_coupled_tail_triplet_candidates import _all_k3_map
    from src.sqg_checkpoint_source import open_sqg_transcode_runtime

    layer_root = args.recipe_root.resolve() / f"layer_{args.layer:03d}"
    layer_root.mkdir(parents=True, exist_ok=True)
    # Recipe helpers may run ahead of the main rolling campaign.  Serialize at
    # the layer boundary so a later campaign invocation waits for the helper
    # and then resumes from its atomic receipts instead of opening a second
    # CUDA writer for the same layer.  Keep the handle alive for all of main;
    # the kernel releases the advisory lock automatically on every exit path.
    recipe_lock = (layer_root / ".recipe.lock").open("a+", encoding="utf-8")
    fcntl.flock(recipe_lock.fileno(), fcntl.LOCK_EX)

    runtime = open_sqg_transcode_runtime(
        args.preflight.resolve(),
        model_root=args.source_sqg_root.resolve(),
        layer=args.layer,
        device=args.device,
        bit_map=_all_k3_map(args.layer),
        probe_rows=1024,
    )
    binding = runtime_binding(runtime, args)
    if args.smoke_cell is not None:
        by_id = {str(cell["cell_id"]): cell for cell in profile_cells(runtime)}
        if args.smoke_cell not in by_id:
            raise ValueError("smoke cell is not in the 16-cell profile panel")
        cell = by_id[args.smoke_cell]
        gate_profile, down_profile = realize_profile_cell(runtime, cell)
        global_h13, global_evidence = build_profile_global_h13(
            runtime,
            gate_profile,
            cell_id=str(cell["cell_id"]),
            chunk_rows=args.chunk_rows,
        )
        prepared = prepare_coupled_expert(
            runtime,
            expert=args.smoke_expert,
            gate_profile=gate_profile,
            down_profile=down_profile,
            global_h13=global_h13,
            global_evidence=global_evidence,
            triplets=balanced_profile_triplets(),
            qsrt_root=args.qsrt_root,
            chunk_rows=args.chunk_rows,
        )
        encoded_down, native_down, down_evidence = fit_coupled_down(
            prepared, beta=BOOTSTRAP_BETA
        )
        result = {
            "schema": "glm52-updated-qsrt-coupled-recipe-smoke-v1",
            "complete": True,
            "layer": args.layer,
            "expert": args.smoke_expert,
            "cell_id": args.smoke_cell,
            "beta": BOOTSTRAP_BETA,
            "score": score_coupled_candidates(
                prepared, native_down, score_role="selection"
            ),
            "payload_sha256": candidate_payloads(prepared, encoded_down),
            "down_objective_evidence_ids": {
                str((*pair, bits)): evidence["evidence_id"]
                for pair, by_bits in down_evidence.items()
                for bits, evidence in by_bits.items()
                if (*pair, bits) in balanced_profile_triplets()
            },
            "runtime_binding_id": binding["binding_id"],
        }
        result["smoke_id"] = canonical_sha256(result)
        prepared.release()
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    binding_path = layer_root / "runtime_binding.json"
    if binding_path.is_file():
        if load_json(binding_path) != binding:
            raise ValueError("coupled recipe runtime binding differs")
    else:
        atomic_json(binding_path, binding)
    bootstrap, bootstrap_holdout = run_profile_search(
        args,
        runtime,
        binding,
        layer_root=layer_root,
        stage="bootstrap_profile",
        beta=BOOTSTRAP_BETA,
    )
    _, beta_choice = run_beta_panel(
        args,
        runtime,
        binding,
        layer_root=layer_root,
        bootstrap_selection=bootstrap,
    )
    selected_beta = float(beta_choice["selected_beta"])
    if selected_beta == BOOTSTRAP_BETA:
        final_selection = bootstrap
        final_holdout = bootstrap_holdout
    else:
        final_selection, final_holdout = run_profile_search(
            args,
            runtime,
            binding,
            layer_root=layer_root,
            stage="final_profile",
            beta=selected_beta,
        )
    final_binding = publish_final_profile(
        args,
        layer_root=layer_root,
        selection=final_selection,
        beta_choice=beta_choice,
    )
    result: dict[str, Any] = {
        "schema": "glm52-updated-qsrt-coupled-no-shortcut-layer-recipe-v1",
        "complete": True,
        "layer": args.layer,
        "selected_beta": selected_beta,
        "selected_profile_id": final_selection["selection_id"],
        "profile_holdout_id": final_holdout["holdout_id"],
        "final_profile_binding_id": final_binding["binding_id"],
        "runtime_binding_id": binding["binding_id"],
        "no_b300_owner_speed_rescue": True,
        "no_fleet_beta_shortcut": True,
        "source_is_frozen_sqg_checkpoint": True,
        "official_bf16_weight_shards_read": False,
    }
    result["result_id"] = canonical_sha256(result)
    result_path = layer_root / "NO_SHORTCUT_COUPLED_RECIPE.json"
    if result_path.is_file():
        if load_json(result_path) != result:
            raise ValueError("coupled recipe result differs")
    else:
        atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
