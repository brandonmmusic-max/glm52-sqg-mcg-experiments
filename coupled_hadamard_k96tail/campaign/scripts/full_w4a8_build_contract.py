#!/usr/bin/env python3
"""Canonical source seal and per-layer beta lineage for the full W4A8 build.

The full model is not permitted to inherit layer 77's beta as a fleet-wide
constant.  Each routed layer must select beta on its own fit-only secondary
split, then freeze that decision before the final W4A8-native profile search.

This module supplies two immutable artifacts:

* a relocation-neutral hash of every executable code tree and stage launcher
  that can affect W4A8 bytes; and
* a per-layer beta-choice receipt binding the fit-only panel/selection, the
  bootstrap profile used by that panel, and the exact full-build source seal.

Both artifacts are canonical JSON.  Consumers re-hash current source and fail
closed before doing GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fresh_pipeline_common import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    load_json_object,
    sha256_file,
)


BUILD_SCHEMA = "glm52-sqg-full-w4a8-build-source-v1"
BETA_CHOICE_SCHEMA = "glm52-sqg-full-w4a8-layer-beta-choice-v1"
OWNER_FIXED_BETA_SCHEMA = "glm52-sqg-full-w4a8-owner-fixed-beta-choice-v1"
FINAL_PROFILE_SCHEMA = "glm52-sqg-full-w4a8-final-profile-binding-v1"
BETA_PANEL_SCHEMA = "glm52-sqg-full-w4a8-beta-panel-v2"
BETA_SELECTION_SCHEMA = "glm52-sqg-full-w4a8-beta-selection-v2"
BETAS = (0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0)
BOOTSTRAP_BETA = 0.0625
OWNER_FIXED_BETA = 0.25
W4A8_PROFILE_SCHEMA = "glm52-full-w4a8-native-profile-selection-v1"
ROUTED_LAYER_COUNT = 76
ROUTED_MATRIX_COUNT = 58_368
NONROUTED_K6_ROLES = (
    "mlp.shared_experts.gate_proj",
    "mlp.shared_experts.up_proj",
    "mlp.shared_experts.down_proj",
    "self_attn.q_b_proj",
    "self_attn.o_proj",
)
NONROUTED_K6_MATRIX_COUNT = ROUTED_LAYER_COUNT * len(NONROUTED_K6_ROLES)
TOTAL_SQG_MATRIX_COUNT = ROUTED_MATRIX_COUNT + NONROUTED_K6_MATRIX_COUNT

# These named stage files make the principal dependency surface easy to audit.
# They are also covered by the complete scripts tree below; the duplication is
# deliberate and prevents a broad tree hash from obscuring the main entrypoints.
STAGE_FILES = (
    "b300_remote/adapt_r10_capture_to_sqg.py",
    "b300_remote/build_full_source_seal_from_r10_inventory.py",
    "b300_remote/build_sm103_encoder_deps.py",
    "b300_remote/campaign_supervisor.py",
    "b300_remote/capture_r10_b300.py",
    "b300_remote/derive_r10_bf16_runtime_inventory.py",
    "b300_remote/hessian_dataset_upload.py",
    "b300_remote/run_r10_capture_8b300.sh",
    "b300_remote/run_layer_stage.py",
    "scripts/allocate_sqg_k34_layer.py",
    "scripts/encode_final_shard.py",
    "scripts/encode_full_w4a8_selected_layer.py",
    "scripts/full_w4a8_build_contract.py",
    "scripts/full_w4a8_production_lineage_v2.py",
    "scripts/materialize_progressive_w4a8_candidate.py",
    "scripts/materialize_progressive_w4a8_candidate_v2.py",
    "scripts/profile_search_full_w4a8_native.py",
    "scripts/run_full_w4a8_beta_panel.sh",
    "scripts/run_full_w4a8_selected_wave.sh",
    "scripts/run_full_w4a8_selected_wave_v2.sh",
    "scripts/run_sqg_w4a8_triplet_layer.sh",
    "scripts/run_sqg_w4a8_triplet_layer_v2.sh",
    "scripts/run_wave_full_w4a8_native_profile_search.sh",
    "scripts/score_glm52_w4a8_activation_quality.py",
    "scripts/score_sqg_w4a8_triplet_candidates.py",
    "scripts/select_full_w4a8_beta_panel.py",
    "scripts/w4a8_cross_term.py",
    "scripts/w4a8_stable_solve.py",
    "scripts/validate_full_sqg_checkpoint.py",
)

# Hash complete executable trees so a newly imported helper cannot fall outside
# the source binding merely because a hand-maintained import list was stale.
CODE_TREES = (
    "b300_remote",
    "scripts",
    "src",
    "bmmlaw_r7_encoder",
    "kquant/kquant",
)

# Progressive recapture begins outside ``scripts/``.  Bind its actual top-level
# producer and launcher explicitly; tests/docs/cache remain outside the seal.
TOP_LEVEL_EXECUTABLES = (
    "capture_calibration.py",
    "run_capture_container.sh",
)

IGNORED_PARTS = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "docs", "test", "tests"}
)
IGNORED_SOURCE_SUFFIXES = frozenset({".md", ".rst"})


def _canonical_id(value: Mapping[str, Any], field: str) -> str:
    material = dict(value)
    observed = material.pop(field, None)
    if not isinstance(observed, str) or observed != canonical_sha256(material):
        raise ValueError(f"canonical {field} differs")
    return observed


def _pretty_canonical_id(value: Mapping[str, Any], field: str) -> str:
    """Validate manifests owned by the v3 allocation/score JSON contract."""

    material = dict(value)
    observed = material.pop(field, None)
    payload = (json.dumps(material, indent=2, sort_keys=True) + "\n").encode("utf-8")
    expected = hashlib.sha256(payload).hexdigest()
    if not isinstance(observed, str) or observed != expected:
        raise ValueError(f"canonical {field} differs")
    return observed


def _portable_tree(root: Path, relative_root: str) -> dict[str, Any]:
    directory = (root / relative_root).resolve()
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError(f"executable code tree is absent or unsafe: {relative_root}")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"executable code tree contains a symlink: {relative_root}/{relative}")
        if (
            not path.is_file()
            or path.suffix in {".pyc", ".pyo"}
            or path.suffix.lower() in IGNORED_SOURCE_SUFFIXES
        ):
            continue
        files[relative.as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if not files:
        raise ValueError(f"executable code tree is empty: {relative_root}")
    result: dict[str, Any] = {
        "root": relative_root,
        "file_count": len(files),
        "files": files,
    }
    result["tree_sha256"] = canonical_sha256(result)
    return result


def build_source_binding(
    project_root: Path,
    *,
    predecessor_preflight: Path | None = None,
    candidate_full_closure: Path | None = None,
) -> dict[str, Any]:
    """Return the exact relocation-neutral source identity for a full build."""

    project_root = project_root.resolve()
    stage_files: dict[str, dict[str, Any]] = {}
    for relative in STAGE_FILES:
        path = project_root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"full-W4A8 stage input is absent or unsafe: {relative}")
        stage_files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    top_level: dict[str, dict[str, Any]] = {}
    for relative in TOP_LEVEL_EXECUTABLES:
        path = project_root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"top-level recapture executable is absent or unsafe: {relative}"
            )
        top_level[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    preflight_binding: dict[str, Any] | None = None
    if predecessor_preflight is not None:
        preflight_path = predecessor_preflight.resolve()
        preflight = load_json_object(preflight_path)
        preflight_binding = {
            "sha256": sha256_file(preflight_path),
            "preflight_id": _canonical_id(preflight, "preflight_id"),
            "source_seal": preflight.get("source_seal"),
            "kquant": preflight.get("kquant"),
            "exllamav3": preflight.get("exllamav3"),
        }
    candidate_binding: dict[str, Any] | None = None
    if candidate_full_closure is not None:
        candidate_path = candidate_full_closure.resolve()
        candidate = load_json_object(candidate_path)
        if (
            candidate.get("schema")
            != "glm52-sqg-candidate-full-closure-preflight-v1"
            or candidate.get("complete") is not True
            or candidate.get("passed") is not True
            or candidate.get("selected_model_eligible") is not True
            or candidate.get("candidate_sweep_fast_requested") is not False
            or candidate.get("raw_bytes_roundtrip_exact") is not True
            or candidate.get("native_e4m3_exact") is not True
        ):
            raise ValueError("candidate full-closure receipt differs")
        candidate_binding = {
            "sha256": sha256_file(candidate_path),
            "receipt_id": _canonical_id(candidate, "receipt_id"),
            "schema": candidate["schema"],
            "receipt_count": candidate.get("receipt_count"),
            "role_classes": candidate.get("role_classes"),
            "rates": candidate.get("rates"),
        }
    result: dict[str, Any] = {
        "schema": BUILD_SCHEMA,
        "complete": True,
        "scope": "full_glm52_native_sqg_w4a8_construction",
        "stage_files": stage_files,
        "top_level_executables": top_level,
        "code_trees": {
            relative: _portable_tree(project_root, relative) for relative in CODE_TREES
        },
        "predecessor_preflight": preflight_binding,
        "candidate_full_closure": candidate_binding,
        "arithmetic_contract": {
            "activation_endpoint": "full-w4a8",
            "h_a8": "mxfp8_e4m3_ue8m0_k32",
            "act_a8": "mxfp8_e4m3_ue8m0_k32",
            "weight_endpoint": "native_exact_e4m3_labels",
            "activation": "silu(gate)*up",
            "accumulation": "fp32",
            "candidate_specific_caller_coordinate_h_b": True,
            "private_down_suh_anchored_by_rate": True,
        },
        "format_contract": {
            "routed_rates": "independent_per_tensor_k3_k4",
            "uniform_k3": False,
            "routed_layers": [3, 78],
            "routed_layer_count": ROUTED_LAYER_COUNT,
            "routed_matrix_count": ROUTED_MATRIX_COUNT,
            "per_routed_layer": {"k3": 384, "k4": 384, "total": 768},
            "mtp78_included": True,
            "nonrouted": {
                "bits": 6,
                "layers": [3, 78],
                "layer_count": ROUTED_LAYER_COUNT,
                "matrix_count": NONROUTED_K6_MATRIX_COUNT,
                "mtp78_included": True,
                "per_layer": list(NONROUTED_K6_ROLES),
            },
            "total_sqg_matrix_count": TOTAL_SQG_MATRIX_COUNT,
            "sole_checkpoint_source": "official_bf16",
            "standard_ignored_tensors_may_remain_direct_bf16": True,
            "topology_neutral": True,
            "mcg_payloads_scales_or_transforms": 0,
            "storage": "glm52_sqg_atoms_v2",
        },
        "beta_policy": {
            "scope": "per_layer",
            "fleet_wide_layer77_beta": False,
            "selection_role": "fit/allocation",
            "construction_role": "fit/calibration",
            "final_profile_search_after_beta_freeze": True,
        },
    }
    result["binding_id"] = canonical_sha256(result)
    return result


def validate_source_binding(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = load_json_object(path)
    format_contract = observed.get("format_contract", {})
    nonrouted = (
        format_contract.get("nonrouted", {})
        if isinstance(format_contract, Mapping)
        else {}
    )
    if (
        observed.get("schema") != BUILD_SCHEMA
        or observed.get("complete") is not True
        or observed.get("beta_policy", {}).get("scope") != "per_layer"
        or not isinstance(format_contract, Mapping)
        or format_contract.get("uniform_k3") is not False
        or format_contract.get("mtp78_included") is not True
        or int(format_contract.get("routed_layer_count", -1)) != ROUTED_LAYER_COUNT
        or int(format_contract.get("routed_matrix_count", -1)) != ROUTED_MATRIX_COUNT
        or not isinstance(nonrouted, Mapping)
        or nonrouted.get("layers") != [3, 78]
        or int(nonrouted.get("layer_count", -1)) != ROUTED_LAYER_COUNT
        or int(nonrouted.get("matrix_count", -1)) != NONROUTED_K6_MATRIX_COUNT
        or nonrouted.get("mtp78_included") is not True
        or tuple(nonrouted.get("per_layer", ())) != NONROUTED_K6_ROLES
        or int(format_contract.get("total_sqg_matrix_count", -1))
        != TOTAL_SQG_MATRIX_COUNT
    ):
        raise ValueError("full-W4A8 build source contract differs")
    _canonical_id(observed, "binding_id")
    preflight = observed.get("predecessor_preflight")
    candidate = observed.get("candidate_full_closure")
    # Rebuild without an external path while preserving the already sealed
    # predecessor identity; executable source equality is the live gate.
    current = build_source_binding(project_root)
    current["predecessor_preflight"] = preflight
    current["candidate_full_closure"] = candidate
    current.pop("binding_id", None)
    current["binding_id"] = canonical_sha256(current)
    if current != observed:
        raise ValueError("current full-W4A8 executable source differs from its seal")
    return observed


def _validate_beta_panel(panel: Mapping[str, Any], *, layer: int) -> str:
    panel_id = _pretty_canonical_id(panel, "panel_id")
    contract = panel.get("subfold_contract")
    rate = panel.get("rate_balance")
    if (
        panel.get("schema") != BETA_PANEL_SCHEMA
        or panel.get("complete") is not True
        or int(panel.get("layer", -1)) != layer
        or panel.get("purpose") != "fit_only_hyperparameter_selection"
        or panel.get("role") != "fit"
        or panel.get("calibration_subfold") != "fit/calibration"
        or panel.get("allocation_subfold") != "fit/allocation"
        or panel.get("selection_used") is not False
        or panel.get("holdout_used") is not False
        or int(panel.get("mcg_inputs", -1)) != 0
        or tuple(float(value) for value in panel.get("betas", ())) != BETAS
        or int(panel.get("panel_size", -1)) != 16
        or not isinstance(contract, Mapping)
        or contract.get("document_disjoint") is not True
        or contract.get("selection_used") is not False
        or contract.get("holdout_used") is not False
        or not isinstance(rate, Mapping)
        or rate.get("independent_per_tensor_k3_k4") is not True
        or rate.get("uniform_k3") is not False
        or float(rate.get("realized_bpw", math.nan)) != 3.5
    ):
        raise ValueError("fit-only per-layer beta panel contract differs")
    profile = panel.get("profile_selection")
    if (
        not isinstance(profile, Mapping)
        or not isinstance(profile.get("sha256"), str)
        or len(profile["sha256"]) != 64
        or not isinstance(profile.get("selection_id"), str)
        or len(profile["selection_id"]) != 64
    ):
        raise ValueError("beta panel bootstrap-profile binding differs")
    return panel_id


def _validate_beta_selection(
    selection: Mapping[str, Any], *, layer: int, panel: Mapping[str, Any], panel_path: Path
) -> str:
    selection_id = _pretty_canonical_id(selection, "selection_id")
    panel_binding = selection.get("panel_binding")
    aggregate = selection.get("aggregate")
    winner = float(selection.get("winner_beta", math.nan))
    if (
        selection.get("schema") != BETA_SELECTION_SCHEMA
        or selection.get("complete") is not True
        or int(selection.get("layer", -1)) != layer
        or selection.get("purpose") != "fit_only_hyperparameter_selection"
        or selection.get("selection_used") is not False
        or selection.get("holdout_used") is not False
        or int(selection.get("mcg_inputs", -1)) != 0
        or not isinstance(panel_binding, Mapping)
        or panel_binding.get("panel_id") != panel.get("panel_id")
        or panel_binding.get("sha256") != sha256_file(panel_path)
        or not isinstance(aggregate, list)
        or len(aggregate) != len(BETAS)
        or winner not in BETAS
    ):
        raise ValueError("per-layer beta selection contract differs")
    rows = {float(row["beta"]): row for row in aggregate}
    if set(rows) != set(BETAS):
        raise ValueError("beta selection aggregate grid differs")
    expected = min(BETAS, key=lambda beta: (float(rows[beta]["fit_allocation_sse"]), beta))
    if winner != expected:
        raise ValueError("beta winner is not the deterministic fit-only minimum")
    return selection_id


def _source_hash(build: Mapping[str, Any], relative: str) -> str:
    stage = build.get("stage_files", {}).get(relative)
    if isinstance(stage, Mapping) and isinstance(stage.get("sha256"), str):
        return str(stage["sha256"])
    for tree_root, tree in build.get("code_trees", {}).items():
        prefix = f"{tree_root}/"
        if relative.startswith(prefix):
            record = tree.get("files", {}).get(relative[len(prefix) :])
            if isinstance(record, Mapping) and isinstance(record.get("sha256"), str):
                return str(record["sha256"])
    raise ValueError(f"beta panel code is outside the full-build source seal: {relative}")


def build_beta_choice(
    *,
    project_root: Path,
    build_binding_path: Path,
    selection_path: Path,
    panel_path: Path,
    bootstrap_profile_path: Path,
    layer: int,
) -> dict[str, Any]:
    """Freeze one layer's beta without using selection-role or holdout rows."""

    build = validate_source_binding(build_binding_path, project_root=project_root)
    panel = load_json_object(panel_path)
    panel_id = _validate_beta_panel(panel, layer=layer)
    panel_code = panel.get("code_binding")
    if not isinstance(panel_code, Mapping) or not panel_code:
        raise ValueError("beta panel executable code binding is absent")
    for relative, digest in panel_code.items():
        if not isinstance(relative, str) or digest != _source_hash(build, relative):
            raise ValueError(f"beta panel code differs from full-build source: {relative}")
    selection = load_json_object(selection_path)
    selection_id = _validate_beta_selection(
        selection, layer=layer, panel=panel, panel_path=panel_path
    )
    profile = panel["profile_selection"]
    bootstrap = load_json_object(bootstrap_profile_path)
    bootstrap_id = bootstrap.get("selection_id", bootstrap.get("selected_cell_id"))
    bootstrap_arithmetic = bootstrap.get("arithmetic")
    if (
        sha256_file(bootstrap_profile_path) != profile["sha256"]
        or bootstrap_id != profile["selection_id"]
        or bootstrap.get("schema") != W4A8_PROFILE_SCHEMA
        or not isinstance(bootstrap_arithmetic, Mapping)
        or float(bootstrap_arithmetic.get("beta", math.nan)) != BOOTSTRAP_BETA
        or bootstrap.get("holdout_used_for_choice") is not False
        or int(bootstrap.get("mcg_inputs", -1)) != 0
    ):
        raise ValueError("beta panel bootstrap profile bytes differ")
    result: dict[str, Any] = {
        "schema": BETA_CHOICE_SCHEMA,
        "complete": True,
        "layer": layer,
        "selected_beta": float(selection["winner_beta"]),
        "selection_rule": "minimum_fit_allocation_raw_sse_then_smaller_beta",
        "data_roles": {
            "h_b_construction": "fit/calibration",
            "beta_choice": "fit/allocation",
            "selection_role_used": False,
            "holdout_used": False,
            "document_disjoint_secondary_split": True,
        },
        "source_selection": {
            "filename": selection_path.name,
            "sha256": sha256_file(selection_path),
            "selection_id": selection_id,
        },
        "source_panel": {
            "filename": panel_path.name,
            "sha256": sha256_file(panel_path),
            "panel_id": panel_id,
        },
        "bootstrap_profile_selection": {
            "filename": bootstrap_profile_path.name,
            "sha256": sha256_file(bootstrap_profile_path),
            "selection_id": bootstrap_id,
            "selected_cell_id": bootstrap.get("selected_cell_id"),
            "beta": BOOTSTRAP_BETA,
            "purpose": "beta_panel_fixed_topology_input_only",
        },
        "full_build_source": {
            "filename": build_binding_path.name,
            "sha256": sha256_file(build_binding_path),
            "binding_id": build["binding_id"],
        },
        "dependency_order": [
            "bootstrap_profile_selection",
            "fit_only_beta_panel",
            "frozen_beta_choice",
            "final_w4a8_native_profile_search",
            "triplet_allocation",
            "selected_encode",
            "atoms_v2_materialization",
        ],
        "final_profile_policy": {
            "if_selected_beta_equals_bootstrap": (
                "reuse_bootstrap_selection_bytes_with_exact_identity_proof"
            ),
            "otherwise": "run_exactly_one_final_w4a8_native_profile_search",
            "beta_may_be_reopened": False,
            "profile_may_be_reselected_at_equal_beta": False,
        },
        "fleet_wide_inference_from_layer77": False,
        "uniform_k3": False,
        "mcg_inputs": 0,
    }
    result["choice_id"] = canonical_sha256(result)
    return result


def validate_beta_choice(
    path: Path,
    *,
    project_root: Path,
    build_binding_path: Path,
    layer: int,
) -> dict[str, Any]:
    value = load_json_object(path)
    choice_id = _canonical_id(value, "choice_id")
    build = validate_source_binding(build_binding_path, project_root=project_root)
    source = value.get("full_build_source")
    roles = value.get("data_roles")
    owner_fixed = value.get("schema") == OWNER_FIXED_BETA_SCHEMA
    if (
        value.get("schema") not in (BETA_CHOICE_SCHEMA, OWNER_FIXED_BETA_SCHEMA)
        or value.get("complete") is not True
        or int(value.get("layer", -1)) != layer
        or float(value.get("selected_beta", math.nan)) not in BETAS
        or value.get("fleet_wide_inference_from_layer77") is not False
        or value.get("uniform_k3") is not False
        or int(value.get("mcg_inputs", -1)) != 0
        or not isinstance(source, Mapping)
        or source.get("sha256") != sha256_file(build_binding_path)
        or source.get("binding_id") != build["binding_id"]
        or not isinstance(roles, Mapping)
        or roles.get("selection_role_used") is not False
        or roles.get("holdout_used") is not False
        or roles.get("document_disjoint_secondary_split") is not (not owner_fixed)
    ):
        raise ValueError("frozen per-layer beta choice differs")
    if owner_fixed and (
        float(value.get("selected_beta", math.nan)) != OWNER_FIXED_BETA
        or value.get("selection_rule") != "owner_directed_speed_recovery_fixed_0p25"
        or value.get("seven_beta_panel_bypassed") is not True
        or value.get("calibration_blend")
        != {"layer_global": 0.75, "expert_local": 0.25}
    ):
        raise ValueError("owner-fixed beta recovery contract differs")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "choice_id": choice_id,
        "layer": layer,
        "selected_beta": float(value["selected_beta"]),
        "source_selection_id": value.get("source_selection", {}).get("selection_id"),
        "source_panel_id": value.get("source_panel", {}).get("panel_id"),
        "bootstrap_profile_selection": value["bootstrap_profile_selection"],
        "full_build_binding_id": build["binding_id"],
    }


def build_owner_fixed_beta_choice(
    *, project_root: Path, build_binding_path: Path,
    bootstrap_profile_path: Path, layer: int,
) -> dict[str, Any]:
    """Freeze the explicit paid-node recovery beta without fabricating a panel."""

    build = validate_source_binding(build_binding_path, project_root=project_root)
    profile = load_json_object(bootstrap_profile_path)
    selection_id = _canonical_id(profile, "selection_id")
    rescue = profile.get("rescue_policy")
    arithmetic = profile.get("arithmetic")
    if (
        profile.get("schema") != W4A8_PROFILE_SCHEMA
        or int(profile.get("layer", -1)) != layer
        or not isinstance(arithmetic, Mapping)
        or float(arithmetic.get("beta", math.nan)) != OWNER_FIXED_BETA
        or not isinstance(rescue, Mapping)
        or rescue.get("schema") != "glm52-w4a8-identity-only-owner-speed-rescue-v1"
        or rescue.get("fake_or_imputed_scores") != 0
        or profile.get("holdout_used_for_choice") is not False
        or int(profile.get("mcg_inputs", -1)) != 0
    ):
        raise ValueError("owner-fixed beta requires the sealed identity rescue selection")
    value: dict[str, Any] = {
        "schema": OWNER_FIXED_BETA_SCHEMA,
        "complete": True,
        "layer": layer,
        "selected_beta": OWNER_FIXED_BETA,
        "selection_rule": "owner_directed_speed_recovery_fixed_0p25",
        "seven_beta_panel_bypassed": True,
        "calibration_blend": {"layer_global": 0.75, "expert_local": 0.25},
        "data_roles": {
            "h_b_construction": "fit/calibration",
            "beta_choice": "owner_policy_not_data_selected",
            "selection_role_used": False,
            "holdout_used": False,
            "document_disjoint_secondary_split": False,
        },
        "bootstrap_profile_selection": {
            "filename": bootstrap_profile_path.name,
            "sha256": sha256_file(bootstrap_profile_path),
            "selection_id": selection_id,
            "selected_cell_id": profile.get("selected_cell_id"),
            "beta": OWNER_FIXED_BETA,
            "purpose": "identity_only_topology_selection_at_fixed_production_beta",
        },
        "full_build_source": {
            "filename": build_binding_path.name,
            "sha256": sha256_file(build_binding_path),
            "binding_id": build["binding_id"],
        },
        "fleet_wide_inference_from_layer77": False,
        "uniform_k3": False,
        "mcg_inputs": 0,
    }
    value["choice_id"] = canonical_sha256(value)
    return value


def build_final_profile_binding(
    *,
    project_root: Path,
    build_binding_path: Path,
    beta_choice_path: Path,
    profile_selection_path: Path,
    profile_preregistration_path: Path,
    layer: int,
) -> dict[str, Any]:
    """Close the final profile to beta, with exact bootstrap reuse at 0.0625."""

    beta = validate_beta_choice(
        beta_choice_path,
        project_root=project_root,
        build_binding_path=build_binding_path,
        layer=layer,
    )
    choice = load_json_object(beta_choice_path)
    selection = load_json_object(profile_selection_path)
    prereg = load_json_object(profile_preregistration_path)
    selection_id = _canonical_id(selection, "selection_id")
    preregistration_id = _canonical_id(prereg, "preregistration_id")
    arithmetic = selection.get("arithmetic")
    prereg_arithmetic = prereg.get("arithmetic")
    expected_choice = {
        "sha256": beta["sha256"],
        "choice_id": beta["choice_id"],
        "layer": layer,
        "selected_beta": beta["selected_beta"],
        "full_build_binding_id": beta["full_build_binding_id"],
    }
    owner_fixed = choice.get("schema") == OWNER_FIXED_BETA_SCHEMA
    reused = beta["selected_beta"] == BOOTSTRAP_BETA
    lineage_ok = (
        prereg.get("beta_mode") == "bootstrap_prior"
        and float(prereg.get("bootstrap_beta_prior", math.nan)) == BOOTSTRAP_BETA
        and prereg.get("beta_choice") is None
        if reused
        else (
            prereg.get("beta_choice") == expected_choice
            if not owner_fixed
            else prereg.get("beta_mode") == "owner_fixed_speed_recovery"
            and float(prereg.get("bootstrap_beta_prior", math.nan)) == BOOTSTRAP_BETA
            and prereg.get("beta_choice") is None
        )
    )
    if (
        selection.get("schema") != W4A8_PROFILE_SCHEMA
        or selection.get("complete") is not True
        or int(selection.get("layer", -1)) != layer
        or selection.get("holdout_used_for_choice") is not False
        or int(selection.get("mcg_inputs", -1)) != 0
        or selection.get("preregistration_id") != preregistration_id
        or selection.get("preregistration_sha256")
        != sha256_file(profile_preregistration_path)
        or not isinstance(arithmetic, Mapping)
        or not isinstance(prereg_arithmetic, Mapping)
        or float(arithmetic.get("beta", math.nan))
        != beta["selected_beta"]
        or float(prereg_arithmetic.get("beta", math.nan))
        != beta["selected_beta"]
        or not lineage_ok
    ):
        raise ValueError("final W4A8 profile/beta-choice binding differs")
    bootstrap = choice["bootstrap_profile_selection"]
    if reused and (
        sha256_file(profile_selection_path) != bootstrap["sha256"]
        or selection_id != bootstrap["selection_id"]
    ):
        raise ValueError(
            "selected beta equals bootstrap prior but profile bytes were reselected"
        )
    if not reused and not owner_fixed and sha256_file(profile_selection_path) == bootstrap["sha256"]:
        raise ValueError("changed beta reused the bootstrap profile bytes")
    if owner_fixed and (
        sha256_file(profile_selection_path) != bootstrap["sha256"]
        or selection_id != bootstrap["selection_id"]
    ):
        raise ValueError("owner-fixed beta changed the sealed rescue profile bytes")
    result: dict[str, Any] = {
        "schema": FINAL_PROFILE_SCHEMA,
        "complete": True,
        "layer": layer,
        "selected_beta": beta["selected_beta"],
        "mode": (
            "owner_fixed_beta_identity_rescue_profile_reuse"
            if owner_fixed
            else (
                "bootstrap_profile_byte_identity_reuse"
                if reused
                else "one_final_profile_search_at_selected_beta"
            )
        ),
        "beta_choice": expected_choice,
        "profile_selection": {
            "filename": profile_selection_path.name,
            "sha256": sha256_file(profile_selection_path),
            "selection_id": selection_id,
            "selected_cell_id": selection.get("selected_cell_id"),
        },
        "profile_preregistration": {
            "filename": profile_preregistration_path.name,
            "sha256": sha256_file(profile_preregistration_path),
            "preregistration_id": preregistration_id,
        },
        "bootstrap_identity_proved": reused,
        "fixed_beta_profile_identity_proved": owner_fixed,
        "owner_speed_recovery": owner_fixed,
        "beta_reopened": False,
        "holdout_used_for_choice": False,
        "mcg_inputs": 0,
    }
    result["profile_binding_id"] = canonical_sha256(result)
    return result


def validate_final_profile_binding(
    path: Path,
    *,
    project_root: Path,
    build_binding_path: Path,
    beta_choice_path: Path,
    layer: int,
) -> dict[str, Any]:
    value = load_json_object(path)
    binding_id = _canonical_id(value, "profile_binding_id")
    beta = validate_beta_choice(
        beta_choice_path,
        project_root=project_root,
        build_binding_path=build_binding_path,
        layer=layer,
    )
    choice = value.get("beta_choice")
    if (
        value.get("schema") != FINAL_PROFILE_SCHEMA
        or value.get("complete") is not True
        or int(value.get("layer", -1)) != layer
        or float(value.get("selected_beta", math.nan)) != beta["selected_beta"]
        or not isinstance(choice, Mapping)
        or choice.get("sha256") != beta["sha256"]
        or choice.get("choice_id") != beta["choice_id"]
        or value.get("beta_reopened") is not False
        or value.get("holdout_used_for_choice") is not False
        or int(value.get("mcg_inputs", -1)) != 0
    ):
        raise ValueError("frozen final profile binding differs")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "profile_binding_id": binding_id,
        "layer": layer,
        "selected_beta": beta["selected_beta"],
        "profile_selection": value["profile_selection"],
        "mode": value["mode"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal-build")
    seal.add_argument("--project-root", type=Path, required=True)
    seal.add_argument("--preflight", type=Path)
    seal.add_argument("--candidate-closure", type=Path)
    seal.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-build")
    verify.add_argument("--project-root", type=Path, required=True)
    verify.add_argument("--binding", type=Path, required=True)
    choose = subparsers.add_parser("seal-beta-choice")
    choose.add_argument("--project-root", type=Path, required=True)
    choose.add_argument("--build-binding", type=Path, required=True)
    choose.add_argument("--beta-selection", type=Path, required=True)
    choose.add_argument("--beta-panel", type=Path, required=True)
    choose.add_argument("--bootstrap-profile", type=Path, required=True)
    choose.add_argument("--layer", type=int, required=True)
    choose.add_argument("--output", type=Path, required=True)
    owner_choose = subparsers.add_parser("seal-owner-fixed-beta-choice")
    owner_choose.add_argument("--project-root", type=Path, required=True)
    owner_choose.add_argument("--build-binding", type=Path, required=True)
    owner_choose.add_argument("--bootstrap-profile", type=Path, required=True)
    owner_choose.add_argument("--layer", type=int, required=True)
    owner_choose.add_argument("--output", type=Path, required=True)
    check = subparsers.add_parser("verify-beta-choice")
    check.add_argument("--project-root", type=Path, required=True)
    check.add_argument("--build-binding", type=Path, required=True)
    check.add_argument("--beta-choice", type=Path, required=True)
    check.add_argument("--layer", type=int, required=True)
    final = subparsers.add_parser("seal-final-profile")
    final.add_argument("--project-root", type=Path, required=True)
    final.add_argument("--build-binding", type=Path, required=True)
    final.add_argument("--beta-choice", type=Path, required=True)
    final.add_argument("--profile-selection", type=Path, required=True)
    final.add_argument("--profile-preregistration", type=Path, required=True)
    final.add_argument("--layer", type=int, required=True)
    final.add_argument("--output", type=Path, required=True)
    final_check = subparsers.add_parser("verify-final-profile")
    final_check.add_argument("--project-root", type=Path, required=True)
    final_check.add_argument("--build-binding", type=Path, required=True)
    final_check.add_argument("--beta-choice", type=Path, required=True)
    final_check.add_argument("--final-profile-binding", type=Path, required=True)
    final_check.add_argument("--layer", type=int, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "seal-build":
        value = build_source_binding(
            args.project_root,
            predecessor_preflight=args.preflight,
            candidate_full_closure=args.candidate_closure,
        )
        atomic_json(args.output, value)
        result: Mapping[str, Any] = value
    elif args.command == "verify-build":
        result = validate_source_binding(args.binding, project_root=args.project_root)
    elif args.command == "seal-beta-choice":
        value = build_beta_choice(
            project_root=args.project_root,
            build_binding_path=args.build_binding,
            selection_path=args.beta_selection,
            panel_path=args.beta_panel,
            bootstrap_profile_path=args.bootstrap_profile,
            layer=args.layer,
        )
        atomic_json(args.output, value)
        result = value
    elif args.command == "seal-owner-fixed-beta-choice":
        value = build_owner_fixed_beta_choice(
            project_root=args.project_root,
            build_binding_path=args.build_binding,
            bootstrap_profile_path=args.bootstrap_profile,
            layer=args.layer,
        )
        atomic_json(args.output, value)
        result = value
    elif args.command == "verify-beta-choice":
        result = validate_beta_choice(
            args.beta_choice,
            project_root=args.project_root,
            build_binding_path=args.build_binding,
            layer=args.layer,
        )
    elif args.command == "seal-final-profile":
        value = build_final_profile_binding(
            project_root=args.project_root,
            build_binding_path=args.build_binding,
            beta_choice_path=args.beta_choice,
            profile_selection_path=args.profile_selection,
            profile_preregistration_path=args.profile_preregistration,
            layer=args.layer,
        )
        atomic_json(args.output, value)
        result = value
    else:
        result = validate_final_profile_binding(
            args.final_profile_binding,
            project_root=args.project_root,
            build_binding_path=args.build_binding,
            beta_choice_path=args.beta_choice,
            layer=args.layer,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
