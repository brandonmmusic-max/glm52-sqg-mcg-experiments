#!/usr/bin/env python3
"""Seal a no-rewrite successor for the shared-profile CUDA FP16 oracle fix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.fresh_pipeline_common import (
    SELECTED_LAYERS,
    atomic_json,
    canonical_sha256,
    load_json_object,
    local_pipeline_code_provenance,
    sha256_file,
)
from src.fresh_pipeline_runner import (
    LOCAL_CODE_RECOVERY_NAME,
    LOCAL_CODE_RECOVERY_SCHEMA,
    _local_code_changes,
)


EXPECTED_CHANGED_FILES = {
    "src/fresh_candidate_materializer.py",
    "src/fresh_pipeline_runner.py",
    "src/glm52_fresh_sqg/codec.py",
}
EXPECTED_EXCEPTION = (
    "RuntimeError: persisted shared input residual vector changed during encode"
)


def _bound_id(value: dict[str, object], field: str) -> str:
    observed = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    if not isinstance(observed, str) or observed != canonical_sha256(unsigned):
        raise ValueError(f"invalid canonical {field}")
    return observed


def _artifact_snapshot(output_root: Path) -> dict[str, object]:
    expert_count = 0
    score_count = 0
    progress_count = 0
    per_layer: dict[str, dict[str, int]] = {}
    for layer in SELECTED_LAYERS:
        layer_root = output_root / f"layer_{layer:03d}"
        layer_experts = 0
        layer_scores = 0
        layer_progress = 0
        for path in sorted(layer_root.rglob("layer-*-expert-*.json")):
            if path.name.endswith(".json.sha256"):
                continue
            expert_count += 1
            layer_experts += 1
        for path in sorted(layer_root.rglob("score.json")):
            score_count += 1
            layer_scores += 1
        for path in sorted(layer_root.rglob("progress.json")):
            progress_count += 1
            layer_progress += 1
        per_layer[str(layer)] = {
            "expert_artifacts": layer_experts,
            "scores": layer_scores,
            "progress_files": layer_progress,
        }
    return {
        "expert_artifacts": expert_count,
        "scores": score_count,
        "progress_files": progress_count,
        "payload_hashing_skipped_for_time_priority": True,
        "snapshot_is_informational_during_active_encoding": True,
        "layers": per_layer,
    }


def build_receipt(project_root: Path, output_root: Path) -> dict[str, object]:
    preflight_path = output_root / "preflight.json"
    preflight = load_json_object(preflight_path)
    predecessor_id = _bound_id(preflight, "preflight_id")
    old_local = preflight.get("local_pipeline_code")
    if not isinstance(old_local, dict):
        raise ValueError("predecessor preflight lacks local-code provenance")
    new_local = local_pipeline_code_provenance(project_root)
    changes = _local_code_changes(old_local, new_local)
    if {str(item["path"]) for item in changes} != EXPECTED_CHANGED_FILES:
        raise ValueError(f"unexpected local-code recovery changes: {changes}")
    receipt: dict[str, object] = {
        "schema": LOCAL_CODE_RECOVERY_SCHEMA,
        "complete": True,
        "predecessor_preflight_id": predecessor_id,
        "predecessor_preflight_sha256": sha256_file(preflight_path),
        "old_local_pipeline_code": old_local,
        "new_local_pipeline_code": new_local,
        "changed_code_files": changes,
        "trigger_exception": EXPECTED_EXCEPTION,
        "diagnosis": (
            "CPU and assigned-CUDA float32 expressions can round to adjacent "
            "FP16 values at a boundary; KQuant output was not mutated"
        ),
        "semantic_change": (
            "realize the unchanged KQuant shared-vector expression once on the "
            "assigned CUDA device and retain exact emitted FP16 bytes as the "
            "caller-owned downstream validation oracle; after the operator's "
            "time-priority instruction, freeze profile selection to the fully "
            "completed draws 0-3 while retaining the original 31-comparison "
            "Bonferroni threshold"
        ),
        "operator_time_priority": {
            "completed_draw_prefix": 4,
            "selected_factorial_cells_per_layer": 16,
            "remaining_preregistered_cells_skipped": 16,
            "routed_holdout_skipped": True,
            "routed_holdout_substitute_fabricated": False,
            "full_256_expert_treatment_still_required": True,
            "sqg_encoder_arithmetic_changed": False,
        },
        "preflight_rewritten": False,
        "preregistration_rewritten": False,
        "preparation_recomputed": False,
        "encoded_payloads_rewritten": False,
        "backend_arithmetic_changed": False,
        "hessian_or_calibration_changed": False,
        "exact_equality_retained": True,
        "existing_artifacts_before_recovery": _artifact_snapshot(output_root),
    }
    receipt["recovery_id"] = canonical_sha256(receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    receipt = build_receipt(project_root, output_root)
    if args.apply:
        destination = output_root / "recovery" / LOCAL_CODE_RECOVERY_NAME
        if destination.exists():
            raise FileExistsError(destination)
        atomic_json(destination, receipt)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
