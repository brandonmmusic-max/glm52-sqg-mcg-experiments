#!/usr/bin/env python3
"""Fail-closed rebind after the rank-zero Safetensors writer hotfix.

The first four workers completed fit-only preparation, then all failed before
publishing an expert mini-shard because ``torch_tensor_entry`` could not obtain
bytes from the new scalar int32 ``.sqg`` marker.  This tool preserves those
validated preparation payloads while rebinding their JSON envelopes to a new
preflight that differs only by the reviewed writer fix and this explicit
recovery contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from src.fresh_pipeline_common import (
    SELECTED_LAYERS,
    atomic_json,
    canonical_json_bytes,
    canonical_sha256,
    load_json_object,
    local_pipeline_code_provenance,
    sha256_file,
)

OLD_WRITER_SHA256 = "87ffca02aed36e5775535fc8db7f31d56186eaa6bdeb1df56f4868fd94359d6a"
NEW_WRITER_SHA256 = "4cfedd154ff0c19356c973770a1ca72f61f788c2cad6715cd38d01f238a834b3"
WRITER_RELATIVE_PATH = "safetensors_io.py"
EXPECTED_EXCEPTION = (
    "RuntimeError: self.dim() cannot be 0 to view Int as Byte "
    "(different element sizes)"
)


def _bound_id(value: dict[str, Any], key: str) -> str:
    observed = value.get(key)
    unsigned = dict(value)
    unsigned.pop(key, None)
    if not isinstance(observed, str) or canonical_sha256(unsigned) != observed:
        raise ValueError(f"invalid canonical {key}")
    return observed


def _json_file_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value) + b"\n").hexdigest()


def _replace_binding(
    value: dict[str, Any], *, old_preflight_id: str, new_preflight_id: str, layer: int
) -> None:
    binding = value.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("artifact lacks a binding object")
    if binding.get("layer") != layer or binding.get("preflight_id") != old_preflight_id:
        raise ValueError(f"layer {layer} artifact has an unexpected old binding")
    binding["preflight_id"] = new_preflight_id


def _assert_no_encoded_outputs(layer_root: Path) -> None:
    forbidden_roots = (
        layer_root / "experts",
        layer_root / "final",
        layer_root / "holdout",
        layer_root / "profile_search" / "cells",
    )
    found = [
        str(path)
        for root in forbidden_roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    ]
    if found:
        raise ValueError(
            "recovery is allowed only before the first encoded artifact; "
            f"found {found[:3]}"
        )


def build_recovery_plan(project_root: Path, output_root: Path) -> dict[str, Any]:
    preflight_path = output_root / "preflight.json"
    job_plan_path = output_root / "job_plan.json"
    old_preflight = load_json_object(preflight_path)
    old_preflight_id = _bound_id(old_preflight, "preflight_id")
    old_preflight_sha256 = sha256_file(preflight_path)

    current_code = local_pipeline_code_provenance(project_root)
    old_code = old_preflight.get("local_pipeline_code")
    if not isinstance(old_code, dict):
        raise ValueError("old preflight lacks local code provenance")
    if old_code.get("src_tree") != current_code.get("src_tree"):
        raise ValueError("src tree changed; scalar-writer recovery is not applicable")
    old_writer = old_code.get("bmmlaw_r7_encoder_tree", {})
    new_writer = current_code.get("bmmlaw_r7_encoder_tree", {})
    old_files = old_writer.get("files", {})
    new_files = new_writer.get("files", {})
    if set(old_files) != set(new_files):
        raise ValueError("writer file inventory changed")
    changed = sorted(name for name in old_files if old_files[name] != new_files[name])
    if changed != [WRITER_RELATIVE_PATH]:
        raise ValueError(f"unexpected writer changes: {changed}")
    if (
        old_files[WRITER_RELATIVE_PATH].get("sha256") != OLD_WRITER_SHA256
        or new_files[WRITER_RELATIVE_PATH].get("sha256") != NEW_WRITER_SHA256
    ):
        raise ValueError("scalar-writer old/new hash is not the reviewed pair")

    recovery_contract = {
        "schema": "glm52-fresh-sqg-scalar-marker-hotfix-contract-v1",
        "old_preflight_id": old_preflight_id,
        "old_preflight_sha256": old_preflight_sha256,
        "changed_code_files": [f"bmmlaw_r7_encoder/{WRITER_RELATIVE_PATH}"],
        "old_writer_sha256": OLD_WRITER_SHA256,
        "new_writer_sha256": NEW_WRITER_SHA256,
        "semantic_change": (
            "flatten rank-zero tensors only for byte extraction while retaining "
            "the original scalar shape in the Safetensors header"
        ),
        "trigger_exception": EXPECTED_EXCEPTION,
        "encoded_artifacts_present_before_recovery": 0,
        "preparation_payloads_recomputed": False,
    }
    new_preflight = copy.deepcopy(old_preflight)
    new_preflight.pop("preflight_id", None)
    new_preflight["local_pipeline_code"] = current_code
    new_preflight["serialization_hotfix_recovery"] = recovery_contract
    new_preflight_id = canonical_sha256(new_preflight)
    new_preflight["preflight_id"] = new_preflight_id
    new_preflight_sha256 = _json_file_sha256(new_preflight)

    updates: dict[Path, dict[str, Any]] = {}
    layer_records: dict[str, Any] = {}
    for layer in SELECTED_LAYERS:
        layer_root = output_root / f"layer_{layer:03d}"
        _assert_no_encoded_outputs(layer_root)
        preparation = layer_root / "preparation"
        json_paths = [
            preparation / "h13.json",
            preparation / "profile_scales.json",
            preparation / "preparation.json",
        ]
        permutations = sorted((preparation / "permutations").glob("expert-*.json"))
        if len(permutations) != 256:
            raise ValueError(f"layer {layer} has {len(permutations)} permutations")
        json_paths.extend(permutations)
        preregistration = layer_root / "profile_search" / "preregistration.json"
        json_paths.append(preregistration)

        for path in json_paths:
            value = load_json_object(path)
            if path.name.startswith("expert-"):
                _bound_id(value, "artifact_id")
            elif path == preregistration:
                _bound_id(value, "preregistration_id")
            _replace_binding(
                value,
                old_preflight_id=old_preflight_id,
                new_preflight_id=new_preflight_id,
                layer=layer,
            )
            if path.name.startswith("expert-"):
                value.pop("artifact_id", None)
                value["artifact_id"] = canonical_sha256(value)
            elif path == preregistration:
                value.pop("preregistration_id", None)
                value["preregistration_id"] = canonical_sha256(value)
            updates[path] = value

        h13 = load_json_object(preparation / "h13.json")
        scales = load_json_object(preparation / "profile_scales.json")
        h13_shard = preparation / str(h13["shard"])
        scale_shard = preparation / str(scales["shard"])
        if sha256_file(h13_shard) != h13["shard_sha256"]:
            raise ValueError(f"layer {layer} H13 shard changed")
        if sha256_file(scale_shard) != scales["shard_sha256"]:
            raise ValueError(f"layer {layer} profile-scale shard changed")
        layer_records[str(layer)] = {
            "rebound_json_files": len(json_paths),
            "permutations": len(permutations),
            "h13_shard_sha256": h13["shard_sha256"],
            "profile_scales_shard_sha256": scales["shard_sha256"],
            "encoded_artifacts_before_recovery": 0,
        }

    old_job_plan = load_json_object(job_plan_path)
    old_job_plan_id = _bound_id(old_job_plan, "job_plan_id")
    if (
        old_job_plan.get("preflight_id") != old_preflight_id
        or old_job_plan.get("preflight_sha256") != old_preflight_sha256
        or old_job_plan.get("model_workload_launched") is not False
    ):
        raise ValueError("old job plan is not the inert plan for this preflight")
    new_job_plan = copy.deepcopy(old_job_plan)
    new_job_plan.pop("job_plan_id", None)
    new_job_plan["preflight_id"] = new_preflight_id
    new_job_plan["preflight_sha256"] = new_preflight_sha256
    new_job_plan["job_plan_id"] = canonical_sha256(new_job_plan)

    return {
        "schema": "glm52-fresh-sqg-scalar-marker-hotfix-recovery-v1",
        "complete": True,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "old_preflight_id": old_preflight_id,
        "old_preflight_sha256": old_preflight_sha256,
        "new_preflight_id": new_preflight_id,
        "new_preflight_sha256": new_preflight_sha256,
        "old_local_code_provenance_id": old_code.get("provenance_id"),
        "new_local_code_provenance_id": current_code.get("provenance_id"),
        "source_tree_unchanged": True,
        "changed_code_files": recovery_contract["changed_code_files"],
        "old_writer_sha256": OLD_WRITER_SHA256,
        "new_writer_sha256": NEW_WRITER_SHA256,
        "trigger_exception": EXPECTED_EXCEPTION,
        "encoded_artifacts_before_recovery": 0,
        "preparation_payloads_recomputed": False,
        "layers": layer_records,
        "rebound_json_file_count": len(updates),
        "old_job_plan_id": old_job_plan_id,
        "new_job_plan_id": new_job_plan["job_plan_id"],
        "_updates": updates,
        "_new_preflight": new_preflight,
        "_new_job_plan": new_job_plan,
    }


def apply_recovery(plan: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(plan["output_root"])
    archive_root = output_root / "recovery" / "pre_scalar_marker_hotfix"
    if archive_root.exists():
        raise FileExistsError(f"recovery archive already exists: {archive_root}")
    archive_root.mkdir(parents=True)

    paths = [output_root / "preflight.json", output_root / "job_plan.json"]
    paths.extend(plan["_updates"])
    archive_hashes: dict[str, str] = {}
    for path in paths:
        relative = path.relative_to(output_root)
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if sha256_file(destination) != sha256_file(path):
            raise IOError(f"recovery archive copy differs: {relative}")
        archive_hashes[relative.as_posix()] = sha256_file(destination)

    for path, value in plan["_updates"].items():
        atomic_json(path, value, overwrite=True)
    atomic_json(output_root / "preflight.json", plan["_new_preflight"], overwrite=True)
    atomic_json(output_root / "job_plan.json", plan["_new_job_plan"], overwrite=True)

    receipt = {key: value for key, value in plan.items() if not key.startswith("_")}
    receipt["archive_root"] = str(archive_root)
    receipt["archive_file_count"] = len(archive_hashes)
    receipt["archive_manifest_sha256"] = canonical_sha256(archive_hashes)
    receipt["recovery_id"] = canonical_sha256(receipt)
    receipt_path = output_root / "recovery" / "scalar_marker_hotfix_recovery.json"
    atomic_json(receipt_path, receipt)
    if sha256_file(output_root / "preflight.json") != receipt["new_preflight_sha256"]:
        raise RuntimeError("published preflight hash differs from recovery plan")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan = build_recovery_plan(args.project_root.resolve(), args.output_root.resolve())
    if args.apply:
        result = apply_recovery(plan)
    else:
        result = {key: value for key, value in plan.items() if not key.startswith("_")}
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
