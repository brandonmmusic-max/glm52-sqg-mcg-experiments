#!/usr/bin/env python3
"""Create a clean successor run after the gate/up absolute-scale fix.

This importer deliberately does not hash the BF16 shards or calibration
payloads.  Those expensive inputs remain bound by the predecessor preflight;
the importer checks their small seals/manifests and current file identities.
Only fit-only preparation is imported.  Every profile-derived artifact is
excluded because gate/up, candidate-conditional H2, and down must be rebuilt.
"""

from __future__ import annotations

import argparse
import copy
import errno
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_PREDECESSOR_TO_CURRENT = {
    "src/fresh_candidate_materializer.py",
    "src/fresh_pipeline_calibration.py",
    "src/fresh_pipeline_runner.py",
    "src/glm52_fresh_sqg/__init__.py",
    "src/glm52_fresh_sqg/codec.py",
}
EXPECTED_LAST_FIX_FILES = {
    "src/fresh_pipeline_calibration.py": {
        "old_sha256": "284019dca8479f9f1903018223ac32526e6332056405e597ab4f51d130a41ab1",
        "new_sha256": "c1929d772ef1f1e510b210f2c97111e15be2ed4f69caa6bf8dcec941f5d9c921",
    },
    "src/glm52_fresh_sqg/codec.py": {
        "old_sha256": "8c0848c1fdcabee72657729aaacf281b92774655a6c7c9011655bb74ae51ded5",
        "new_sha256": "0a4b7240b3a1cdd083f17750384ac031386db7aa193c138ecef49353fffdf27d",
    },
    "src/glm52_fresh_sqg/__init__.py": {
        "old_sha256": "b520e495087cf18e1ce9ea6483fc4ef7c0136fb0dbf7b4a35f0c7443130f92a3",
        "new_sha256": "c0583057b49237df183c67611b63763131c8b8624c91ecdc74b742b10adde933",
    },
}
PREDECESSOR_RECOVERY = Path("recovery/shared_profile_cuda_fp16_recovery.json")
REJECTED_CANDIDATE = Path(
    "/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-FRESH-SQG4-r1"
)


def _bound_id(value: dict[str, Any], field: str, canonical_sha256) -> str:
    observed = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    if not isinstance(observed, str) or observed != canonical_sha256(unsigned):
        raise ValueError(f"invalid canonical {field}")
    return observed


def _file_map(value: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for section, prefix in (
        ("src_tree", "src"),
        ("bmmlaw_r7_encoder_tree", "bmmlaw_r7_encoder"),
    ):
        tree = value.get(section)
        files = tree.get("files") if isinstance(tree, Mapping) else None
        if not isinstance(files, Mapping):
            raise ValueError(f"local provenance lacks {section}")
        for relative, record in files.items():
            if not isinstance(record, Mapping):
                raise ValueError(f"invalid local provenance record: {relative}")
            result[f"{prefix}/{relative}"] = record
    return result


def _changes(
    old: Mapping[str, object], new: Mapping[str, object]
) -> list[dict[str, object]]:
    old_files = _file_map(old)
    new_files = _file_map(new)
    return [
        {"path": path, "old": old_files.get(path), "new": new_files.get(path)}
        for path in sorted(set(old_files) | set(new_files))
        if old_files.get(path) != new_files.get(path)
    ]


def _validate_last_fix(
    previous: Mapping[str, object], current: Mapping[str, object]
) -> list[dict[str, object]]:
    changes = _changes(previous, current)
    if {str(item["path"]) for item in changes} != set(EXPECTED_LAST_FIX_FILES):
        raise ValueError(f"unexpected code changes after predecessor recovery: {changes}")
    for item in changes:
        expected = EXPECTED_LAST_FIX_FILES[str(item["path"])]
        if (
            item["old"].get("sha256") != expected["old_sha256"]
            or item["new"].get("sha256") != expected["new_sha256"]
        ):
            raise ValueError(f"unreviewed old/new code pair: {item['path']}")
    return changes


def _validate_small_inputs(preflight: Mapping[str, object], sha256_file) -> dict[str, object]:
    checked: dict[str, object] = {}
    for section, field in (
        ("source_seal", "sha256"),
        ("capture", "sha256"),
        ("bit_contract", "sha256"),
    ):
        record = preflight[section]
        path = Path(str(record["path"]))
        observed = sha256_file(path)
        if observed != record[field]:
            raise ValueError(f"predecessor small input drift: {section}")
        checked[section] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": observed,
        }

    source_seal = json.loads(Path(str(preflight["source_seal"]["path"])).read_text())
    shard_root = Path(str(source_seal["shard_root"]))
    expected_identities = preflight["source_seal"]["shard_file_identity"]
    observed_identities: dict[str, object] = {}
    for name, expected in sorted(expected_identities.items()):
        path = shard_root / name
        stat = path.stat()
        observed = {
            "bytes": stat.st_size,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
        }
        if observed != expected:
            raise ValueError(f"BF16 shard file identity drift without payload hash: {name}")
        observed_identities[name] = observed
    checked["bf16_shard_file_identities"] = {
        "count": len(observed_identities),
        "all_match_predecessor": True,
        "payload_hashing_performed": False,
    }
    checked["capture_payload_hashing_performed"] = False
    return checked


def _link_exact(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    sha256_file,
) -> str:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"preparation payload hash differs: {source}")
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        # Docker exposes the read-only predecessor and writable successor as
        # separate bind mounts.  Linux rejects a hardlink across that mount
        # boundary even when both host paths are on the same filesystem.
        shutil.copy2(source, destination, follow_symlinks=False)
        if sha256_file(destination) != expected_sha256:
            raise RuntimeError(f"cross-mount preparation copy differs: {destination}")
        return "copied_after_exdev"
    src = source.stat()
    dst = destination.stat()
    if (src.st_dev, src.st_ino) != (dst.st_dev, dst.st_ino):
        raise RuntimeError(f"preparation payload was not hardlinked: {destination}")
    return "hardlinked"


def _rewrite_binding(
    value: dict[str, Any],
    *,
    layer: int,
    old_preflight_id: str,
    new_preflight_id: str,
    run_id: str,
) -> None:
    binding = value.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("preparation JSON lacks binding")
    if (
        binding.get("layer") != layer
        or binding.get("preflight_id") != old_preflight_id
        or binding.get("run_id") != run_id
    ):
        raise ValueError(f"layer {layer} predecessor preparation binding differs")
    binding["preflight_id"] = new_preflight_id


def build_successor(
    predecessor_root: Path,
    output_root: Path,
) -> dict[str, object]:
    from src.fresh_pipeline_common import (
        SELECTED_LAYERS,
        canonical_sha256,
        load_json_object,
        local_pipeline_code_provenance,
        sha256_file,
    )
    from src.fresh_pipeline_runner import _local_code_changes

    predecessor_root = predecessor_root.resolve()
    output_root = output_root.resolve()
    if predecessor_root == output_root:
        raise ValueError("successor output must differ from rejected predecessor")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("successor output root must be empty")

    predecessor_path = predecessor_root / "preflight.json"
    predecessor = load_json_object(predecessor_path)
    predecessor_id = _bound_id(predecessor, "preflight_id", canonical_sha256)
    predecessor_sha256 = sha256_file(predecessor_path)
    run_id = str(predecessor["settings"]["run_id"])
    if run_id != "glm52-fresh-sqg-r1":
        raise ValueError("successor must retain the exact predecessor run_id/seeds")

    prior_recovery_path = predecessor_root / PREDECESSOR_RECOVERY
    prior_recovery = load_json_object(prior_recovery_path)
    prior_recovery_id = _bound_id(prior_recovery, "recovery_id", canonical_sha256)
    if (
        prior_recovery.get("predecessor_preflight_id") != predecessor_id
        or prior_recovery.get("predecessor_preflight_sha256") != predecessor_sha256
        or prior_recovery.get("old_local_pipeline_code")
        != predecessor.get("local_pipeline_code")
        or prior_recovery.get("changed_code_files")
        != _local_code_changes(
            predecessor["local_pipeline_code"],
            prior_recovery["new_local_pipeline_code"],
        )
    ):
        raise ValueError("predecessor recovery chain differs")

    current_code = local_pipeline_code_provenance(PROJECT_ROOT)
    last_fix_changes = _validate_last_fix(
        prior_recovery["new_local_pipeline_code"], current_code
    )
    all_changes = _changes(predecessor["local_pipeline_code"], current_code)
    if {str(item["path"]) for item in all_changes} != EXPECTED_PREDECESSOR_TO_CURRENT:
        raise ValueError(f"unexpected predecessor-to-current code changes: {all_changes}")
    small_inputs = _validate_small_inputs(predecessor, sha256_file)

    contract: dict[str, object] = {
        "schema": "glm52-fresh-sqg-absolute-gate-scale-successor-contract-v1",
        "predecessor_preflight_id": predecessor_id,
        "predecessor_preflight_sha256": predecessor_sha256,
        "predecessor_recovery_id": prior_recovery_id,
        "predecessor_recovery_sha256": sha256_file(prior_recovery_path),
        "run_id_retained_for_identical_seeds": run_id,
        "last_fix_code_changes": last_fix_changes,
        "all_predecessor_code_changes": all_changes,
        "diagnosis": (
            "KQuant input_channel_scale_profile is an absolute row-RMS; "
            "mean-one gate/up profiles caused saturated global-scale search"
        ),
        "semantic_fix": (
            "retain absolute aggregate gate/up RMS while applying only relative "
            "family modulation; down output profiles remain mean-one relative"
        ),
        "reuse": {
            "h13": True,
            "profile_scale_base_evidence": True,
            "fit_derived_permutations": True,
            "bf16_source": True,
            "capture": True,
            "bit_contract": True,
        },
        "invalidated": {
            "profile_search": True,
            "selected_profiles": True,
            "final_expert_shards": True,
            "consolidated_experts": True,
            "final_layers": True,
            "run_seal": True,
            "materialized_candidate": True,
            "candidate_kld": True,
        },
        "candidate_conditional_h2_rebuilt_after_gate_up": True,
        "down_reencoded_after_h2": True,
        "full_bf16_payload_hashing_performed": False,
        "full_capture_payload_hashing_performed": False,
        "small_input_validation": small_inputs,
        "rejected_artifacts_retained_read_only": str(predecessor_root),
        "rejected_candidate_retained": str(REJECTED_CANDIDATE),
    }
    contract["contract_id"] = canonical_sha256(contract)

    successor = copy.deepcopy(predecessor)
    successor.pop("preflight_id", None)
    successor["local_pipeline_code"] = current_code
    successor["absolute_gate_scale_successor"] = contract
    successor_id = canonical_sha256(successor)
    successor["preflight_id"] = successor_id

    return {
        "predecessor_root": predecessor_root,
        "output_root": output_root,
        "predecessor": predecessor,
        "predecessor_id": predecessor_id,
        "predecessor_sha256": predecessor_sha256,
        "successor": successor,
        "successor_id": successor_id,
        "run_id": run_id,
        "contract": contract,
        "prior_recovery_id": prior_recovery_id,
    }


def apply_successor(plan: Mapping[str, object]) -> dict[str, object]:
    from src.fresh_pipeline_common import (
        SELECTED_LAYERS,
        atomic_json,
        canonical_sha256,
        load_bit_contract,
        load_json_object,
        sha256_file,
    )
    from src.fresh_pipeline_runner import (
        LayerRuntime,
        PipelinePaths,
        PipelineSettings,
        _load_h13,
        _load_permutation,
        _load_scale_evidence,
        load_preflight,
        prepare_layer,
    )

    predecessor_root = Path(plan["predecessor_root"])
    output_root = Path(plan["output_root"])
    predecessor_id = str(plan["predecessor_id"])
    successor_id = str(plan["successor_id"])
    run_id = str(plan["run_id"])

    output_root.mkdir(parents=True, exist_ok=False) if not output_root.exists() else None
    atomic_json(output_root / "preflight.json", plan["successor"])

    layers: dict[str, object] = {}
    for layer in SELECTED_LAYERS:
        old_preparation = predecessor_root / f"layer_{layer:03d}" / "preparation"
        new_preparation = output_root / f"layer_{layer:03d}" / "preparation"
        new_preparation.mkdir(parents=True)

        h13_json = load_json_object(old_preparation / "h13.json")
        scale_json = load_json_object(old_preparation / "profile_scales.json")
        for value in (h13_json, scale_json):
            _rewrite_binding(
                value,
                layer=layer,
                old_preflight_id=predecessor_id,
                new_preflight_id=successor_id,
                run_id=run_id,
            )
        h13_transfer = _link_exact(
            old_preparation / str(h13_json["shard"]),
            new_preparation / str(h13_json["shard"]),
            expected_sha256=str(h13_json["shard_sha256"]),
            sha256_file=sha256_file,
        )
        scale_transfer = _link_exact(
            old_preparation / str(scale_json["shard"]),
            new_preparation / str(scale_json["shard"]),
            expected_sha256=str(scale_json["shard_sha256"]),
            sha256_file=sha256_file,
        )
        atomic_json(new_preparation / "h13.json", h13_json)
        atomic_json(new_preparation / "profile_scales.json", scale_json)

        old_permutations = old_preparation / "permutations"
        new_permutations = new_preparation / "permutations"
        new_permutations.mkdir()
        permutation_paths = sorted(old_permutations.glob("expert-*.json"))
        if len(permutation_paths) != 256:
            raise ValueError(f"layer {layer} predecessor permutation count differs")
        for old_path in permutation_paths:
            value = load_json_object(old_path)
            _bound_id(value, "artifact_id", canonical_sha256)
            _rewrite_binding(
                value,
                layer=layer,
                old_preflight_id=predecessor_id,
                new_preflight_id=successor_id,
                run_id=run_id,
            )
            value.pop("artifact_id")
            value["artifact_id"] = canonical_sha256(value)
            atomic_json(new_permutations / old_path.name, value)

        completion = load_json_object(old_preparation / "preparation.json")
        _rewrite_binding(
            completion,
            layer=layer,
            old_preflight_id=predecessor_id,
            new_preflight_id=successor_id,
            run_id=run_id,
        )
        atomic_json(new_preparation / "preparation.json", completion)
        layers[str(layer)] = {
            "h13_shard_sha256": h13_json["shard_sha256"],
            "profile_scales_shard_sha256": scale_json["shard_sha256"],
            "preparation_payload_transfers": {
                "h13": h13_transfer,
                "profile_scales": scale_transfer,
            },
            "rebound_permutations": 256,
            "profile_derived_artifacts_imported": 0,
        }

    # Full loader validates current code plus all small executable/input seals.
    loaded = load_preflight(output_root / "preflight.json")
    if loaded["preflight_id"] != successor_id:
        raise RuntimeError("successor preflight failed its own loader")
    runtime_paths = PipelinePaths.resolve(**loaded["paths"])
    runtime_settings = PipelineSettings(**loaded["settings"])
    bit_contract = load_bit_contract(runtime_paths.bit_contract)
    for layer in SELECTED_LAYERS:
        # Preparation validation is CPU-only and intentionally does not open or
        # hash BF16/capture payloads.  Real GPU/visibility binding is enforced
        # by the subsequent smoke and profile workers.
        runtime = LayerRuntime(
            paths=runtime_paths,
            settings=runtime_settings,
            layer=layer,
            device="cpu",
            source=None,  # type: ignore[arg-type]
            capture=None,  # type: ignore[arg-type]
            bit_map=bit_contract[layer].bit_map,
            source_seal=loaded["source_seal"],
            preflight=loaded,
        )
        _load_h13(runtime)
        _load_scale_evidence(runtime)
        for expert in range(256):
            _load_permutation(runtime, expert)
        prepare_layer(runtime)

    receipt: dict[str, object] = {
        "schema": "glm52-fresh-sqg-absolute-gate-scale-successor-receipt-v1",
        "complete": True,
        "predecessor_preflight_id": predecessor_id,
        "predecessor_preflight_sha256": plan["predecessor_sha256"],
        "predecessor_recovery_id": plan["prior_recovery_id"],
        "successor_preflight_id": successor_id,
        "successor_preflight_sha256": sha256_file(output_root / "preflight.json"),
        "contract_id": plan["contract"]["contract_id"],
        "run_id_retained_for_identical_seeds": run_id,
        "layers": layers,
        "validated_with_current_runtime": True,
        "profile_search_present": False,
        "expert_shards_present": False,
        "final_layers_present": False,
        "full_bf16_payload_hashing_performed": False,
        "full_capture_payload_hashing_performed": False,
    }
    receipt["receipt_id"] = canonical_sha256(receipt)
    atomic_json(output_root / "successor_preflight_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan = build_successor(args.predecessor_root, args.output_root)
    if args.apply:
        result = apply_successor(plan)
    else:
        result = {
            "predecessor_preflight_id": plan["predecessor_id"],
            "successor_preflight_id": plan["successor_id"],
            "run_id_retained_for_identical_seeds": plan["run_id"],
            "contract": plan["contract"],
            "output_root": str(plan["output_root"]),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
