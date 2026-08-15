"""Independently seal compact full-W4A8 layer retention before pruning.

The consolidated layer shard is the runtime artifact.  Per-expert shards are
resumable construction intermediates.  This module preserves every expert
manifest in a deterministic compressed archive, independently validates the
consolidated layer, and inventories every removable byte.  Pruning is a
separate, explicit operation bound to the receipt hash.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from safetensors import safe_open
from bmmlaw_r7_encoder.safetensors_io import SafeTensorReader

from .fresh_pipeline_artifacts import (
    expert_stem,
    validate_expert_artifact,
    validate_layer_artifact,
)
from .fresh_pipeline_common import atomic_json, canonical_sha256, sha256_file
from .fresh_pipeline_common import (
    LAYER_ARTIFACT_SCHEMA,
    PROJECTIONS,
    SQG_MARKER,
    assert_no_forbidden_tensor_names,
    canonical_json_bytes,
    load_json_object,
    permutation_scope,
    tensor_prefix,
)
from .glm52_fresh_sqg import validate_tensor_manifest
from .glm52_fresh_sqg.reference import tensor_sha256
from .glm52_atoms_v2 import (
    _rate_map,
    _topology_contract,
    materialize_glm52_atoms_v2_profile,
    profile_filename,
    validate_glm52_atoms_v2_profile,
)


RETENTION_SCHEMA = "glm52-sqg-full-w4a8-layer-retention-v1"
ARCHIVE_SCHEMA = "glm52-sqg-full-w4a8-expert-evidence-archive-v1"


def _gzip_bytes(value: object) -> bytes:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    output = bytearray()

    class _Sink:
        def write(self, data: bytes) -> int:
            output.extend(data)
            return len(data)
        def flush(self) -> None:
            return None

    with gzip.GzipFile(filename="", mode="wb", fileobj=_Sink(), mtime=0) as handle:
        handle.write(payload)
    return bytes(output)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists():
        raise FileExistsError(f"retention output exists: {path}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _expert_paths(layer_root: Path, layer: int, expert: int) -> tuple[Path, Path]:
    stem = expert_stem(layer, expert)
    canonical = layer_root / "experts" / f"{stem}.json"
    matches = list((layer_root / "expert_shards").glob(f"experts_*_*/{stem}.json"))
    if len(matches) != 1:
        raise ValueError(f"expert {expert}: construction mini-shard domain differs")
    return canonical, matches[0]


def seal_layer_retention(layer_manifest: str | Path) -> dict[str, Any]:
    manifest_path = Path(layer_manifest).resolve()
    layer_artifact = validate_layer_artifact(manifest_path)
    layer = int(layer_artifact["layer"])
    layer_root = manifest_path.parent.parent
    profile_path = manifest_path.parent / profile_filename(layer)
    if not profile_path.exists():
        materialize_glm52_atoms_v2_profile(manifest_path, profile_path)
    profile = validate_glm52_atoms_v2_profile(profile_path)
    archive_manifests: dict[str, Any] = {}
    prunable: list[dict[str, Any]] = []
    inode_sizes: dict[tuple[int, int], int] = {}
    inode_path_counts: dict[tuple[int, int], int] = {}
    for expert in range(256):
        canonical, construction = _expert_paths(layer_root, layer, expert)
        canonical_value = validate_expert_artifact(canonical)
        construction_value = validate_expert_artifact(construction)
        expected = layer_artifact["expert_manifest_sha256"][str(expert)]
        if canonical_value != construction_value or sha256_file(canonical) != expected:
            raise ValueError(f"expert {expert}: consolidated evidence binding differs")
        archive_manifests[str(expert)] = canonical_value
        for manifest in (canonical, construction):
            for path in (
                manifest,
                manifest.with_suffix(".json.sha256"),
                manifest.with_suffix(".safetensors"),
            ):
                stat = path.stat()
                key = (stat.st_dev, stat.st_ino)
                inode_sizes[key] = stat.st_size
                inode_path_counts[key] = inode_path_counts.get(key, 0) + 1
                prunable.append(
                    {
                        "path": str(path.relative_to(layer_root)),
                        "bytes": stat.st_size,
                        "sha256": sha256_file(path),
                        "device": stat.st_dev,
                        "inode": stat.st_ino,
                        "link_count": stat.st_nlink,
                    }
                )
    archive = {
        "schema": ARCHIVE_SCHEMA,
        "complete": True,
        "layer": layer,
        "run_id": layer_artifact["run_id"],
        "layer_manifest_sha256": sha256_file(manifest_path),
        "layer_shard_sha256": layer_artifact["shard_sha256"],
        "atoms_v2_profile": {
            "path": str(profile_path.relative_to(layer_root)),
            "sha256": sha256_file(profile_path),
            "manifest_id": profile["manifest_id"],
        },
        "expert_manifests": archive_manifests,
    }
    archive["archive_id"] = canonical_sha256(archive)
    retention_root = layer_root / "retention"
    archive_path = retention_root / "expert-evidence.json.gz"
    _atomic_bytes(archive_path, _gzip_bytes(archive))
    logical_bytes = sum(row["bytes"] for row in prunable)
    unique_bytes = sum(inode_sizes.values())
    reclaimable = sum(
        size
        for key, size in inode_sizes.items()
        if next(
            row["link_count"]
            for row in prunable
            if (row["device"], row["inode"]) == key
        )
        == inode_path_counts[key]
    )
    receipt = {
        "schema": RETENTION_SCHEMA,
        "complete": True,
        "pruned": False,
        "operator_action_required": True,
        "layer": layer,
        "run_id": layer_artifact["run_id"],
        "layer_manifest": str(manifest_path.relative_to(layer_root)),
        "layer_manifest_sha256": sha256_file(manifest_path),
        "layer_shard_sha256": layer_artifact["shard_sha256"],
        "atoms_v2_profile": archive["atoms_v2_profile"],
        "archive": {
            "path": str(archive_path.relative_to(layer_root)),
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
            "archive_id": archive["archive_id"],
            "expert_count": 256,
        },
        "prunable_files": sorted(prunable, key=lambda row: row["path"]),
        "budget": {
            "logical_bytes": logical_bytes,
            "unique_inode_bytes": unique_bytes,
            "reclaimable_bytes_at_seal": reclaimable,
        },
    }
    receipt["receipt_id"] = canonical_sha256(receipt)
    receipt_path = retention_root / "retention-receipt.json"
    atomic_json(receipt_path, receipt)
    digest = sha256_file(receipt_path)
    _atomic_bytes(
        retention_root / "retention-receipt.json.sha256",
        f"{digest}  retention-receipt.json\n".encode(),
    )
    return {"receipt": str(receipt_path), "receipt_sha256": digest, **receipt["budget"]}


def prune_sealed_layer(layer_root: str | Path, *, expected_receipt_sha256: str) -> dict[str, Any]:
    root = Path(layer_root).resolve()
    receipt_path = root / "retention" / "retention-receipt.json"
    if sha256_file(receipt_path) != expected_receipt_sha256:
        raise ValueError("retention receipt hash differs")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != RETENTION_SCHEMA or receipt.get("pruned") is not False:
        raise ValueError("retention receipt is not an unpruned seal")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if canonical_sha256(body) != receipt.get("receipt_id"):
        raise ValueError("retention receipt identity differs")
    retained_manifest = root / receipt["layer_manifest"]
    validate_layer_artifact(retained_manifest)
    archive_path = root / receipt["archive"]["path"]
    if sha256_file(archive_path) != receipt["archive"]["sha256"]:
        raise ValueError("retained expert-evidence archive differs")
    paths: list[Path] = []
    for row in receipt["prunable_files"]:
        path = root / row["path"]
        if path.resolve().parent == root or root not in path.resolve().parents:
            raise ValueError("retention prune path escapes layer root")
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"prunable file drifted: {path}")
        paths.append(path)
    for path in paths:
        path.unlink()
    for directory in sorted(
        {path.parent for path in paths},
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    result = {
        "schema": "glm52-sqg-full-w4a8-layer-retention-pruned-v1",
        "complete": True,
        "receipt_sha256": expected_receipt_sha256,
        "removed_files": len(paths),
        "logical_bytes_removed": sum(row["bytes"] for row in receipt["prunable_files"]),
        "consolidated_layer_preserved": True,
        "expert_evidence_archive_preserved": True,
    }
    result["result_id"] = canonical_sha256(result)
    atomic_json(root / "retention" / "pruned.json", result)
    # A prune is not reported complete until the compact path itself validates
    # the retained archive against every consolidated payload.
    validate_compact_layer_artifact(retained_manifest)
    return result


def validate_compact_layer_artifact(layer_manifest: str | Path) -> dict[str, Any]:
    """Validate a post-prune layer solely from sealed evidence and final bytes."""

    manifest_path = Path(layer_manifest).resolve()
    root = manifest_path.parent.parent
    retention = root / "retention"
    receipt_path = retention / "retention-receipt.json"
    seal_path = retention / "retention-receipt.json.sha256"
    pruned_path = retention / "pruned.json"
    if not all(path.is_file() and not path.is_symlink() for path in (receipt_path, seal_path, pruned_path)):
        raise ValueError("compact retention closure is absent")
    receipt_sha = sha256_file(receipt_path)
    if seal_path.read_text(encoding="ascii") != f"{receipt_sha}  retention-receipt.json\n":
        raise ValueError("compact retention receipt seal differs")
    receipt = load_json_object(receipt_path)
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if (
        receipt.get("schema") != RETENTION_SCHEMA
        or canonical_sha256(receipt_body) != receipt.get("receipt_id")
        or (root / receipt.get("layer_manifest", "")).resolve() != manifest_path
    ):
        raise ValueError("compact retention receipt identity differs")
    pruned = load_json_object(pruned_path)
    pruned_body = {key: value for key, value in pruned.items() if key != "result_id"}
    if (
        pruned.get("receipt_sha256") != receipt_sha
        or pruned.get("removed_files") != 1536
        or pruned.get("consolidated_layer_preserved") is not True
        or pruned.get("expert_evidence_archive_preserved") is not True
        or canonical_sha256(pruned_body) != pruned.get("result_id")
    ):
        raise ValueError("compact prune marker differs")
    if len(receipt.get("prunable_files", [])) != 1536:
        raise ValueError("compact prunable-file census differs")
    residual_experts = list((root / "experts").glob("layer-*-expert-*")) + list(
        (root / "expert_shards").glob("experts_*_*/layer-*-expert-*")
    )
    if residual_experts:
        raise ValueError("compact layer retains an ambiguous expert mini-shard set")

    archive_path = root / receipt["archive"]["path"]
    if archive_path != retention / "expert-evidence.json.gz" or sha256_file(archive_path) != receipt["archive"]["sha256"]:
        raise ValueError("compact expert archive differs")
    try:
        archive = json.loads(gzip.decompress(archive_path.read_bytes()))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("compact expert archive cannot be decoded") from exc
    archive_body = {key: value for key, value in archive.items() if key != "archive_id"}
    if (
        archive.get("schema") != ARCHIVE_SCHEMA
        or canonical_sha256(archive_body) != archive.get("archive_id")
        or archive.get("archive_id") != receipt["archive"]["archive_id"]
        or archive.get("layer_manifest_sha256") != sha256_file(manifest_path)
        or archive.get("atoms_v2_profile") != receipt.get("atoms_v2_profile")
    ):
        raise ValueError("compact expert archive identity differs")

    manifest = load_json_object(manifest_path)
    layer = int(manifest.get("layer", -1))
    if (
        manifest.get("schema") != LAYER_ARTIFACT_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("shard_sha256") != receipt["layer_shard_sha256"]
        or manifest.get("run_id") != archive.get("run_id")
        or layer != receipt.get("layer")
    ):
        raise ValueError("compact layer manifest binding differs")
    shard = manifest_path.parent / f"fresh-sqg-layer-{layer:03d}.safetensors"
    if not shard.is_file() or shard.is_symlink() or sha256_file(shard) != manifest["shard_sha256"]:
        raise ValueError("compact consolidated shard differs")
    reader = SafeTensorReader(shard)
    expected_prefixes = {
        tensor_prefix(layer, expert, projection)
        for expert in range(256)
        for projection in PROJECTIONS
    }
    shared_gate = f"model.layers.{layer}.mlp.experts.r7_shared.gate_up_suh"
    shared_down = f"model.layers.{layer}.mlp.experts.r7_shared.down_svh"
    expected_names = {shared_gate, shared_down}
    for prefix in expected_prefixes:
        projection = prefix.rsplit(".", 1)[-1]
        expected_names.update(
            {f"{prefix}.trellis", f"{prefix}.sqg", f"{prefix}.svh" if projection != "down_proj" else f"{prefix}.suh"}
        )
    if set(reader.tensors) != expected_names or set(manifest.get("payload_sha256", {})) != expected_names:
        raise ValueError("compact consolidated tensor inventory differs")
    assert_no_forbidden_tensor_names(expected_names)
    for name, info in reader.tensors.items():
        if info.payload.sha256() != manifest["payload_sha256"][name]:
            raise ValueError(f"compact consolidated payload differs: {name}")
    bit_map = manifest.get("bit_map", {})
    values = tuple(bit_map.values())
    if set(bit_map) != expected_prefixes or (values.count(3), values.count(4), sum(values)) != (384, 384, 2688):
        raise ValueError("compact layer bit budget differs")
    vector_refs = manifest.get("vector_refs", {})
    if set(vector_refs) != expected_prefixes:
        raise ValueError("compact layer vector-reference domain differs")
    for prefix in expected_prefixes:
        projection = prefix.rsplit(".", 1)[-1]
        expected_refs = (
            {"suh": shared_gate, "svh": f"{prefix}.svh"}
            if projection != "down_proj"
            else {"suh": f"{prefix}.suh", "svh": shared_down}
        )
        if vector_refs[prefix] != expected_refs:
            raise ValueError(f"compact vector reference differs: {prefix}")
    run = manifest.get("fresh_sqg_run_manifest", {})
    run_items = run.get("tensor_manifests", [])
    run_by_id = {str(item.get("tensor_id")): item for item in run_items}
    permutations = run.get("fresh_physical_permutations", [])
    if (
        set(run_by_id) != expected_prefixes
        or run.get("run_id") != f"{manifest.get('run_id')}/layer-{layer:03d}"
        or run.get("production") is not True
        or run.get("codec", {}).get("marker", {}).get("int32") != SQG_MARKER
        or {item.get("scope") for item in permutations}
        != {permutation_scope(layer, expert) for expert in range(256)}
        or len(permutations) != 256
        or any(item.get("production_qualified") is not True for item in permutations)
        or run.get("forbidden_reads", {}).get("observed") != []
        or run.get("codec", {}).get("fallback_allowed") is not False
    ):
        raise ValueError("compact nested run closure differs")
    for prefix, item in run_by_id.items():
        validate_tensor_manifest(item)
        if item.get("bits") != bit_map[prefix]:
            raise ValueError(f"compact nested rate differs: {prefix}")
    archived_experts = archive.get("expert_manifests", {})
    expert_hashes = manifest.get("expert_manifest_sha256", {})
    if set(archived_experts) != set(expert_hashes) or len(archived_experts) != 256:
        raise ValueError("compact archived expert domain differs")
    shared_profiles = manifest.get("shared_profiles", {})
    if set(shared_profiles) != {"gate_up_input", "down_output"}:
        raise ValueError("compact shared-profile evidence differs")
    permutation_by_scope = {item["scope"]: item for item in permutations}
    with safe_open(shard, framework="pt", device="cpu") as handle:
        shared_hashes = {"gate": tensor_sha256(handle.get_tensor(shared_gate)), "down": tensor_sha256(handle.get_tensor(shared_down))}
        if (
            shared_hashes["gate"]
            != shared_profiles["gate_up_input"].get("expected_stored_fp16_sha256")
            or shared_hashes["down"]
            != shared_profiles["down_output"].get("expected_stored_fp16_sha256")
        ):
            raise ValueError("compact shared-profile vectors differ")
        for expert in range(256):
            expert_manifest = archived_experts[str(expert)]
            serialized = canonical_json_bytes(expert_manifest) + b"\n"
            if hashlib.sha256(serialized).hexdigest() != expert_hashes[str(expert)]:
                raise ValueError(f"compact expert manifest hash differs: {expert}")
            if (
                expert_manifest.get("purpose") != "final_treatment"
                or expert_manifest.get("run_id") != manifest.get("run_id")
                or expert_manifest.get("shared_profiles") != shared_profiles
                or expert_manifest.get("permutation") != permutation_by_scope[permutation_scope(layer, expert)]
                or any(expert_manifest.get("lineage", {}).get(key) != 0 for key in (
                    "mcg_payload_reads", "mcg_transform_reads", "mcg_scale_reads",
                    "mcg_permutation_reads", "mcg_seed_reads", "mcg_decoded_weight_reads",
                ))
            ):
                raise ValueError(f"compact expert lineage differs: {expert}")
            for projection in PROJECTIONS:
                prefix = tensor_prefix(layer, expert, projection)
                nested = run_by_id[prefix]
                if expert_manifest["tensor_manifests"][prefix] != nested:
                    raise ValueError(f"compact layer/expert tensor lineage differs: {prefix}")
                private = "svh" if projection != "down_proj" else "suh"
                shared = "suh" if projection != "down_proj" else "svh"
                shared_name = shared_gate if projection != "down_proj" else shared_down
                if (
                    reader.tensors[f"{prefix}.trellis"].payload.sha256() != nested["packed_trellis_payload_sha256"]
                    or tensor_sha256(handle.get_tensor(f"{prefix}.{private}")) != nested["scales"][f"{private}_sha256"]
                    or shared_hashes["gate" if projection != "down_proj" else "down"] != nested["scales"][f"{shared}_sha256"]
                    or manifest["payload_sha256"][shared_name] != expert_manifest["payload_sha256"][f"{prefix}.{shared}"]
                    or manifest["payload_sha256"][f"{prefix}.trellis"] != expert_manifest["payload_sha256"][f"{prefix}.trellis"]
                    or manifest["payload_sha256"][f"{prefix}.{private}"] != expert_manifest["payload_sha256"][f"{prefix}.{private}"]
                    or manifest["payload_sha256"][f"{prefix}.sqg"] != expert_manifest["payload_sha256"][f"{prefix}.sqg"]
                ):
                    raise ValueError(f"compact layer/expert payload binding differs: {prefix}")
                if int(handle.get_tensor(f"{prefix}.sqg")) != SQG_MARKER:
                    raise ValueError(f"compact SQG marker differs: {prefix}")
    if manifest.get("lineage", {}).get("mcg_tensor_count") != 0 or manifest.get("lineage", {}).get("sqg_tensor_count") != 768:
        raise ValueError("compact layer codebook census differs")
    return manifest


def validate_compact_atoms_v2_profile(
    path: str | Path, artifact: dict[str, Any]
) -> dict[str, Any]:
    """Validate the retained atoms-v2 profile without requiring mini-shards."""

    profile_path = Path(path).resolve()
    layer = int(artifact["layer"])
    root = profile_path.parent.parent
    receipt = load_json_object(root / "retention" / "retention-receipt.json")
    recorded = receipt.get("atoms_v2_profile", {})
    if (
        profile_path != (root / recorded.get("path", "")).resolve()
        or profile_path.name != profile_filename(layer)
        or sha256_file(profile_path) != recorded.get("sha256")
    ):
        raise ValueError("compact atoms-v2 profile receipt binding differs")
    profile = load_json_object(profile_path)
    body = {key: value for key, value in profile.items() if key != "manifest_id"}
    source = profile.get("source", {})
    shard = profile_path.parent / artifact["shard"]
    if (
        canonical_sha256(body) != profile.get("manifest_id")
        or profile.get("manifest_id") != recorded.get("manifest_id")
        or profile.get("layer") != layer
        or source.get("layer_manifest_sha256")
        != sha256_file(profile_path.parent / f"fresh-sqg-layer-{layer:03d}.json")
        or source.get("layer_shard_sha256") != artifact["shard_sha256"]
        or source.get("layer_shard_bytes") != shard.stat().st_size
        or profile.get("rate_map") != _rate_map(layer, artifact)
        or profile.get("topology") != _topology_contract(layer)
        or profile.get("shared_profiles") != artifact["shared_profiles"]
        or profile.get("lineage", {}).get("run_id") != artifact["run_id"]
        or profile.get("lineage", {}).get("mcg_tensor_count") != 0
        or profile.get("lineage", {}).get("sqg_tensor_count") != 768
    ):
        raise ValueError("compact atoms-v2 profile closure differs")
    return profile


__all__ = [
    "seal_layer_retention",
    "prune_sealed_layer",
    "validate_compact_layer_artifact",
    "validate_compact_atoms_v2_profile",
]
