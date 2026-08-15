"""Construction-only rolling GLM SQG W4A8 checkpoint support.

This module deliberately does not create a release candidate.  It hardlinks
unchanged MCG bytes from the sealed r33 source, substitutes an explicit
contiguous prefix of fully encoded native-W4A8 layer artifacts, and emits a
contract that is rejected by the runtime unless construction mode is enabled.
The checkpoint exists only so the exact upstream W4A8 path can generate the
fit rows used to calibrate the next block.
"""

from __future__ import annotations

from copy import deepcopy
import errno
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from bmmlaw_r7_encoder.safetensors_io import SafeTensorReader

from .calibration_capture import (
    TEACHER_IDENTITY_RECEIPT_SHA256,
    TEACHER_IDENTITY_SEAL_SHA256,
)
from .fresh_candidate_materializer import (
    GLOBAL_CODEBOOK,
    REWRITTEN_IDENTITY_FILES,
    SQG_CODEBOOK,
    _expected_layer_names,
    _file_record,
    _fsync_directory,
    _hardlink,
    _atomic_copy,
    _load_json,
    _paths_overlap,
    _regular,
    _teacher_receipt_files,
    _validate_selected_payload_map,
)
from .fresh_pipeline_artifacts import validate_layer_artifact
from .fresh_pipeline_common import atomic_json, canonical_sha256, sha256_file
from .glm52_atoms_v2 import (
    materialize_glm52_atoms_v2_profile,
    profile_filename,
    validate_glm52_atoms_v2_profile,
)
from .glm52_fresh_sqg.manifest import W4A8_DERIVED_SOURCE_KIND
from .teacher_identity import IDENTITY_FILES, INDEX_FILE
from .w4a8_retention import (
    validate_compact_atoms_v2_profile,
    validate_compact_layer_artifact,
)


MANIFEST_SCHEMA = "glm52-sqg-progressive-w4a8-construction-v1"
RUNTIME_SCHEMA = "glm52_sqg_atoms_v2_w4a8_partial_construction_v1"
MANIFEST_NAME = "PROGRESSIVE_W4A8_CONSTRUCTION.json"
VERIFIED_NAME = ".progressive_w4a8_verified"
ROUTED_LAYERS = tuple(range(3, 78))


def _publish_exact(
    source: Path, destination: Path, *, expected_sha256: str
) -> str:
    """Publish exact bytes, preferring identity-preserving hardlinks.

    The teacher and rolling output should normally share a filesystem so the
    very large unchanged complement is hardlinked.  Native layer artifacts may
    live on the build NVMe; EXDEV is therefore an expected, explicitly handled
    case.  A cross-device publication is an atomic copy whose streamed digest
    must equal the sealed source digest before it is accepted.
    """

    try:
        _hardlink(source, destination)
        return "hardlink"
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    copied_bytes, copied_sha256 = _atomic_copy(source, destination)
    if copied_sha256 != expected_sha256 or copied_bytes != source.stat().st_size:
        destination.unlink(missing_ok=True)
        raise ValueError(f"cross-device copied bytes differ: {source}")
    return "verified_copy"


def _require_hex64(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} is not a canonical SHA256")
    return value


def _layer_source_shard(index: Mapping[str, Any], layer: int) -> str:
    prefix = f"model.layers.{layer}.mlp.experts."
    names = {value for key, value in index["weight_map"].items() if key.startswith(prefix)}
    expected = f"r7-experts-layer-{layer:03d}.safetensors"
    if names != {expected}:
        raise ValueError(f"source layer {layer} shard topology differs: {names}")
    return expected


def _derived_down_target(layer: int, artifact: Mapping[str, Any]) -> dict[str, Any]:
    manifests = artifact.get("fresh_sqg_run_manifest", {}).get("tensor_manifests")
    if not isinstance(manifests, list) or len(manifests) != 768:
        raise ValueError(f"layer {layer}: tensor-manifest census differs")
    rows: list[dict[str, Any]] = []
    betas: set[float] = set()
    for item in manifests:
        if not isinstance(item, Mapping) or item.get("matrix_role") != "down":
            continue
        source = item.get("source")
        if not isinstance(source, Mapping) or source.get("kind") != W4A8_DERIVED_SOURCE_KIND:
            raise ValueError(
                f"layer {layer}: down tensor is not an exact fit-only W4A8-derived target"
            )
        if source.get("mcg_source") is not False:
            raise ValueError(f"layer {layer}: derived down target has MCG lineage")
        beta = source.get("beta")
        if (
            isinstance(beta, bool)
            or not isinstance(beta, int | float)
            or not math.isfinite(float(beta))
            or not 0.0 <= float(beta) <= 1.0
        ):
            raise ValueError(f"layer {layer}: derived down beta differs")
        evidence = source.get("fit_hb_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"layer {layer}: fit H/B evidence is absent")
        row = {
            "tensor_id": item.get("tensor_id"),
            "derived_target_tensor_sha256": _require_hex64(
                source.get("derived_target_tensor_sha256"), "derived target"
            ),
            "execution_contract_sha256": _require_hex64(
                source.get("execution_contract_sha256"), "execution contract"
            ),
            "fit_hb_evidence_id": _require_hex64(
                evidence.get("evidence_id"), "fit H/B evidence ID"
            ),
            "fit_hessian_sha256": _require_hex64(
                evidence.get("hessian_sha256"), "fit H"
            ),
            "fit_cross_term_sha256": _require_hex64(
                evidence.get("cross_term_sha256"), "fit B"
            ),
        }
        if not isinstance(row["tensor_id"], str):
            raise ValueError(f"layer {layer}: derived down tensor ID is absent")
        rows.append(row)
        betas.add(float(beta))
    if len(rows) != 256 or len(betas) != 1:
        raise ValueError(
            f"layer {layer}: expected 256 down targets and one beta, got "
            f"{len(rows)} / {sorted(betas)}"
        )
    rows.sort(key=lambda value: value["tensor_id"])
    return {
        "derived_down_target_id": canonical_sha256(rows),
        "down_target_beta": betas.pop(),
        "expert_target_count": len(rows),
        "expert_targets_sha256": canonical_sha256(rows),
    }


def _validate_layer_closure(manifest_path: Path) -> dict[str, Any]:
    try:
        return validate_layer_artifact(manifest_path)
    except ValueError:
        # Compact closure is admissible only after the separately sealed prune
        # marker exists; its validator replays the expert lineage against the
        # consolidated payload.  This does not relax ordinary artifact errors.
        if not (manifest_path.parent.parent / "retention" / "pruned.json").is_file():
            raise
        return validate_compact_layer_artifact(manifest_path)


def _load_w4a8_layer(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    artifact = _validate_layer_closure(manifest_path)
    layer = int(artifact["layer"])
    if manifest_path.name != f"fresh-sqg-layer-{layer:03d}.json":
        raise ValueError(f"layer {layer}: noncanonical artifact filename")
    evidence = artifact.get("evidence", {})
    lineage = artifact.get("lineage", {})
    if (
        evidence.get("activation_endpoint") != "full-w4a8"
        or evidence.get("w4a8_derived_down_lineage") is not True
        or evidence.get("mcg_inputs") != 0
        or lineage.get("sqg_tensor_count") != 768
        or lineage.get("mcg_tensor_count") != 0
    ):
        raise ValueError(f"layer {layer}: artifact is not native full-W4A8")
    shard = manifest_path.parent / str(artifact["shard"])
    if sha256_file(shard) != artifact["shard_sha256"]:
        raise ValueError(f"layer {layer}: artifact shard hash differs")
    census = _validate_selected_payload_map(shard, layer, artifact["payload_sha256"])
    profile_path = manifest_path.parent / profile_filename(layer)
    if not profile_path.exists():
        if (manifest_path.parent.parent / "retention" / "pruned.json").is_file():
            raise ValueError(f"layer {layer}: compact closure lacks its sealed atoms-v2 profile")
        materialize_glm52_atoms_v2_profile(manifest_path, profile_path)
    try:
        profile = validate_glm52_atoms_v2_profile(profile_path)
    except ValueError:
        if not (manifest_path.parent.parent / "retention" / "pruned.json").is_file():
            raise
        profile = validate_compact_atoms_v2_profile(profile_path, artifact)
    target = _derived_down_target(layer, artifact)
    return {
        "layer": layer,
        "manifest_path": manifest_path,
        "manifest": artifact,
        "manifest_sha256": sha256_file(manifest_path),
        "shard_path": shard,
        "profile_path": profile_path,
        "profile_sha256": sha256_file(profile_path),
        "profile_id": profile["manifest_id"],
        "census": census,
        "down_target": target,
    }


def _runtime_contract(layers: Sequence[dict[str, Any]]) -> dict[str, Any]:
    encoded = [int(item["layer"]) for item in layers]
    return {
        "schema": RUNTIME_SCHEMA,
        "execution": "progressive_fixed_point_recapture",
        "purpose": "construction_only",
        "final_release_eligible": False,
        "acceptance_eligible": False,
        "codebook": SQG_CODEBOOK,
        "rates": "independent_per_tensor_k3_k4",
        "direct_e4m3_weights": True,
        "allow_a16_fallback": False,
        "activation": "silu_gate_times_up",
        "topology": "topology_neutral",
        "per_layer_bit_census": {"k3": 384, "k4": 384, "total": 768},
        "parallelism": {
            "tensor_parallel_size": 4,
            "decode_context_parallel_size": 4,
            "mtp": True,
            "long_context_max_tokens": 1_048_576,
        },
        "encoded_layers": encoded,
        "unchanged_mcg_layers": sorted(set(ROUTED_LAYERS) - set(encoded)),
        "down_targets": {
            str(item["layer"]): {
                "derived_down_target_id": item["down_target"]["derived_down_target_id"],
                "down_target_beta": item["down_target"]["down_target_beta"],
            }
            for item in layers
        },
    }


def _source_context(source: Path, receipt: Path) -> dict[str, Any]:
    if sha256_file(receipt) != TEACHER_IDENTITY_RECEIPT_SHA256:
        raise ValueError("teacher receipt serialized bytes differ")
    receipt_files = _teacher_receipt_files(receipt)
    if _load_json(receipt).get("seal_sha256") != TEACHER_IDENTITY_SEAL_SHA256:
        raise ValueError("teacher receipt seal differs")
    for name, record in receipt_files.items():
        path = _regular(source / name, label=f"source {name}")
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"source {name} size differs from sealed receipt")
    index = _load_json(source / INDEX_FILE)
    quant = _load_json(source / "quantization_config.json")
    config = _load_json(source / "config.json")
    if config.get("quantization_config") != quant:
        raise ValueError("source embedded/external quantization configs differ")
    if quant.get("r7_routed_experts", {}).get("codebook") != GLOBAL_CODEBOOK:
        raise ValueError("progressive source must be the sealed all-MCG checkpoint")
    payloads = {name for name, record in receipt_files.items() if record["role"] == "indexed_payload"}
    sidecars = {name for name, record in receipt_files.items() if record["role"] == "r7_loader_sidecar"}
    if set(index.get("weight_map", {}).values()) != payloads:
        raise ValueError("source index payload domain differs")
    return {
        "files": receipt_files,
        "payloads": payloads,
        "sidecars": sidecars,
        "index": index,
        "quant": quant,
        "config": config,
    }


def _rewrite_index(source: Path, source_index: Mapping[str, Any], layers: Sequence[dict[str, Any]]) -> dict[str, Any]:
    value = deepcopy(dict(source_index))
    weight_map = value["weight_map"]
    metadata = value["metadata"]
    old_bytes = 0
    new_bytes = 0
    for item in layers:
        layer = int(item["layer"])
        old_name = _layer_source_shard(source_index, layer)
        old_keys = {key for key, shard in weight_map.items() if shard == old_name}
        new_keys = set(item["manifest"]["payload_sha256"])
        if len(old_keys) != 2_306 or new_keys != _expected_layer_names(layer):
            raise ValueError(f"layer {layer}: index replacement domain differs")
        old_reader = SafeTensorReader(source / old_name)
        new_reader = SafeTensorReader(item["shard_path"])
        if set(old_reader.tensors) != old_keys or set(new_reader.tensors) != new_keys:
            raise ValueError(f"layer {layer}: index/shard tensor domain differs")
        old_bytes += sum(info.nbytes for info in old_reader.tensors.values())
        new_bytes += sum(info.nbytes for info in new_reader.tensors.values())
        for key in old_keys:
            del weight_map[key]
        for key in new_keys:
            weight_map[key] = old_name
    total = metadata.get("total_size")
    if type(total) is not int or total <= old_bytes:
        raise ValueError("source index total_size differs")
    metadata["total_size"] = total - old_bytes + new_bytes
    value["weight_map"] = dict(sorted(weight_map.items()))
    return value


def _sidecar(item: Mapping[str, Any]) -> dict[str, Any]:
    layer = int(item["layer"])
    artifact = item["manifest"]
    value = {
        "schema": "glm52-sqg-progressive-w4a8-layer-sidecar-v1",
        "complete": True,
        "layer": layer,
        "codebook": SQG_CODEBOOK,
        "execution": "full_w4a8",
        "allow_a16_fallback": False,
        "shard": f"r7-experts-layer-{layer:03d}.safetensors",
        "shard_sha256": artifact["shard_sha256"],
        "bit_map": artifact["bit_map"],
        "bit_histogram": {"3": 384, "4": 384},
        "shared_vectors": artifact["shared_vectors"],
        "vector_refs": artifact["vector_refs"],
        "payload_sha256": artifact["payload_sha256"],
        "atoms_v2": {
            "profile": item["profile_path"].name,
            "profile_sha256": item["profile_sha256"],
            "profile_id": item["profile_id"],
        },
        "derived_down": item["down_target"],
        "lineage": {
            "layer_manifest": str(item["manifest_path"]),
            "layer_manifest_sha256": item["manifest_sha256"],
            "sqg_tensors": 768,
            "mcg_tensors": 0,
            "topology_neutral": True,
            "native_e4m3": True,
        },
    }
    value["sidecar_id"] = canonical_sha256(value)
    return value


def materialize_progressive_w4a8_candidate(
    *,
    source_model: str | Path,
    teacher_receipt: str | Path,
    layer_manifests: Sequence[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    """Build a rolling partial-W4A8 checkpoint without launching a model."""

    source = Path(source_model).resolve()
    receipt = Path(teacher_receipt).resolve()
    destination = Path(output).absolute()
    if not source.is_dir() or source.is_symlink():
        raise ValueError("source model must be a real directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    if _paths_overlap(destination, source):
        raise ValueError("rolling output must be disjoint from the source model")
    layers = sorted((_load_w4a8_layer(path) for path in layer_manifests), key=lambda item: item["layer"])
    encoded = [item["layer"] for item in layers]
    if not encoded or encoded != list(range(3, encoded[-1] + 1)):
        raise ValueError("encoded layers must be one contiguous prefix beginning at 3")
    if len(set(encoded)) != len(encoded):
        raise ValueError("encoded layer manifests contain duplicates")
    source_ctx = _source_context(source, receipt)
    runtime_contract = _runtime_contract(layers)
    old_shards = {_layer_source_shard(source_ctx["index"], layer) for layer in encoded}
    old_sidecars = {f"r7-experts-layer-{layer:03d}.json" for layer in encoded}

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    _fsync_directory(destination.parent)
    records: dict[str, Any] = {}
    try:
        for name in sorted(source_ctx["payloads"] - old_shards):
            record = source_ctx["files"][name]
            publication = _publish_exact(
                source / name,
                destination / name,
                expected_sha256=record["sha256"],
            )
            records[name] = _file_record(
                destination / name,
                role="unchanged_mcg_or_dense_payload",
                materialization=publication,
                origin=source / name,
                expected_sha256=record["sha256"],
            )
        layer_records: dict[str, Any] = {}
        for item in layers:
            layer = item["layer"]
            shard_name = f"r7-experts-layer-{layer:03d}.safetensors"
            publication = _publish_exact(
                item["shard_path"],
                destination / shard_name,
                expected_sha256=item["manifest"]["shard_sha256"],
            )
            records[shard_name] = _file_record(
                destination / shard_name,
                role="native_full_w4a8_sqg_payload",
                materialization=publication,
                origin=item["shard_path"],
                expected_sha256=item["manifest"]["shard_sha256"],
            )
            sidecar_name = f"r7-experts-layer-{layer:03d}.json"
            atomic_json(destination / sidecar_name, _sidecar(item))
            records[sidecar_name] = _file_record(
                destination / sidecar_name,
                role="native_full_w4a8_sqg_sidecar",
                materialization="generated",
                origin=item["manifest_path"],
            )
            layer_records[str(layer)] = {
                "layer_manifest": str(item["manifest_path"]),
                "layer_manifest_sha256": item["manifest_sha256"],
                "layer_shard_sha256": item["manifest"]["shard_sha256"],
                "atoms_v2_profile_sha256": item["profile_sha256"],
                "atoms_v2_profile_id": item["profile_id"],
                "census": item["census"],
                "down_target": item["down_target"],
            }
        for name in sorted(source_ctx["sidecars"] - old_sidecars):
            record = source_ctx["files"][name]
            publication = _publish_exact(
                source / name,
                destination / name,
                expected_sha256=record["sha256"],
            )
            records[name] = _file_record(
                destination / name,
                role="unchanged_mcg_sidecar",
                materialization=publication,
                origin=source / name,
                expected_sha256=record["sha256"],
            )
        for name in sorted(set(IDENTITY_FILES) - REWRITTEN_IDENTITY_FILES):
            record = source_ctx["files"][name]
            publication = _publish_exact(
                source / name,
                destination / name,
                expected_sha256=record["sha256"],
            )
            records[name] = _file_record(
                destination / name,
                role="unchanged_loader_identity",
                materialization=publication,
                origin=source / name,
                expected_sha256=record["sha256"],
            )

        quant = deepcopy(source_ctx["quant"])
        r7 = quant["r7_routed_experts"]
        r7["codebook_overrides"] = {str(layer): SQG_CODEBOOK for layer in encoded}
        r7["codebook_tensor_overrides"] = {}
        quant["glm_sqg_w4a8"] = runtime_contract
        config = deepcopy(source_ctx["config"])
        config["quantization_config"] = quant
        index = _rewrite_index(source, source_ctx["index"], layers)
        for name, value in (
            (INDEX_FILE, index),
            ("quantization_config.json", quant),
            ("config.json", config),
        ):
            atomic_json(destination / name, value)
            records[name] = _file_record(
                destination / name,
                role="rewritten_construction_loader_identity",
                materialization="generated",
                origin=source / name,
            )

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "complete": True,
            "purpose": "construction_only_progressive_fixed_point_recapture",
            "final_release_eligible": False,
            "acceptance_eligible": False,
            "model_workload_launched": False,
            "source": {
                "root": str(source),
                "teacher_receipt": str(receipt),
                "teacher_receipt_sha256": TEACHER_IDENTITY_RECEIPT_SHA256,
                "source_bytes_mutated": False,
            },
            "encoded_layers": encoded,
            "unchanged_mcg_layers": runtime_contract["unchanged_mcg_layers"],
            "activation_endpoint": "full-w4a8",
            "allow_a16_fallback": False,
            "runtime_contract": runtime_contract,
            "layers": layer_records,
            "files": dict(sorted(records.items())),
        }
        manifest["manifest_id"] = canonical_sha256(manifest)
        atomic_json(destination / MANIFEST_NAME, manifest)
        validate_progressive_w4a8_candidate(destination, source_model=source)
        marker = destination / VERIFIED_NAME
        marker.write_text(f"{sha256_file(destination / MANIFEST_NAME)}\n", encoding="ascii")
        _fsync_directory(destination)
        return validate_progressive_w4a8_candidate(
            destination, source_model=source, require_marker=True
        )
    except BaseException:
        # Forensic partial output is intentionally retained without a marker.
        raise


def validate_progressive_w4a8_candidate(
    candidate: str | Path,
    *,
    source_model: str | Path | None = None,
    require_marker: bool = False,
) -> dict[str, Any]:
    root = Path(candidate).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("progressive candidate must be a real directory")
    manifest_path = _regular(root / MANIFEST_NAME, label="progressive manifest")
    manifest = _load_json(manifest_path)
    body = dict(manifest)
    manifest_id = body.pop("manifest_id", None)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("complete") is not True
        or manifest_id != canonical_sha256(body)
        or manifest.get("purpose") != "construction_only_progressive_fixed_point_recapture"
        or manifest.get("final_release_eligible") is not False
        or manifest.get("acceptance_eligible") is not False
        or manifest.get("activation_endpoint") != "full-w4a8"
        or manifest.get("allow_a16_fallback") is not False
    ):
        raise ValueError("progressive construction manifest differs")
    encoded = manifest.get("encoded_layers")
    if not isinstance(encoded, list) or not encoded or encoded != list(range(3, encoded[-1] + 1)):
        raise ValueError("progressive encoded layer prefix differs")
    contract = manifest.get("runtime_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("schema") != RUNTIME_SCHEMA
        or contract.get("encoded_layers") != encoded
        or contract.get("unchanged_mcg_layers") != manifest.get("unchanged_mcg_layers")
        or contract.get("allow_a16_fallback") is not False
    ):
        raise ValueError("progressive runtime contract differs")
    quant = _load_json(root / "quantization_config.json")
    config = _load_json(root / "config.json")
    if quant.get("glm_sqg_w4a8") != contract or config.get("quantization_config") != quant:
        raise ValueError("progressive embedded/external runtime contract differs")
    r7 = quant.get("r7_routed_experts", {})
    if (
        r7.get("codebook") != GLOBAL_CODEBOOK
        or r7.get("codebook_overrides") != {str(layer): SQG_CODEBOOK for layer in encoded}
        or r7.get("codebook_tensor_overrides") != {}
    ):
        raise ValueError("progressive codebook override domain differs")
    records = manifest.get("files")
    if not isinstance(records, Mapping):
        raise ValueError("progressive file manifest is absent")
    expected = set(records) | {MANIFEST_NAME}
    if require_marker:
        expected.add(VERIFIED_NAME)
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("progressive root file allowlist differs")
    for name, record in records.items():
        if not isinstance(record, Mapping) or not isinstance(record.get("role"), str):
            raise ValueError(f"progressive file record differs: {name}")
        path = _regular(root / name, label=f"progressive file {name}")
        if path.stat().st_size != record.get("bytes"):
            raise ValueError(f"progressive file size differs: {name}")
        if record.get("role") in {
            "native_full_w4a8_sqg_payload",
            "native_full_w4a8_sqg_sidecar",
            "rewritten_construction_loader_identity",
        } and sha256_file(path) != record.get("sha256"):
            raise ValueError(f"progressive generated/substituted bytes differ: {name}")
    source = Path(manifest["source"]["root"])
    if source_model is not None and Path(source_model).resolve() != source:
        raise ValueError("progressive source-model binding differs")
    if not source.is_dir() or source.is_symlink():
        raise ValueError("progressive protected source model is absent/unsafe")
    for name, record in records.items():
        role = record.get("role")
        if role.startswith("unchanged_"):
            mode = record.get("materialization")
            if record.get("origin") != str(source / name) or mode not in {
                "hardlink",
                "verified_copy",
            }:
                raise ValueError(f"unchanged progressive publication differs: {name}")
            same_inode = os.path.samefile(root / name, source / name)
            if mode == "hardlink" and not same_inode:
                raise ValueError(f"unchanged progressive hardlink differs: {name}")
            if mode == "verified_copy" and same_inode:
                raise ValueError(f"unchanged progressive copy shares source inode: {name}")
            if mode == "verified_copy" and (
                sha256_file(root / name) != record.get("sha256")
                or sha256_file(source / name) != record.get("sha256")
            ):
                raise ValueError(f"unchanged progressive copied bytes differ: {name}")
        elif role == "native_full_w4a8_sqg_payload":
            origin = Path(str(record.get("origin")))
            mode = record.get("materialization")
            if not origin.is_file() or mode not in {"hardlink", "verified_copy"}:
                raise ValueError(f"W4A8 artifact publication differs: {name}")
            same_inode = os.path.samefile(root / name, origin)
            if mode == "hardlink" and not same_inode:
                raise ValueError(f"W4A8 artifact hardlink binding differs: {name}")
            if mode == "verified_copy" and (
                same_inode
                or sha256_file(root / name) != record.get("sha256")
                or sha256_file(origin) != record.get("sha256")
            ):
                raise ValueError(f"W4A8 artifact copied bytes differ: {name}")
            if os.path.samefile(root / name, source / name):
                raise ValueError(f"W4A8 layer reused its legacy MCG shard: {name}")

    source_index = _load_json(source / INDEX_FILE)
    candidate_index = _load_json(root / INDEX_FILE)
    encoded_set = set(encoded)
    source_outside = {
        key: shard
        for key, shard in source_index["weight_map"].items()
        if not any(
            key.startswith(f"model.layers.{layer}.mlp.experts.")
            for layer in encoded_set
        )
    }
    candidate_outside = {
        key: shard
        for key, shard in candidate_index["weight_map"].items()
        if not any(
            key.startswith(f"model.layers.{layer}.mlp.experts.")
            for layer in encoded_set
        )
    }
    if candidate_outside != source_outside:
        raise ValueError("progressive index changed a tensor outside encoded layers")
    layer_records = manifest.get("layers")
    if not isinstance(layer_records, Mapping) or set(layer_records) != {
        str(layer) for layer in encoded
    }:
        raise ValueError("progressive layer-record domain differs")
    for layer in encoded:
        recorded = layer_records[str(layer)]
        if not isinstance(recorded, Mapping):
            raise ValueError(f"progressive layer {layer} record differs")
        context = _load_w4a8_layer(recorded.get("layer_manifest"))
        expected_record = {
            "layer_manifest": str(context["manifest_path"]),
            "layer_manifest_sha256": context["manifest_sha256"],
            "layer_shard_sha256": context["manifest"]["shard_sha256"],
            "atoms_v2_profile_sha256": context["profile_sha256"],
            "atoms_v2_profile_id": context["profile_id"],
            "census": context["census"],
            "down_target": context["down_target"],
        }
        if recorded != expected_record:
            raise ValueError(f"progressive layer {layer} artifact binding differs")
        if _load_json(root / f"r7-experts-layer-{layer:03d}.json") != _sidecar(context):
            raise ValueError(f"progressive layer {layer} sidecar differs")
        expected_target = {
            "derived_down_target_id": context["down_target"]["derived_down_target_id"],
            "down_target_beta": context["down_target"]["down_target_beta"],
        }
        if contract["down_targets"].get(str(layer)) != expected_target:
            raise ValueError(f"progressive layer {layer} runtime down target differs")
        shard_name = f"r7-experts-layer-{layer:03d}.safetensors"
        mapped = {
            key
            for key, shard in candidate_index["weight_map"].items()
            if shard == shard_name
            and key.startswith(f"model.layers.{layer}.mlp.experts.")
        }
        if mapped != set(SafeTensorReader(root / shard_name).tensors):
            raise ValueError(f"progressive layer {layer} index/shard domain differs")
    if require_marker:
        expected_marker = f"{sha256_file(manifest_path)}\n"
        if (root / VERIFIED_NAME).read_text(encoding="ascii") != expected_marker:
            raise ValueError("progressive verification marker differs")
    return {
        "schema": "glm52-sqg-progressive-w4a8-validation-v1",
        "candidate": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_id": manifest_id,
        "encoded_layers": encoded,
        "unchanged_mcg_layers": manifest["unchanged_mcg_layers"],
        "activation_endpoint": "full-w4a8",
        "allow_a16_fallback": False,
        "construction_only": True,
        "verified_marker_present": require_marker,
    }


__all__ = [
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "RUNTIME_SCHEMA",
    "VERIFIED_NAME",
    "materialize_progressive_w4a8_candidate",
    "validate_progressive_w4a8_candidate",
]
