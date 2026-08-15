"""No-launch materialization and validation of the four-layer SQG candidate.

The builder constructs a new flat runtime checkpoint from an explicit
allowlist.  It never recursively clones the source model, never links a
selected legacy R7 shard or sidecar, and publishes its verification marker
only after every selected SQG payload is bound to the sealed layer artifacts
and four-layer run seal.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from bmmlaw_r7_encoder.safetensors_io import SafeTensorReader

from .calibration_capture import (
    TEACHER_IDENTITY_RECEIPT_SHA256,
    TEACHER_IDENTITY_SEAL_SHA256,
    validate_teacher_identity_evidence,
)
from .fresh_pipeline_artifacts import validate_layer_artifact
from .fresh_pipeline_common import (
    BIT_CONTRACT_SCHEMA,
    EXPECTED_BIT_UNITS_PER_LAYER,
    EXPECTED_K3_PER_LAYER,
    EXPECTED_K4_PER_LAYER,
    HOLDOUT_REPORT_SCHEMA,
    LAYER_ARTIFACT_SCHEMA,
    NUM_EXPERTS,
    PROFILE_SEARCH_SCHEMA,
    PROJECTIONS,
    RUN_SEAL_SCHEMA,
    SELECTED_LAYERS,
    SQG_MARKER,
    atomic_json,
    canonical_sha256,
    load_bit_contract,
    sha256_file,
    tensor_prefix,
)
from .fresh_pipeline_runner import load_preflight
from .teacher_identity import (
    IDENTITY_FILES,
    INDEX_FILE,
    validate_teacher_identity_receipt,
)


CANDIDATE_SCHEMA = "glm52-fresh-sqg-four-layer-candidate-v1"
CANDIDATE_SIDECAR_SCHEMA = "glm52-fresh-sqg-runtime-layer-sidecar-v1"
CANDIDATE_REPORT_SCHEMA = "glm52-pure-sqg-four-layer-preflight-v2"
SQG_CODEBOOK = "sqg_xor_cheb_t12"
GLOBAL_CODEBOOK = "mcg"
MANIFEST_NAME = "MANIFEST.json"
VERIFIED_NAME = ".manifest_verified"
RUN_SEAL_COPY_NAME = "FRESH_SQG_RUN_SEAL.json"
REWRITTEN_IDENTITY_FILES = {
    INDEX_FILE,
    "config.json",
    "quantization_config.json",
}
EXPECTED_OVERRIDES = {
    str(layer): SQG_CODEBOOK for layer in SELECTED_LAYERS
}
_BASE_RUN_SCOPE = {
    "four_layer_replacement_artifacts": True,
    "runnable_model_materialized": False,
    "production_container_touched": False,
    "existing_model_mutated": False,
}
_TIME_PRIORITY_HOLDOUT_SCOPE = {
    "status": "skipped",
    "reason": "operator_time_priority",
    "layers": list(SELECTED_LAYERS),
    "report_count": 0,
    "routed_score_count": 0,
}
_TIME_PRIORITY_SELECTION_SCOPE = {
    "completed_draw_prefix": 4,
    "selected_cell_count": 16,
    "preregistered_draws": 8,
    "preregistered_cell_count": 32,
    "operator_time_priority": True,
    "unscored_cells_excluded": 16,
    "all_selected_cells_exact_sqg": True,
    "selection_math_uses_original_31_comparison_bonferroni": True,
}


def _time_priority_holdout_record(
    selection_sha256: str,
    layer_artifact_sha256: str,
) -> dict[str, object]:
    return {
        "status": "skipped",
        "reason": "operator_time_priority",
        "report_present": False,
        "routed_score_present": False,
        "used_for_calibration": False,
        "used_for_profile_choice": False,
        "selection_sha256": selection_sha256,
        "layer_artifact_sha256": layer_artifact_sha256,
    }


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load exact JSON object {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source}: expected a JSON object")
    return value


def _hex64(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not canonical SHA256")
    return value


def _regular(path: Path, *, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is absent or not a real regular file: {path}")
    return path


def _real_directory(path: str | Path, *, label: str) -> Path:
    raw = Path(path)
    if not raw.is_dir() or raw.is_symlink():
        raise ValueError(f"{label} must be a real directory: {raw}")
    return raw.resolve()


def _exact_child_directory(parent: Path, name: str, *, label: str) -> Path:
    """Return one literal, non-symlink child directory of a canonical parent."""

    if Path(name).name != name:
        raise ValueError(f"{label} has an unsafe name: {name!r}")
    child = parent / name
    if not child.is_dir() or child.is_symlink() or child.resolve() != child:
        raise ValueError(f"{label} must be a real contained directory: {child}")
    return child


def _exact_child_file(parent: Path, name: str, *, label: str) -> Path:
    """Return one literal, non-symlink regular file beneath a checked directory."""

    if Path(name).name != name:
        raise ValueError(f"{label} has an unsafe name: {name!r}")
    child = parent / name
    _regular(child, label=label)
    if child.resolve() != child:
        raise ValueError(f"{label} escapes its declared directory: {child}")
    return child


def _validate_canonical_id(
    value: Mapping[str, Any], field: str, *, label: str
) -> str:
    expected = _hex64(value.get(field), label=f"{label} {field}")
    body = {key: item for key, item in value.items() if key != field}
    if canonical_sha256(body) != expected:
        raise ValueError(f"{label} canonical {field} binding differs")
    return expected


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _assert_output_disjoint(output: Path, protected: Mapping[str, Path]) -> None:
    for label, path in protected.items():
        if _paths_overlap(output, path):
            raise ValueError(
                f"candidate output overlaps protected {label}: {output} vs {path}"
            )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_copy(source: Path, destination: Path) -> tuple[int, str]:
    """Copy bytes into a new inode while hashing the exact published stream."""

    _regular(source, label="copy source")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace candidate file: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    total = 0
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            while payload := reader.read(64 << 20):
                writer.write(payload)
                digest.update(payload)
                total += len(payload)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if os.path.samefile(source, destination):
        raise RuntimeError("selected artifact copy unexpectedly shares its source inode")
    return total, digest.hexdigest()


def _hardlink(source: Path, destination: Path) -> None:
    _regular(source, label="hardlink source")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace candidate file: {destination}")
    os.link(source, destination, follow_symlinks=False)
    _fsync_directory(destination.parent)
    if not os.path.samefile(source, destination):
        raise RuntimeError("unchanged runtime hardlink does not share source inode")


def _file_record(
    path: Path,
    *,
    role: str,
    materialization: str,
    origin: Path | str,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    _regular(path, label=f"candidate {role}")
    digest = sha256_file(path) if expected_sha256 is None else expected_sha256
    _hex64(digest, label=f"candidate {path.name} SHA256")
    return {
        "bytes": path.stat().st_size,
        "sha256": digest,
        "role": role,
        "materialization": materialization,
        "origin": str(origin),
    }


def _teacher_receipt_files(receipt_path: Path) -> dict[str, dict[str, Any]]:
    receipt = _load_json(receipt_path)
    if (
        receipt.get("schema") != "glm52-r33-teacher-identity-receipt-v1"
        or receipt.get("seal_sha256") != TEACHER_IDENTITY_SEAL_SHA256
        or not isinstance(receipt.get("seal"), dict)
        or not isinstance(receipt["seal"].get("files"), list)
    ):
        raise ValueError("teacher identity receipt differs from the frozen source")
    files: dict[str, dict[str, Any]] = {}
    for raw in receipt["seal"]["files"]:
        if not isinstance(raw, dict) or set(raw) != {
            "path",
            "role",
            "bytes",
            "sha256",
        }:
            raise ValueError("teacher identity receipt file record differs")
        name = raw.get("path")
        if not isinstance(name, str) or Path(name).name != name or name in files:
            raise ValueError("teacher receipt contains an unsafe/duplicate filename")
        _hex64(raw.get("sha256"), label=f"teacher {name} SHA256")
        files[name] = dict(raw)
    return files


def _source_inventory(
    source: Path,
    receipt_path: Path,
    *,
    verify_teacher: bool,
    teacher_validation: dict[str, object] | None = None,
) -> dict[str, Any]:
    if teacher_validation is None:
        if not verify_teacher:
            raise ValueError("prevalidated teacher evidence was not supplied")
        teacher_validation = validate_teacher_identity_receipt(
            source,
            receipt_path,
            expected_seal_sha256=TEACHER_IDENTITY_SEAL_SHA256,
            verify_mode="full",
            workers=4,
        )
    validation = validate_teacher_identity_evidence(teacher_validation)
    if sha256_file(receipt_path) != TEACHER_IDENTITY_RECEIPT_SHA256:
        raise ValueError("teacher receipt serialized bytes differ")
    files = _teacher_receipt_files(receipt_path)
    payloads = {
        name for name, record in files.items() if record["role"] == "indexed_payload"
    }
    sidecars = {
        name for name, record in files.items() if record["role"] == "r7_loader_sidecar"
    }
    identities = set(IDENTITY_FILES)
    if (
        len(payloads) != 156
        or len(sidecars) != 75
        or not identities.issubset(files)
        or set(files) != payloads | sidecars | identities
    ):
        raise ValueError("teacher receipt runtime inventory differs")

    index = _load_json(source / INDEX_FILE)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or set(weight_map.values()) != payloads:
        raise ValueError("teacher index-referenced shard allowlist differs")
    if any(
        not isinstance(key, str)
        or not isinstance(shard, str)
        or Path(shard).name != shard
        for key, shard in weight_map.items()
    ):
        raise ValueError("teacher weight map contains an unsafe entry")
    external_quant = _load_json(source / "quantization_config.json")
    embedded_config = _load_json(source / "config.json")
    if embedded_config.get("quantization_config") != external_quant:
        raise ValueError("teacher embedded/external quantization configs disagree")
    r7 = external_quant.get("r7_routed_experts")
    if (
        not isinstance(r7, dict)
        or r7.get("codebook") != GLOBAL_CODEBOOK
        or r7.get("bit_map_manifests") != sorted(sidecars)
    ):
        raise ValueError("teacher R7 runtime sidecar declaration differs")
    selected_shards: dict[int, str] = {}
    for layer in SELECTED_LAYERS:
        prefix = f"model.layers.{layer}.mlp.experts."
        names = {shard for key, shard in weight_map.items() if key.startswith(prefix)}
        expected = f"r7-experts-layer-{layer:03d}.safetensors"
        if names != {expected}:
            raise ValueError(f"teacher selected layer {layer} shard topology differs")
        selected_shards[layer] = expected
    return {
        "validation": validation,
        "files": files,
        "payloads": payloads,
        "sidecars": sidecars,
        "index": index,
        "external_quant": external_quant,
        "embedded_config": embedded_config,
        "selected_shards": selected_shards,
    }


def validate_run_seal(
    run_seal_path: str | Path,
    artifacts_root: str | Path,
) -> dict[str, Any]:
    """Validate the run seal and all four original construction artifacts."""

    root = _real_directory(artifacts_root, label="fresh artifact root")
    raw_seal_path = Path(run_seal_path).absolute()
    seal_path = _regular(raw_seal_path, label="fresh run seal")
    if seal_path.parent != root or seal_path.name != "run_seal.json":
        raise ValueError("run seal must be the root of the fresh artifact tree")
    seal = _load_json(seal_path)
    run_seal_id = seal.get("run_seal_id")
    body = {key: value for key, value in seal.items() if key != "run_seal_id"}
    legacy_census = {
        "layers": 4,
        "experts": 1_024,
        "sqg_tensors": 3_072,
        "mcg_tensors": 0,
        "K3": 1_536,
        "K4": 1_536,
        "other_rates": 0,
        "factorial_cells_per_layer": 32,
        "exact_factorial_cells_total": 128,
    }
    time_priority_census = {
        **legacy_census,
        "factorial_cells_per_layer": 16,
        "exact_factorial_cells_total": 64,
    }
    legacy_mode = (
        seal.get("census") == legacy_census
        and seal.get("scope") == _BASE_RUN_SCOPE
    )
    time_priority_mode = (
        seal.get("census") == time_priority_census
        and seal.get("scope")
        == {**_BASE_RUN_SCOPE, "routed_holdout": _TIME_PRIORITY_HOLDOUT_SCOPE}
    )
    if (
        seal.get("schema") != RUN_SEAL_SCHEMA
        or seal.get("complete") is not True
        or run_seal_id != canonical_sha256(body)
        or set(seal.get("layers", {})) != {
            str(layer) for layer in SELECTED_LAYERS
        }
        or (not legacy_mode and not time_priority_mode)
        or seal.get("lineage")
        != {
            "official_bf16": True,
            "fresh_capture": True,
            "frozen_bit_map_only": True,
            "fresh_sqg_encoding": True,
            "legacy_mcg_artifact_reads": 0,
            "fallback_count": 0,
        }
    ):
        raise ValueError("fresh four-layer run seal differs")

    preflight_path = _exact_child_file(
        root, "preflight.json", label="fresh pipeline preflight"
    )
    preflight_sha256 = sha256_file(preflight_path)
    if preflight_sha256 != _hex64(
        seal.get("preflight_sha256"), label="run-seal preflight SHA256"
    ):
        raise ValueError("run seal does not bind the exact root preflight")
    preflight = load_preflight(
        preflight_path, require_execution_environment=False
    )
    preflight_paths = preflight.get("paths")
    preflight_settings = preflight.get("settings")
    if (
        not isinstance(preflight_paths, Mapping)
        or not isinstance(preflight_settings, Mapping)
        or Path(str(preflight_paths.get("output_root"))).resolve() != root
        or preflight_settings.get("run_id") != seal.get("run_id")
    ):
        raise ValueError("run-seal preflight output/run binding differs")

    expected_binding = {
        "run_id": seal["run_id"],
        "preflight_id": preflight["preflight_id"],
        "source_seal_sha256": preflight["source_seal"]["sha256"],
        "capture_manifest_sha256": preflight["capture"]["sha256"],
        "bit_contract_sha256": preflight["bit_contract"]["sha256"],
        "kquant": preflight["kquant"],
        "sigma_reg": preflight_settings["sigma_reg"],
    }

    layers: dict[int, dict[str, Any]] = {}
    for layer in SELECTED_LAYERS:
        record = seal["layers"][str(layer)]
        common_record_fields = {
            "layer_artifact",
            "layer_artifact_sha256",
            "layer_shard_sha256",
            "selection_sha256",
            "selection_id",
            "selected_cell_id",
            "sqg_tensors",
            "mcg_tensors",
            "K3",
            "K4",
        }
        holdout_record_fields = (
            {"holdout"}
            if time_priority_mode
            else {"holdout_sha256", "holdout_report_id"}
        )
        if (
            not isinstance(record, dict)
            or set(record) != common_record_fields | holdout_record_fields
        ):
            raise ValueError(f"run-seal layer {layer} record is absent")

        layer_root = _exact_child_directory(
            root, f"layer_{layer:03d}", label=f"layer {layer} artifact directory"
        )
        profile_root = _exact_child_directory(
            layer_root,
            "profile_search",
            label=f"layer {layer} profile-search directory",
        )
        final_root = _exact_child_directory(
            layer_root, "final", label=f"layer {layer} final directory"
        )
        selection_path = _exact_child_file(
            profile_root, "selection.json", label=f"layer {layer} selection"
        )
        expected_manifest_path = _exact_child_file(
            final_root,
            f"fresh-sqg-layer-{layer:03d}.json",
            label=f"layer {layer} artifact manifest",
        )
        seal_file = _exact_child_file(
            final_root,
            f"fresh-sqg-layer-{layer:03d}.json.sha256",
            label=f"layer {layer} artifact manifest seal",
        )
        holdout_path = final_root / "holdout.json"
        if time_priority_mode:
            if holdout_path.exists() or holdout_path.is_symlink():
                raise ValueError(
                    f"run-seal layer {layer} skipped-holdout report unexpectedly exists"
                )
        else:
            holdout_path = _exact_child_file(
                final_root, "holdout.json", label=f"layer {layer} holdout report"
            )
        recorded_path = Path(str(record.get("layer_artifact")))
        if not recorded_path.is_absolute() or recorded_path != expected_manifest_path:
            raise ValueError(f"run-seal layer {layer} artifact path differs")

        selection = _load_json(selection_path)
        selection_id = _validate_canonical_id(
            selection, "selection_id", label=f"layer {layer} selection"
        )
        selection_sha256 = sha256_file(selection_path)
        binding = dict(expected_binding, layer=layer)
        selection_domain_matches = (
            selection.get("factorial_cell_count") == 16
            and selection.get("required_factorial_cell_count") == 16
            and selection.get("preregistered_factorial_cell_count") == 32
            and selection.get("selection_scope")
            == _TIME_PRIORITY_SELECTION_SCOPE
            if time_priority_mode
            else selection.get("factorial_cell_count") == 32
            and selection.get("required_factorial_cell_count") == 32
        )
        if (
            selection.get("schema") != PROFILE_SEARCH_SCHEMA
            or selection.get("complete") is not True
            or selection.get("binding") != binding
            or not selection_domain_matches
            or selection.get("all_cells_exact_sqg") is not True
            or selection.get("all_cells_candidate_conditional_h2") is not True
            or selection.get("all_cells_exact_down_sqg") is not True
            or selection.get("all_cells_routed_selection_scored") is not True
            or selection.get("proxy_pruning_used") is not False
            or selection.get("holdout_used") is not False
            or selection.get("selection_frozen_before_holdout") is not True
            or selection.get("fallback_count") != 0
            or selection.get("mcg_inputs") != 0
            or selection_sha256
            != _hex64(
                record.get("selection_sha256"),
                label=f"run-seal layer {layer} selection SHA256",
            )
            or selection_id != record.get("selection_id")
            or selection.get("selected_cell_id") != record.get("selected_cell_id")
        ):
            raise ValueError(f"run-seal layer {layer} selection binding differs")

        expected_shard_name = f"fresh-sqg-layer-{layer:03d}.safetensors"
        raw_manifest = _load_json(expected_manifest_path)
        if raw_manifest.get("shard") != expected_shard_name:
            raise ValueError(f"run-seal layer {layer} shard path is unsafe")
        shard = _exact_child_file(
            final_root, expected_shard_name, label=f"layer {layer} artifact shard"
        )
        manifest = validate_layer_artifact(expected_manifest_path)
        manifest_sha256 = sha256_file(expected_manifest_path)
        evidence = manifest.get("evidence", {})
        if (
            manifest.get("schema") != LAYER_ARTIFACT_SCHEMA
            or manifest.get("run_id") != seal.get("run_id")
            or record.get("layer_artifact_sha256")
            != manifest_sha256
            or record.get("layer_shard_sha256") != manifest.get("shard_sha256")
            or sha256_file(shard) != manifest.get("shard_sha256")
            or not isinstance(evidence, Mapping)
            or evidence.get("profile_selection_sha256") != selection_sha256
            or evidence.get("profile_selection_id") != selection_id
            or evidence.get("selected_cell_id") != selection.get("selected_cell_id")
            or record.get("sqg_tensors") != 768
            or record.get("mcg_tensors") != 0
            or record.get("K3") != EXPECTED_K3_PER_LAYER
            or record.get("K4") != EXPECTED_K4_PER_LAYER
        ):
            raise ValueError(f"run-seal layer {layer} artifact binding differs")

        holdout_sha256: str | None = None
        if time_priority_mode:
            if record.get("holdout") != _time_priority_holdout_record(
                selection_sha256, manifest_sha256
            ):
                raise ValueError(
                    f"run-seal layer {layer} skipped-holdout receipt differs"
                )
        else:
            holdout = _load_json(holdout_path)
            holdout_id = _validate_canonical_id(
                holdout, "report_id", label=f"layer {layer} holdout"
            )
            holdout_sha256 = sha256_file(holdout_path)
            routed_score = holdout.get("routed_functional_score")
            if not isinstance(routed_score, Mapping):
                raise ValueError(f"run-seal layer {layer} holdout score is absent")
            _validate_canonical_id(
                routed_score, "score_id", label=f"layer {layer} holdout score"
            )
            if (
                holdout.get("schema") != HOLDOUT_REPORT_SCHEMA
                or holdout.get("complete") is not True
                or holdout.get("binding") != binding
                or holdout.get("role") != "holdout"
                or holdout.get("selection_sha256") != selection_sha256
                or holdout.get("layer_artifact_sha256") != manifest_sha256
                or holdout.get("selection_frozen_before_holdout") is not True
                or holdout.get("artifact_changed_by_holdout") is not False
                or holdout.get("holdout_used_for_calibration") is not False
                or holdout.get("holdout_used_for_profile_choice") is not False
                or holdout.get("holdout_report_only") is not True
                or holdout.get("fallback_count") != 0
                or holdout.get("mcg_inputs") != 0
                or routed_score.get("role") != "holdout"
                or routed_score.get("expert_count") != NUM_EXPERTS
                or holdout_sha256
                != _hex64(
                    record.get("holdout_sha256"),
                    label=f"run-seal layer {layer} holdout SHA256",
                )
                or holdout_id != record.get("holdout_report_id")
            ):
                raise ValueError(f"run-seal layer {layer} holdout binding differs")
        layers[layer] = {
            "record": record,
            "manifest": manifest,
            "manifest_path": expected_manifest_path,
            "manifest_sha256": manifest_sha256,
            "manifest_seal_path": seal_file,
            "manifest_seal_sha256": sha256_file(seal_file),
            "shard_path": shard,
            "selection_path": selection_path,
            "selection_sha256": selection_sha256,
            "holdout_path": None if time_priority_mode else holdout_path,
            "holdout_sha256": holdout_sha256,
        }
    return {
        "seal": seal,
        "path": seal_path,
        "sha256": sha256_file(seal_path),
        "layers": layers,
    }


def _rewritten_quantization(source_quant: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(source_quant))
    r7 = value.get("r7_routed_experts")
    if not isinstance(r7, dict) or r7.get("codebook") != GLOBAL_CODEBOOK:
        raise ValueError("source global R7 codebook differs")
    r7["codebook_overrides"] = dict(EXPECTED_OVERRIDES)
    r7["codebook_tensor_overrides"] = {}
    return value


def _expected_layer_names(layer: int) -> set[str]:
    expected = {
        f"model.layers.{layer}.mlp.experts.r7_shared.gate_up_suh",
        f"model.layers.{layer}.mlp.experts.r7_shared.down_svh",
    }
    for expert in range(NUM_EXPERTS):
        for projection in PROJECTIONS:
            prefix = tensor_prefix(layer, expert, projection)
            expected.add(f"{prefix}.trellis")
            expected.add(f"{prefix}.sqg")
            expected.add(
                f"{prefix}.suh" if projection == "down_proj" else f"{prefix}.svh"
            )
    return expected


def _validate_selected_payload_map(
    shard: Path,
    layer: int,
    payload_sha256: Mapping[str, object],
) -> dict[str, int]:
    reader = SafeTensorReader(shard)
    expected = _expected_layer_names(layer)
    if set(reader.tensors) != expected or set(payload_sha256) != expected:
        raise ValueError(f"selected layer {layer} tensor inventory differs")
    marker_count = 0
    k3 = 0
    k4 = 0
    with shard.open("rb") as handle:
        for name, info in reader.tensors.items():
            expected_digest = _hex64(
                payload_sha256.get(name), label=f"selected payload {name} SHA256"
            )
            if info.payload.sha256() != expected_digest:
                raise ValueError(f"selected payload hash differs: {name}")
            if name.endswith(".mcg") or name.endswith(".mul1"):
                raise ValueError(f"selected layer retains legacy marker: {name}")
            if name.endswith(".sqg"):
                if info.dtype != "I32" or info.shape != () or info.nbytes != 4:
                    raise ValueError(f"SQG marker geometry differs: {name}")
                handle.seek(info.payload.start)
                value = int.from_bytes(handle.read(4), "little", signed=True)
                if value != SQG_MARKER:
                    raise ValueError(f"SQG marker payload differs: {name}")
                marker_count += 1
            elif name.endswith(".trellis"):
                if info.dtype != "I16" or len(info.shape) != 3:
                    raise ValueError(f"SQG trellis geometry differs: {name}")
                bits = info.shape[2] // 16
                if info.shape[2] % 16 or bits not in (3, 4):
                    raise ValueError(f"SQG trellis rate differs: {name}")
                if bits == 3:
                    k3 += 1
                else:
                    k4 += 1
            elif name.endswith(".suh"):
                if ".down_proj.suh" not in name and "r7_shared.gate_up_suh" not in name:
                    raise ValueError(f"stale per-tensor shared-side SUH remains: {name}")
            elif name.endswith(".svh"):
                if ".down_proj.svh" in name or (
                    ".gate_proj.svh" not in name
                    and ".up_proj.svh" not in name
                    and "r7_shared.down_svh" not in name
                ):
                    raise ValueError(f"stale per-tensor shared-side SVH remains: {name}")
    if (marker_count, k3, k4) != (768, 384, 384):
        raise ValueError(f"selected layer {layer} SQG/K3/K4 census differs")
    return {"markers": marker_count, "K3": k3, "K4": k4}


def _runtime_sidecar(
    layer: int,
    artifact: Mapping[str, Any],
    artifact_context: Mapping[str, Any],
    run_context: Mapping[str, Any],
    candidate_shard_name: str,
) -> dict[str, Any]:
    values = list(artifact.get("bit_map", {}).values())
    if (
        set(artifact.get("payload_sha256", {})) != _expected_layer_names(layer)
        or (values.count(3), values.count(4), sum(values))
        != (
            EXPECTED_K3_PER_LAYER,
            EXPECTED_K4_PER_LAYER,
            EXPECTED_BIT_UNITS_PER_LAYER,
        )
    ):
        raise ValueError(f"layer {layer} artifact bit/payload domain differs")
    value: dict[str, Any] = {
        "schema": CANDIDATE_SIDECAR_SCHEMA,
        "complete": True,
        "schema_version": 1,
        "layer": layer,
        "codebook": SQG_CODEBOOK,
        "marker": "FRESH_SQG",
        "recipe_version": "fresh-sqg-four-layer-v1",
        "shard": candidate_shard_name,
        "shard_sha256": artifact["shard_sha256"],
        "allocation_bit_units": EXPECTED_BIT_UNITS_PER_LAYER,
        "allocation_target_bpw": "3.5",
        "bit_map": artifact["bit_map"],
        "bit_histogram": {"3": EXPECTED_K3_PER_LAYER, "4": EXPECTED_K4_PER_LAYER},
        "shared_vectors": artifact["shared_vectors"],
        "vector_refs": artifact["vector_refs"],
        "payload_sha256": artifact["payload_sha256"],
        "artifact_binding": {
            "manifest_sha256": artifact_context["manifest_sha256"],
            "manifest_seal_sha256": artifact_context["manifest_seal_sha256"],
            "source_shard_sha256": artifact["shard_sha256"],
            "payload_manifest_sha256": canonical_sha256(
                artifact["payload_sha256"]
            ),
        },
        "run_seal_binding": {
            "run_id": run_context["seal"]["run_id"],
            "run_seal_id": run_context["seal"]["run_seal_id"],
            "run_seal_sha256": run_context["sha256"],
            "layer_record": run_context["seal"]["layers"][str(layer)],
        },
        "lineage": {
            "fresh_sqg": True,
            "legacy_mcg_payloads": 0,
            "legacy_mcg_transforms": 0,
            "legacy_mcg_scales": 0,
            "legacy_mcg_permutations": 0,
            "legacy_mcg_seeds": 0,
            "source_sidecar_fields_copied": [],
        },
    }
    value["sidecar_id"] = canonical_sha256(value)
    return value


def _rewritten_index(
    source_index: Mapping[str, Any],
    source: Path,
    run_context: Mapping[str, Any],
    selected_source_shards: Mapping[int, str],
) -> dict[str, Any]:
    value = deepcopy(dict(source_index))
    weight_map = value.get("weight_map")
    metadata = value.get("metadata")
    if not isinstance(weight_map, dict) or not isinstance(metadata, dict):
        raise ValueError("source weight index structure differs")
    old_payload_bytes = 0
    new_payload_bytes = 0
    for layer in SELECTED_LAYERS:
        source_shard_name = selected_source_shards[layer]
        old_keys = {key for key, shard in weight_map.items() if shard == source_shard_name}
        artifact = run_context["layers"][layer]["manifest"]
        new_keys = set(artifact["payload_sha256"])
        expected_prefix = f"model.layers.{layer}.mlp.experts."
        if (
            len(old_keys) != 2_306
            or any(not key.startswith(expected_prefix) for key in old_keys)
            or new_keys != _expected_layer_names(layer)
        ):
            raise ValueError(f"selected layer {layer} index replacement domain differs")
        old_reader = SafeTensorReader(source / source_shard_name)
        new_reader = SafeTensorReader(run_context["layers"][layer]["shard_path"])
        if set(old_reader.tensors) != old_keys or set(new_reader.tensors) != new_keys:
            raise ValueError(f"selected layer {layer} shard/index header differs")
        old_payload_bytes += sum(info.nbytes for info in old_reader.tensors.values())
        new_payload_bytes += sum(info.nbytes for info in new_reader.tensors.values())
        for key in old_keys:
            del weight_map[key]
        candidate_name = f"r7-experts-layer-{layer:03d}.safetensors"
        for key in new_keys:
            if key in weight_map:
                raise ValueError(f"fresh selected tensor collides in index: {key}")
            weight_map[key] = candidate_name
    total_size = metadata.get("total_size")
    if type(total_size) is not int or total_size <= old_payload_bytes:
        raise ValueError("source index total_size differs")
    metadata["total_size"] = total_size - old_payload_bytes + new_payload_bytes
    value["weight_map"] = dict(sorted(weight_map.items()))
    return value


def _manifest_id(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "manifest_id"}
    )


def _write_verified_marker(candidate: Path) -> None:
    manifest_sha = sha256_file(candidate / MANIFEST_NAME)
    marker = candidate / VERIFIED_NAME
    if marker.exists() or marker.is_symlink():
        raise FileExistsError(f"candidate verification marker already exists: {marker}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{VERIFIED_NAME}.", suffix=".tmp", dir=candidate
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(f"{manifest_sha}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
        _fsync_directory(candidate)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def materialize_candidate(
    *,
    source_model: str | Path,
    teacher_receipt: str | Path,
    run_seal: str | Path,
    artifacts_root: str | Path,
    bit_contract: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Build a new runnable candidate without loading or launching a model."""

    source = _real_directory(source_model, label="protected source model")
    receipt = _regular(Path(teacher_receipt).absolute(), label="teacher receipt")
    artifact_root = _real_directory(artifacts_root, label="fresh artifact root")
    run_path = _regular(Path(run_seal).absolute(), label="fresh run seal")
    bit_path = _regular(Path(bit_contract).absolute(), label="bit contract")
    raw_destination = Path(output).absolute()
    destination_parent = _real_directory(
        raw_destination.parent,
        label="candidate output parent",
    )
    destination = destination_parent / raw_destination.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"candidate output must not already exist: {destination}")
    _assert_output_disjoint(
        destination,
        {
            "source model": source,
            "teacher receipt": receipt.parent,
            "artifact tree": artifact_root,
            "bit contract": bit_path.parent,
        },
    )
    budgets = load_bit_contract(bit_path)
    source_context = _source_inventory(source, receipt, verify_teacher=True)
    run_context = validate_run_seal(run_path, artifact_root)
    for layer in SELECTED_LAYERS:
        if run_context["layers"][layer]["manifest"]["bit_map"] != budgets[layer].bit_map:
            raise ValueError(f"layer {layer} artifact differs from frozen bit contract")

    destination.mkdir(parents=False)
    _fsync_directory(destination.parent)
    records: dict[str, dict[str, object]] = {}
    selected_source_shards = set(source_context["selected_shards"].values())
    try:
        for name in sorted(source_context["payloads"] - selected_source_shards):
            source_path = source / name
            target = destination / name
            _hardlink(source_path, target)
            record = source_context["files"][name]
            records[name] = _file_record(
                target,
                role="unchanged_indexed_payload",
                materialization="hardlink",
                origin=source_path,
                expected_sha256=record["sha256"],
            )

        selected: dict[str, Any] = {}
        for layer in SELECTED_LAYERS:
            context = run_context["layers"][layer]
            artifact = context["manifest"]
            candidate_shard_name = f"r7-experts-layer-{layer:03d}.safetensors"
            candidate_shard = destination / candidate_shard_name
            copied_bytes, copied_sha = _atomic_copy(
                context["shard_path"], candidate_shard
            )
            if copied_sha != artifact["shard_sha256"]:
                raise ValueError(f"copied selected layer {layer} shard hash differs")
            records[candidate_shard_name] = {
                "bytes": copied_bytes,
                "sha256": copied_sha,
                "role": "fresh_selected_sqg_payload",
                "materialization": "copy",
                "origin": str(context["shard_path"]),
            }
            payload_census = _validate_selected_payload_map(
                candidate_shard,
                layer,
                artifact["payload_sha256"],
            )
            sidecar_name = f"r7-experts-layer-{layer:03d}.json"
            sidecar = _runtime_sidecar(
                layer,
                artifact,
                context,
                run_context,
                candidate_shard_name,
            )
            atomic_json(destination / sidecar_name, sidecar)
            records[sidecar_name] = _file_record(
                destination / sidecar_name,
                role="fresh_selected_sqg_sidecar",
                materialization="generated",
                origin=context["manifest_path"],
            )
            selected[str(layer)] = {
                "source_legacy_shard": source_context["selected_shards"][layer],
                "source_legacy_sidecar": sidecar_name,
                "candidate_shard": candidate_shard_name,
                "candidate_sidecar": sidecar_name,
                "artifact_manifest": str(context["manifest_path"]),
                "artifact_manifest_sha256": context["manifest_sha256"],
                "artifact_manifest_seal_sha256": context[
                    "manifest_seal_sha256"
                ],
                "artifact_shard_sha256": artifact["shard_sha256"],
                "payload_manifest_sha256": canonical_sha256(
                    artifact["payload_sha256"]
                ),
                "payload_census": payload_census,
                "legacy_source_inode_reused": False,
            }

        selected_sidecars = {
            f"r7-experts-layer-{layer:03d}.json" for layer in SELECTED_LAYERS
        }
        for name in sorted(source_context["sidecars"] - selected_sidecars):
            source_path = source / name
            target = destination / name
            _hardlink(source_path, target)
            record = source_context["files"][name]
            records[name] = _file_record(
                target,
                role="unchanged_r7_sidecar",
                materialization="hardlink",
                origin=source_path,
                expected_sha256=record["sha256"],
            )

        for name in sorted(set(IDENTITY_FILES) - REWRITTEN_IDENTITY_FILES):
            source_path = source / name
            target = destination / name
            _hardlink(source_path, target)
            record = source_context["files"][name]
            records[name] = _file_record(
                target,
                role="unchanged_loader_identity",
                materialization="hardlink",
                origin=source_path,
                expected_sha256=record["sha256"],
            )

        quant = _rewritten_quantization(source_context["external_quant"])
        config = deepcopy(source_context["embedded_config"])
        config["quantization_config"] = quant
        index = _rewritten_index(
            source_context["index"],
            source,
            run_context,
            source_context["selected_shards"],
        )
        for name, value in (
            ("config.json", config),
            ("quantization_config.json", quant),
            (INDEX_FILE, index),
        ):
            atomic_json(destination / name, value)
            records[name] = _file_record(
                destination / name,
                role="rewritten_loader_identity",
                materialization="generated",
                origin=source / name,
            )

        copied_run_bytes, copied_run_sha = _atomic_copy(
            run_context["path"], destination / RUN_SEAL_COPY_NAME
        )
        if copied_run_sha != run_context["sha256"]:
            raise ValueError("candidate run-seal copy differs")
        records[RUN_SEAL_COPY_NAME] = {
            "bytes": copied_run_bytes,
            "sha256": copied_run_sha,
            "role": "fresh_run_seal",
            "materialization": "copy",
            "origin": str(run_context["path"]),
        }

        manifest: dict[str, Any] = {
            "schema": CANDIDATE_SCHEMA,
            "complete": True,
            "source": {
                "root": str(source),
                "teacher_receipt": str(receipt),
                "teacher_identity": source_context["validation"],
                "source_bytes_mutated": False,
                "recursive_clone_used": False,
            },
            "run_seal": {
                "external_path": str(run_context["path"]),
                "candidate_copy": RUN_SEAL_COPY_NAME,
                "sha256": run_context["sha256"],
                "run_seal_id": run_context["seal"]["run_seal_id"],
                "run_id": run_context["seal"]["run_id"],
            },
            "bit_contract": {
                "path": str(bit_path),
                "sha256": sha256_file(bit_path),
                "schema": BIT_CONTRACT_SCHEMA,
                "only_inherited_quantization_control": "per-tensor K3/K4 assignment",
            },
            "selected_layers": list(SELECTED_LAYERS),
            "codebooks": {
                "global_unselected": GLOBAL_CODEBOOK,
                "selected_overrides": dict(EXPECTED_OVERRIDES),
                "tensor_overrides": {},
            },
            "selected": selected,
            "runtime_allowlist": {
                "construction": "explicit_index_and_loader_metadata_allowlist",
                "recursive_source_clone": False,
                "file_count": len(records),
                "files": dict(sorted(records.items())),
                "source_calibration_files_copied": 0,
                "source_encoding_artifacts_copied": 0,
                "source_documentation_files_copied": 0,
                "stale_source_manifest_files_copied": 0,
                "selected_legacy_shards_linked": 0,
                "selected_legacy_sidecars_linked": 0,
            },
            "construction_exclusions": {
                "mcg_payload_bytes": 0,
                "mcg_transform_vectors": 0,
                "mcg_scale_vectors": 0,
                "mcg_permutations": 0,
                "mcg_encoder_seeds": 0,
                "mcg_decoded_weight_reads": 0,
                "stale_per_tensor_shared_side_scales": 0,
            },
            "model_workload_launched": False,
            "production_container_touched": False,
        }
        manifest["manifest_id"] = _manifest_id(manifest)
        if set(path.name for path in destination.iterdir()) != set(records):
            raise ValueError("candidate contains a file outside the runtime allowlist")
        atomic_json(destination / MANIFEST_NAME, manifest)
        validate_candidate(
            destination,
            source_model=source,
            teacher_receipt=receipt,
            run_seal=run_path,
            artifacts_root=artifact_root,
            bit_contract=bit_path,
            verify_all_hashes=False,
            require_verified_marker=False,
            _source_context=source_context,
            _run_context=run_context,
        )
        _write_verified_marker(destination)
        return validate_candidate(
            destination,
            source_model=source,
            teacher_receipt=receipt,
            run_seal=run_path,
            artifacts_root=artifact_root,
            bit_contract=bit_path,
            verify_all_hashes=False,
            require_verified_marker=True,
            _source_context=source_context,
            _run_context=run_context,
        )
    except BaseException:
        # Partial directories intentionally remain forensic evidence and never
        # receive the sole verification marker.  A retry must choose a new path.
        raise


def validate_candidate(
    candidate: str | Path,
    *,
    source_model: str | Path,
    teacher_receipt: str | Path,
    run_seal: str | Path,
    artifacts_root: str | Path,
    bit_contract: str | Path,
    verify_all_hashes: bool = True,
    require_verified_marker: bool = True,
    _source_context: dict[str, Any] | None = None,
    _run_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Independently close candidate topology, bytes, lineage, and config."""

    root = _real_directory(candidate, label="candidate model")
    source = _real_directory(source_model, label="protected source model")
    receipt = _regular(Path(teacher_receipt).resolve(), label="teacher receipt")
    artifact_root = _real_directory(artifacts_root, label="fresh artifact root")
    run_path = _regular(Path(run_seal).resolve(), label="fresh run seal")
    bit_path = _regular(Path(bit_contract).resolve(), label="bit contract")
    _assert_output_disjoint(
        root,
        {
            "source model": source,
            "teacher receipt": receipt.parent,
            "artifact tree": artifact_root,
            "bit contract": bit_path.parent,
        },
    )
    source_context = _source_context or _source_inventory(
        source,
        receipt,
        verify_teacher=True,
    )
    run_context = _run_context or validate_run_seal(run_path, artifact_root)
    budgets = load_bit_contract(bit_path)
    manifest_path = _regular(root / MANIFEST_NAME, label="candidate manifest")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != CANDIDATE_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("manifest_id") != _manifest_id(manifest)
        or manifest.get("selected_layers") != list(SELECTED_LAYERS)
        or manifest.get("model_workload_launched") is not False
        or manifest.get("production_container_touched") is not False
    ):
        raise ValueError("candidate root manifest differs")
    if manifest.get("source") != {
        "root": str(source),
        "teacher_receipt": str(receipt),
        "teacher_identity": source_context["validation"],
        "source_bytes_mutated": False,
        "recursive_clone_used": False,
    }:
        raise ValueError("candidate source binding differs")
    if manifest.get("run_seal") != {
        "external_path": str(run_context["path"]),
        "candidate_copy": RUN_SEAL_COPY_NAME,
        "sha256": run_context["sha256"],
        "run_seal_id": run_context["seal"]["run_seal_id"],
        "run_id": run_context["seal"]["run_id"],
    }:
        raise ValueError("candidate run-seal binding differs")
    if manifest.get("bit_contract") != {
        "path": str(bit_path),
        "sha256": sha256_file(bit_path),
        "schema": BIT_CONTRACT_SCHEMA,
        "only_inherited_quantization_control": "per-tensor K3/K4 assignment",
    }:
        raise ValueError("candidate bit-contract binding differs")
    if manifest.get("codebooks") != {
        "global_unselected": GLOBAL_CODEBOOK,
        "selected_overrides": dict(EXPECTED_OVERRIDES),
        "tensor_overrides": {},
    }:
        raise ValueError("candidate codebook override contract differs")
    if manifest.get("construction_exclusions") != {
        "mcg_payload_bytes": 0,
        "mcg_transform_vectors": 0,
        "mcg_scale_vectors": 0,
        "mcg_permutations": 0,
        "mcg_encoder_seeds": 0,
        "mcg_decoded_weight_reads": 0,
        "stale_per_tensor_shared_side_scales": 0,
    }:
        raise ValueError("candidate construction exclusions differ")

    allowlist = manifest.get("runtime_allowlist")
    if not isinstance(allowlist, dict) or not isinstance(allowlist.get("files"), dict):
        raise ValueError("candidate runtime allowlist is absent")
    records = allowlist["files"]
    expected_root = set(records) | {MANIFEST_NAME}
    if require_verified_marker:
        expected_root.add(VERIFIED_NAME)
    observed_root = {path.name for path in root.iterdir()}
    if observed_root != expected_root or any(path.is_dir() for path in root.iterdir()):
        raise ValueError("candidate root contains an unexpected/missing runtime file")
    if allowlist != {
        "construction": "explicit_index_and_loader_metadata_allowlist",
        "recursive_source_clone": False,
        "file_count": len(records),
        "files": records,
        "source_calibration_files_copied": 0,
        "source_encoding_artifacts_copied": 0,
        "source_documentation_files_copied": 0,
        "stale_source_manifest_files_copied": 0,
        "selected_legacy_shards_linked": 0,
        "selected_legacy_sidecars_linked": 0,
    }:
        raise ValueError("candidate runtime allowlist counters differ")

    selected_source_shards = set(source_context["selected_shards"].values())
    expected_files = (
        source_context["payloads"]
        | source_context["sidecars"]
        | set(IDENTITY_FILES)
        | {RUN_SEAL_COPY_NAME}
    )
    if set(records) != expected_files:
        raise ValueError("candidate runtime filename allowlist differs")
    for name, record in records.items():
        path = _regular(root / name, label=f"candidate allowlisted {name}")
        if (
            not isinstance(record, dict)
            or set(record) != {
                "bytes",
                "sha256",
                "role",
                "materialization",
                "origin",
            }
            or type(record.get("bytes")) is not int
            or record["bytes"] != path.stat().st_size
        ):
            raise ValueError(f"candidate file record differs: {name}")
        _hex64(record.get("sha256"), label=f"candidate {name} SHA256")
        if verify_all_hashes and sha256_file(path) != record["sha256"]:
            raise ValueError(f"candidate file SHA256 differs: {name}")

    quant = _rewritten_quantization(source_context["external_quant"])
    config = deepcopy(source_context["embedded_config"])
    config["quantization_config"] = quant
    if (
        _load_json(root / "quantization_config.json") != quant
        or _load_json(root / "config.json") != config
        or _load_json(root / "config.json").get("quantization_config")
        != _load_json(root / "quantization_config.json")
    ):
        raise ValueError("candidate embedded/external quantization rewrite differs")
    r7 = quant["r7_routed_experts"]
    if (
        r7.get("codebook") != GLOBAL_CODEBOOK
        or r7.get("codebook_overrides") != EXPECTED_OVERRIDES
        or r7.get("codebook_tensor_overrides") != {}
    ):
        raise ValueError("candidate selected-layer-only codebook policy differs")

    expected_index = _rewritten_index(
        source_context["index"],
        source,
        run_context,
        source_context["selected_shards"],
    )
    candidate_index = _load_json(root / INDEX_FILE)
    if candidate_index != expected_index:
        raise ValueError("candidate weight-index rewrite differs")
    weight_map = candidate_index["weight_map"]

    selected_reports: list[dict[str, Any]] = []
    selected_manifest = manifest.get("selected")
    if not isinstance(selected_manifest, dict) or set(selected_manifest) != {
        str(layer) for layer in SELECTED_LAYERS
    }:
        raise ValueError("candidate selected-layer manifest domain differs")
    for layer in SELECTED_LAYERS:
        context = run_context["layers"][layer]
        artifact = context["manifest"]
        if artifact["bit_map"] != budgets[layer].bit_map:
            raise ValueError(f"candidate layer {layer} frozen bit map differs")
        shard_name = f"r7-experts-layer-{layer:03d}.safetensors"
        sidecar_name = f"r7-experts-layer-{layer:03d}.json"
        shard = root / shard_name
        sidecar = _load_json(root / sidecar_name)
        expected_sidecar = _runtime_sidecar(
            layer,
            artifact,
            context,
            run_context,
            shard_name,
        )
        if sidecar != expected_sidecar:
            raise ValueError(f"candidate layer {layer} runtime sidecar differs")
        if os.path.samefile(shard, source / source_context["selected_shards"][layer]):
            raise ValueError(f"candidate layer {layer} reused the legacy MCG shard inode")
        if os.path.samefile(root / sidecar_name, source / sidecar_name):
            raise ValueError(f"candidate layer {layer} reused the legacy MCG sidecar inode")
        if sha256_file(shard) != artifact["shard_sha256"]:
            raise ValueError(f"candidate layer {layer} copied shard differs")
        census = _validate_selected_payload_map(
            shard,
            layer,
            artifact["payload_sha256"],
        )
        selected_keys = {
            key
            for key, mapped in weight_map.items()
            if mapped == shard_name
        }
        if selected_keys != set(artifact["payload_sha256"]):
            raise ValueError(f"candidate layer {layer} index payload domain differs")
        expected_selected = {
            "source_legacy_shard": source_context["selected_shards"][layer],
            "source_legacy_sidecar": sidecar_name,
            "candidate_shard": shard_name,
            "candidate_sidecar": sidecar_name,
            "artifact_manifest": str(context["manifest_path"]),
            "artifact_manifest_sha256": context["manifest_sha256"],
            "artifact_manifest_seal_sha256": context["manifest_seal_sha256"],
            "artifact_shard_sha256": artifact["shard_sha256"],
            "payload_manifest_sha256": canonical_sha256(
                artifact["payload_sha256"]
            ),
            "payload_census": census,
            "legacy_source_inode_reused": False,
        }
        if selected_manifest[str(layer)] != expected_selected:
            raise ValueError(f"candidate layer {layer} manifest binding differs")
        selected_reports.append(
            {
                "layer": layer,
                "trellis_tensors": 768,
                "sqg_markers": census["markers"],
                "mcg_markers": 0,
                "bit_histogram": {"3": census["K3"], "4": census["K4"], "5": 0},
                "bit_map_matches_sanitized_contract": True,
            }
        )

    selected_sidecars = {
        f"r7-experts-layer-{layer:03d}.json" for layer in SELECTED_LAYERS
    }
    unchanged_names = (
        (source_context["payloads"] - selected_source_shards)
        | (source_context["sidecars"] - selected_sidecars)
        | (set(IDENTITY_FILES) - REWRITTEN_IDENTITY_FILES)
    )
    for name in unchanged_names:
        candidate_path = root / name
        source_path = source / name
        record = records[name]
        source_record = source_context["files"][name]
        if (
            record["materialization"] != "hardlink"
            or record["origin"] != str(source_path)
            or record["sha256"] != source_record["sha256"]
            or record["bytes"] != source_record["bytes"]
            or not os.path.samefile(candidate_path, source_path)
        ):
            raise ValueError(f"unchanged runtime file binding differs: {name}")

    if _load_json(root / RUN_SEAL_COPY_NAME) != run_context["seal"]:
        raise ValueError("candidate run-seal copy differs")
    if require_verified_marker:
        marker = _regular(root / VERIFIED_NAME, label="candidate verification marker")
        expected_marker = f"{sha256_file(manifest_path)}\n".encode("ascii")
        if marker.read_bytes() != expected_marker:
            raise ValueError("candidate verification marker differs")

    return {
        "schema": CANDIDATE_REPORT_SCHEMA,
        "candidate": str(root),
        "protected_source": str(source),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "candidate_manifest_id": manifest["manifest_id"],
        "run_seal_sha256": run_context["sha256"],
        "run_seal_id": run_context["seal"]["run_seal_id"],
        "sanitized_bit_contract": manifest["bit_contract"],
        "construction_exclusions": manifest["construction_exclusions"],
        "selected_layers": list(SELECTED_LAYERS),
        "global_unselected_layer_codebook": GLOBAL_CODEBOOK,
        "selected_layer_codebook": SQG_CODEBOOK,
        "tensor_overrides": {},
        "sqg_marker": {"suffix": ".sqg", "int32": SQG_MARKER},
        "selected_trellis_tensors": 3_072,
        "selected_sqg_markers": 3_072,
        "selected_mcg_markers": 0,
        "sqg_markers_outside_selected_layers": 0,
        "marker_payloads_verified": True,
        "selected_payload_hashes_verified": True,
        "selected_legacy_inodes_reused": 0,
        "source_bytes_mutated": False,
        "runtime_allowlist_file_count": len(records),
        "verified_marker_present": require_verified_marker,
        "layers": selected_reports,
    }
