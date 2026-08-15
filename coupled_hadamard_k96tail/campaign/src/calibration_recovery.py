"""CPU-only recovery for the completed 2026-08-10 capture promotion.

The live worker fsynced and promoted every payload before the host rejected
the first v1 layer manifest.  Its RPC result was process-local, so the exact
raw-router hash/statistics and reference-route diagnostics no longer exist.
This module never synthesizes those values.  It emits a distinct recovery ABI
which seals everything recoverable from disk and records the missing fields as
unavailable.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path, PurePosixPath
import stat
import uuid

import numpy as np

from .calibration_capture import (
    CAPTURE_SCHEMA,
    CONSTRUCTION_EXCLUSIONS,
    FILE_ABI,
    FULL_CAPTURE_PREFLIGHT_SCHEMA,
    HIDDEN,
    JIT_CACHE_BINDING_FILE,
    JIT_CACHE_BINDING_SCHEMA,
    NUM_EXPERTS,
    OWNER_DOCUMENTS,
    OWNER_TOKENS,
    REFERENCE_ROUTE_ATOL,
    REFERENCE_ROUTE_DESCRIPTION,
    REFERENCE_ROUTE_RTOL,
    REFERENCE_ROUTE_SCHEMA,
    REFERENCE_ROUTE_SELECTION_ATOL,
    ROUTED_SCALING_FACTOR,
    ROUTING_SOURCE,
    SELECTED_LAYERS,
    TOPK,
    TOPK_WEIGHT_SEMANTICS,
    _read_json,
    _validate_document_vectors,
    _validate_routes,
    expected_document_audit_sha256,
    expected_role_rows,
    layer_dir,
    validate_capture_code_evidence,
    validate_dcp4_smoke_evidence,
    validate_full_capture_preflight_record,
    validate_router_audit,
    validate_smoke_run_token,
)
from .calibration_plan import (
    EXPECTED_SPLIT,
    atomic_json,
    canonical_json_bytes,
    load_document_plan,
    sha256_file,
)
from .capture_runtime import RUNTIME_IMAGE_ID, RUNTIME_IMAGE_REFERENCE


RECOVERED_CAPTURE_SCHEMA = "glm52-fresh-sqg-calibration-capture-recovered-v1"
RECOVERED_LAYER_SCHEMA = "glm52-fresh-sqg-layer-capture-recovered-v1"
RECOVERY_RECEIPT_SCHEMA = "glm52-fresh-sqg-post-finalize-recovery-receipt-v1"
RECOVERY_CODE_SCHEMA = "glm52-fresh-sqg-recovery-code-manifest-v1"
RECOVERY_EVIDENCE_MODE = "recovered_post_finalize_fail_closed"
RECOVERY_CODE_FILES = (
    "capture_calibration.py",
    "src/calibration_capture.py",
    "src/calibration_recovery.py",
)

# This recovery is deliberately incident-specific.  That archived manifest is
# independently present in both the successful smoke evidence and its sealed
# pre-full JIT binding stamp.
ARCHIVED_CAPTURE_CODE_MANIFEST_SHA256 = (
    "fe7423354f267a17eb9e14861575d8a0b73f9bc741e557a27b06f26d66f2cd79"
)
ARCHIVED_DRIVER_SHA256 = (
    "0787e2a0d0a3736fc2afba0749b5a421ab13b70dd0d668ea3fbe54b467d7ed93"
)
ARCHIVED_WORKER_SHA256 = (
    "f7b2e6dd72e3dbb2f5dbd82ad94037a6dba1963a5066f62694262e4dab01612f"
)
ARCHIVED_CONTRACT_SHA256 = (
    "c25781cbf86ec7063202403811fc935303069eb708d166b070b7ba3b93c86dec"
)


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _safe_relative(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is absent")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{label} is unsafe")
    return value


def recovery_code_evidence(project_root: str | Path) -> dict:
    root = Path(project_root).resolve()
    files: dict[str, dict[str, object]] = {}
    for relative in RECOVERY_CODE_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"recovery code file is absent/unsafe: {relative}")
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "schema": RECOVERY_CODE_SCHEMA,
        "allowlist": list(RECOVERY_CODE_FILES),
        "files": files,
        "manifest_sha256": _canonical_digest(files),
    }


def _validate_recovery_code(value: object, project_root: str | Path | None) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "allowlist",
        "files",
        "manifest_sha256",
    }:
        raise ValueError("recovery code manifest field set differs")
    if (
        value.get("schema") != RECOVERY_CODE_SCHEMA
        or value.get("allowlist") != list(RECOVERY_CODE_FILES)
        or not isinstance(value.get("files"), dict)
        or set(value["files"]) != set(RECOVERY_CODE_FILES)
        or value.get("manifest_sha256") != _canonical_digest(value["files"])
    ):
        raise ValueError("recovery code manifest differs")
    for relative, record in value["files"].items():
        if (
            not isinstance(record, dict)
            or set(record) != {"bytes", "sha256"}
            or type(record.get("bytes")) is not int
            or int(record["bytes"]) <= 0
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
        ):
            raise ValueError(f"recovery code entry differs: {relative}")
    if project_root is not None and value != recovery_code_evidence(project_root):
        raise ValueError("recovery code differs from current executable bytes")
    return dict(value)


def _validate_archived_capture_code(value: object) -> dict:
    code = validate_capture_code_evidence(value)
    if code.get("manifest_sha256") != ARCHIVED_CAPTURE_CODE_MANIFEST_SHA256:
        raise ValueError("archived capture-code manifest is not recoverable")
    files = code["files"]
    expected = {
        "capture_calibration.py": ARCHIVED_DRIVER_SHA256,
        "src/calibration_capture.py": ARCHIVED_CONTRACT_SHA256,
        "src/glm52_capture_worker.py": ARCHIVED_WORKER_SHA256,
    }
    for name, digest in expected.items():
        if files.get(name, {}).get("sha256") != digest:
            raise ValueError(f"archived recovery-critical code differs: {name}")
    return code


def _validate_inventory(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "directories",
        "files",
        "inventory_sha256",
    }:
        raise ValueError("archived JIT inventory field set differs")
    directories = value.get("directories")
    files = value.get("files")
    if not isinstance(directories, list) or not isinstance(files, dict):
        raise ValueError("archived JIT inventory body differs")
    if len(directories) != len(set(directories)):
        raise ValueError("archived JIT inventory repeats a directory")
    for relative in directories:
        _safe_relative(relative, label="archived JIT directory")
    for relative, record in files.items():
        _safe_relative(relative, label="archived JIT file")
        if (
            not isinstance(record, dict)
            or set(record) != {"bytes", "sha256"}
            or type(record.get("bytes")) is not int
            or int(record["bytes"]) < 0
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
        ):
            raise ValueError(f"archived JIT file record differs: {relative}")
    body = {"directories": directories, "files": files}
    if value.get("inventory_sha256") != _canonical_digest(body):
        raise ValueError("archived JIT inventory digest differs")
    return dict(value)


def _archived_preflight(
    *,
    jit_cache: Path,
    smoke_dir: Path,
    plan_path: Path,
    project_root: Path,
) -> tuple[dict, dict, dict, str]:
    smoke_path = smoke_dir / "dcp4_smoke_evidence.json"
    smoke_raw = _read_json(smoke_path)
    archived_code = _validate_archived_capture_code(
        smoke_raw.get("runtime", {}).get("capture_code")
    )
    smoke = validate_dcp4_smoke_evidence(
        smoke_dir,
        plan_path=plan_path,
        project_root=project_root,
        archived_capture_code=archived_code,
    )
    stamp_path = jit_cache / JIT_CACHE_BINDING_FILE
    if not stamp_path.is_file() or stamp_path.is_symlink():
        raise ValueError("archived JIT binding stamp is absent/unsafe")
    stamp = _read_json(stamp_path)
    expected_fields = {
        "schema",
        "complete",
        "jit_cache",
        "smoke",
        "plan",
        "runtime",
        "cache_inventory",
        "binding_id",
    }
    if (
        set(stamp) != expected_fields
        or stamp.get("schema") != JIT_CACHE_BINDING_SCHEMA
        or stamp.get("complete") is not True
    ):
        raise ValueError("archived JIT binding field set/schema differs")
    body = {key: item for key, item in stamp.items() if key != "binding_id"}
    if stamp.get("binding_id") != _canonical_digest(body):
        raise ValueError("archived JIT binding ID differs")
    inventory = _validate_inventory(stamp.get("cache_inventory"))
    cache_stat = jit_cache.stat()
    if stamp.get("jit_cache") != {
        "path": str(jit_cache),
        "device": int(cache_stat.st_dev),
        "inode": int(cache_stat.st_ino),
    }:
        raise ValueError("archived JIT cache identity differs")
    smoke_sha = sha256_file(smoke_path)
    if stamp.get("smoke") != {
        "directory": str(smoke_dir),
        "evidence_sha256": smoke_sha,
        "run_token": smoke["smoke_run_token"],
    }:
        raise ValueError("archived JIT smoke binding differs")
    plan = load_document_plan(plan_path)
    if stamp.get("plan") != {
        "sha256": sha256_file(plan_path),
        "fingerprint": plan["plan_fingerprint"],
        "first_document": plan["documents"][0],
    }:
        raise ValueError("archived JIT plan binding differs")
    if stamp.get("runtime") != {
        "image_reference": RUNTIME_IMAGE_REFERENCE,
        "image_id": RUNTIME_IMAGE_ID,
        "capture_code": archived_code,
    }:
        raise ValueError("archived JIT runtime/code binding differs")
    preflight = {
        "schema": FULL_CAPTURE_PREFLIGHT_SCHEMA,
        "smoke": {
            "evidence_sha256": smoke_sha,
            "run_token": smoke["smoke_run_token"],
        },
        "jit_cache": {
            "binding_id": stamp["binding_id"],
            "binding_stamp_sha256": sha256_file(stamp_path),
            "inventory_sha256": inventory["inventory_sha256"],
            "device": int(cache_stat.st_dev),
            "inode": int(cache_stat.st_ino),
        },
        "plan": {
            "sha256": sha256_file(plan_path),
            "fingerprint": plan["plan_fingerprint"],
        },
        "runtime": {
            "image_reference": RUNTIME_IMAGE_REFERENCE,
            "image_id": RUNTIME_IMAGE_ID,
            "capture_code_manifest_sha256": archived_code["manifest_sha256"],
        },
    }
    preflight["receipt_sha256"] = _canonical_digest(preflight)
    validate_full_capture_preflight_record(
        preflight, plan=plan, capture_code=archived_code
    )
    return smoke, archived_code, preflight, sha256_file(stamp_path)


def _identity(path: Path) -> tuple[int, int, int, int, int]:
    item = path.lstat()
    if stat.S_ISLNK(item.st_mode) or not stat.S_ISREG(item.st_mode):
        raise ValueError(f"capture payload is not a regular file: {path}")
    if item.st_nlink != 1:
        raise ValueError(f"capture payload has another hard link: {path}")
    return (
        int(item.st_dev),
        int(item.st_ino),
        int(item.st_size),
        int(item.st_mtime_ns),
        int(item.st_ctime_ns),
    )


def _hash_file(path: Path, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=chunk_bytes) as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _hidden_record(path: Path) -> tuple[str, dict[str, int]]:
    digest = hashlib.sha256()
    words = nonfinite = zero = 0
    with path.open("rb", buffering=64 << 20) as handle:
        while block := handle.read(64 << 20):
            digest.update(block)
            values = np.frombuffer(block, dtype="<u2")
            words += int(values.size)
            nonfinite += int(np.count_nonzero((values & 0x7F80) == 0x7F80))
            zero += int(np.count_nonzero((values & 0x7FFF) == 0))
    return digest.hexdigest(), {
        "bfloat16_words": words,
        "nonfinite_words": nonfinite,
        "zero_words": zero,
    }


def _expected_layer_names(*, with_manifest: bool) -> set[str]:
    names = set(FILE_ABI)
    if with_manifest:
        names.add("layer_manifest.json")
    return names


def _require_payload_layout(root: Path, *, with_manifests: bool) -> None:
    root_names = {path.name for path in root.iterdir()}
    expected = {
        "document_plan.json",
        *(f"layer_{layer:03d}" for layer in SELECTED_LAYERS),
    }
    if with_manifests:
        expected.add("capture_manifest.json")
    if root_names != expected:
        raise ValueError("recovery capture root contains unexpected/missing entries")
    for layer in SELECTED_LAYERS:
        directory = layer_dir(root, layer)
        item = directory.lstat()
        if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
            raise ValueError(f"recovery layer directory is absent/unsafe: {layer}")
        names = {path.name for path in directory.iterdir()}
        if names != _expected_layer_names(with_manifest=with_manifests):
            raise ValueError(f"recovery layer {layer} file set differs")


def _unavailable_route_evidence(layer: int, router_audit: dict) -> dict:
    return {
        "schema": "glm52-fresh-sqg-recovered-route-evidence-v1",
        "layer": layer,
        "outcome": "passed_fail_closed_online_before_payload_write",
        "checked_rows_inferred_from_successful_promotion": OWNER_TOKENS,
        "raw_router_return_weights": {
            "available": False,
            "reason": "process-local worker result was not persisted",
            "payload_stored": False,
            "effective_payload": "topk_weights.f32le.bin",
            "dtype": "float32-le",
            "shape": [OWNER_TOKENS, TOPK],
            "bytes": OWNER_TOKENS * TOPK * 4,
            "sha256": None,
            "sum": None,
            "sq_sum": None,
            "row_sum_min": None,
            "row_sum_max": None,
            "router_return_scale": router_audit["router_routed_scaling_factor"],
            "runner_output_scale": router_audit["runner_output_scale"],
            "effective_scale_product": ROUTED_SCALING_FACTOR,
        },
        "reference_route_diagnostics": {
            "available": False,
            "reason": "process-local worker result was not persisted",
            "schema_used_online": REFERENCE_ROUTE_SCHEMA,
            "reference_used_online": REFERENCE_ROUTE_DESCRIPTION,
            "selection_atol": REFERENCE_ROUTE_SELECTION_ATOL,
            "weight_rtol": REFERENCE_ROUTE_RTOL,
            "weight_atol": REFERENCE_ROUTE_ATOL,
            "selection_violation_rows": None,
            "boundary_ambiguous_rows": None,
            "max_selection_violation": None,
            "weight_mismatch_rows": None,
            "max_abs_weight_error": None,
        },
    }


def _validate_unavailable_route_evidence(
    value: object, *, layer: int, router_audit: dict
) -> dict:
    expected = _unavailable_route_evidence(layer, router_audit)
    if value != expected:
        raise ValueError(f"layer {layer}: recovered unavailable-route evidence differs")
    return dict(value)


def _payload_records(root: Path, plan: dict) -> tuple[dict, dict]:
    identities: dict[Path, tuple[int, int, int, int, int]] = {}
    seen_inodes: set[tuple[int, int]] = set()
    records: dict[str, dict] = {}
    statistics: dict[str, dict] = {}
    for layer in SELECTED_LAYERS:
        directory = layer_dir(root, layer)
        layer_files: dict[str, dict] = {}
        hidden_validation: dict[str, int] | None = None
        for name, abi in FILE_ABI.items():
            path = directory / name
            identity = _identity(path)
            if identity[:2] in seen_inodes:
                raise ValueError("two recovery payload names alias one inode")
            seen_inodes.add(identity[:2])
            identities[path] = identity
            expected_bytes = OWNER_TOKENS * int(abi["bytes_per_row"])
            if identity[2] != expected_bytes:
                raise ValueError(f"layer {layer}: {name} size differs")
            if name == "hidden.bf16.bin":
                digest, hidden_validation = _hidden_record(path)
                if hidden_validation["nonfinite_words"] != 0:
                    raise ValueError(f"layer {layer}: hidden payload contains nonfinite BF16")
            else:
                digest = _hash_file(path)
            layer_files[name] = {
                "bytes": expected_bytes,
                "sha256": digest,
                "dtype": abi["dtype"],
                "shape": (
                    [OWNER_TOKENS, HIDDEN]
                    if name.startswith("hidden")
                    else [OWNER_TOKENS, TOPK]
                    if name.startswith("topk")
                    else [OWNER_TOKENS]
                ),
            }
        _validate_document_vectors(directory, plan)
        routed, gate_stats = _validate_routes(directory)
        if hidden_validation is None:
            raise AssertionError("hidden validation was not constructed")
        records[str(layer)] = layer_files
        statistics[str(layer)] = {
            "routed_counts": routed,
            **gate_stats,
            "hidden_validation": hidden_validation,
        }
    for path, before in identities.items():
        if _identity(path) != before:
            raise ValueError(f"capture payload changed during recovery scan: {path}")
    return records, statistics


def _receipt(
    *,
    plan: dict,
    files: dict,
    smoke: dict,
    archived_code: dict,
    preflight: dict,
    binding_stamp_sha256: str,
    recovery_code: dict,
) -> dict:
    payload_digest = _canonical_digest(files)
    body = {
        "schema": RECOVERY_RECEIPT_SCHEMA,
        "evidence_mode": RECOVERY_EVIDENCE_MODE,
        "source_capture_schema": CAPTURE_SCHEMA,
        "archived_capture_code": archived_code,
        "recovery_code": recovery_code,
        "promotion_proof": {
            "worker_entrypoint": (
                "src.glm52_capture_worker.FreshSQGCaptureWorkerExtension."
                "fresh_sqg_capture_finalize"
            ),
            "reference_check_order": "fail-closed before every payload row write",
            "promotion_order": (
                "document/token closure, restore routers, flush, fsync, close, "
                "then os.replace each partial with its final name"
            ),
            "all_expected_final_names_observed": True,
            "partial_payloads_observed": 0,
            "final_payload_count": len(SELECTED_LAYERS) * len(FILE_ABI),
            "documents_verified_by_promotion_precondition": OWNER_DOCUMENTS,
            "rows_per_layer_verified_by_promotion_precondition": OWNER_TOKENS,
            "payload_manifest_sha256": payload_digest,
        },
        "unavailable_ephemeral_worker_evidence": {
            "original_capture_run_uuid": None,
            "raw_router_return_sha256_and_statistics": None,
            "reference_route_diagnostics": None,
            "reason": (
                "worker finalization returned them only in process memory; host v1 "
                "manifest validation failed before persistence"
            ),
        },
        "runtime_binding": {
            "source": "bound successful DCP4 smoke plus enforced pre-full receipt",
            "smoke_evidence_sha256": preflight["smoke"]["evidence_sha256"],
            "smoke_run_token": validate_smoke_run_token(
                preflight["smoke"]["run_token"]
            ),
            "smoke_runtime": smoke["runtime"],
            "router_audits": smoke["router_audits"],
            "full_capture_preflight": preflight,
            "jit_binding_stamp_sha256": binding_stamp_sha256,
            "post_full_jit_inventory_reused_as_preflight": False,
        },
        "document_plan": {
            "sha256": _canonical_digest(plan),
            "fingerprint": plan["plan_fingerprint"],
        },
    }
    body["receipt_sha256"] = _canonical_digest(body)
    return body


def _validate_receipt(
    value: object,
    *,
    plan: dict,
    files: dict,
    project_root: Path | None,
) -> dict:
    if not isinstance(value, dict) or value.get("schema") != RECOVERY_RECEIPT_SCHEMA:
        raise ValueError("recovery receipt is absent or uses another schema")
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value.get("receipt_sha256") != _canonical_digest(body):
        raise ValueError("recovery receipt digest differs")
    if value.get("evidence_mode") != RECOVERY_EVIDENCE_MODE:
        raise ValueError("recovery evidence mode differs")
    archived = _validate_archived_capture_code(value.get("archived_capture_code"))
    _validate_recovery_code(value.get("recovery_code"), project_root)
    proof = value.get("promotion_proof")
    expected_proof = {
        "worker_entrypoint": (
            "src.glm52_capture_worker.FreshSQGCaptureWorkerExtension."
            "fresh_sqg_capture_finalize"
        ),
        "reference_check_order": "fail-closed before every payload row write",
        "promotion_order": (
            "document/token closure, restore routers, flush, fsync, close, "
            "then os.replace each partial with its final name"
        ),
        "all_expected_final_names_observed": True,
        "partial_payloads_observed": 0,
        "final_payload_count": len(SELECTED_LAYERS) * len(FILE_ABI),
        "documents_verified_by_promotion_precondition": OWNER_DOCUMENTS,
        "rows_per_layer_verified_by_promotion_precondition": OWNER_TOKENS,
        "payload_manifest_sha256": _canonical_digest(files),
    }
    if proof != expected_proof:
        raise ValueError("recovery promotion proof differs")
    unavailable = value.get("unavailable_ephemeral_worker_evidence")
    if not isinstance(unavailable, dict) or any(
        unavailable.get(key) is not None
        for key in (
            "original_capture_run_uuid",
            "raw_router_return_sha256_and_statistics",
            "reference_route_diagnostics",
        )
    ):
        raise ValueError("recovery receipt fabricates unavailable worker evidence")
    runtime = value.get("runtime_binding")
    if not isinstance(runtime, dict) or runtime.get(
        "post_full_jit_inventory_reused_as_preflight"
    ) is not False:
        raise ValueError("recovery runtime binding differs")
    preflight = validate_full_capture_preflight_record(
        runtime.get("full_capture_preflight"), plan=plan, capture_code=archived
    )
    if (
        runtime.get("smoke_evidence_sha256")
        != preflight["smoke"]["evidence_sha256"]
        or runtime.get("smoke_run_token") != preflight["smoke"]["run_token"]
    ):
        raise ValueError("recovery smoke/preflight binding differs")
    smoke_runtime = runtime.get("smoke_runtime")
    if (
        not isinstance(smoke_runtime, dict)
        or smoke_runtime.get("capture_code") != archived
        or smoke_runtime.get("full_capture_preflight") is not None
    ):
        raise ValueError("recovery archived smoke runtime differs")
    audits = runtime.get("router_audits")
    if not isinstance(audits, dict) or set(audits) != {
        str(layer) for layer in SELECTED_LAYERS
    }:
        raise ValueError("recovery router-audit binding differs")
    for layer in SELECTED_LAYERS:
        validate_router_audit(audits[str(layer)], layer=layer)
    if value.get("document_plan") != {
        "sha256": _canonical_digest(plan),
        "fingerprint": plan["plan_fingerprint"],
    }:
        raise ValueError("recovery receipt document plan differs")
    return dict(value)


def recover_finalized_capture(
    capture_dir: str | Path,
    *,
    smoke_dir: str | Path,
    jit_cache_dir: str | Path,
    project_root: str | Path,
) -> dict:
    """Seal final-promoted payloads without model/GPU access."""

    root = Path(capture_dir).resolve()
    smoke_root = Path(smoke_dir).resolve()
    cache = Path(jit_cache_dir).resolve()
    project = Path(project_root).resolve()
    _require_payload_layout(root, with_manifests=False)
    plan_path = root / "document_plan.json"
    plan = load_document_plan(plan_path)
    smoke, archived_code, preflight, stamp_sha = _archived_preflight(
        jit_cache=cache,
        smoke_dir=smoke_root,
        plan_path=plan_path,
        project_root=project,
    )
    files, statistics = _payload_records(root, plan)
    code = recovery_code_evidence(project)
    receipt = _receipt(
        plan=plan,
        files=files,
        smoke=smoke,
        archived_code=archived_code,
        preflight=preflight,
        binding_stamp_sha256=stamp_sha,
        recovery_code=code,
    )
    recovery_uuid = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"glm52-fresh-sqg:{receipt['receipt_sha256']}")
    )
    receipt_sha = receipt["receipt_sha256"]
    layer_manifests: list[tuple[Path, dict]] = []
    audits = receipt["runtime_binding"]["router_audits"]
    for layer in SELECTED_LAYERS:
        stats = statistics[str(layer)]
        manifest = {
            "schema": RECOVERED_LAYER_SCHEMA,
            "evidence_mode": RECOVERY_EVIDENCE_MODE,
            "capture_run_uuid": recovery_uuid,
            "original_capture_run_uuid": None,
            "recovery_receipt_sha256": receipt_sha,
            "document_plan_fingerprint": plan["plan_fingerprint"],
            "layer": layer,
            "tokens": OWNER_TOKENS,
            "documents_verified": OWNER_DOCUMENTS,
            "document_audit_sha256": expected_document_audit_sha256(plan),
            "hidden": HIDDEN,
            "topk": TOPK,
            "num_experts": NUM_EXPERTS,
            "routing_source": ROUTING_SOURCE,
            "topk_weights_semantics": TOPK_WEIGHT_SEMANTICS,
            "routed_scaling_factor": ROUTED_SCALING_FACTOR,
            "role_rows": {
                str(key): value
                for key, value in sorted(expected_role_rows(plan).items())
            },
            "routed_counts": stats["routed_counts"],
            "gate_sum": stats["gate_sum"],
            "gate_sq_sum": stats["gate_sq_sum"],
            "gate_row_sum_min": stats["gate_row_sum_min"],
            "gate_row_sum_max": stats["gate_row_sum_max"],
            "hidden_validation": stats["hidden_validation"],
            "router_audit_binding": {
                "source": "bound successful DCP4 smoke",
                "audit": audits[str(layer)],
            },
            "route_evidence": _unavailable_route_evidence(
                layer, audits[str(layer)]
            ),
            "files": files[str(layer)],
        }
        layer_manifests.append(
            (layer_dir(root, layer) / "layer_manifest.json", manifest)
        )
    for path, manifest in layer_manifests:
        atomic_json(path, manifest)
    root_manifest = {
        "schema": RECOVERED_CAPTURE_SCHEMA,
        "complete": True,
        "evidence_mode": RECOVERY_EVIDENCE_MODE,
        "capture_run_uuid": recovery_uuid,
        "original_capture_run_uuid": None,
        "document_plan": {
            "path": "document_plan.json",
            "sha256": sha256_file(plan_path),
            "fingerprint": plan["plan_fingerprint"],
        },
        "selected_layers": list(SELECTED_LAYERS),
        "tokens_per_layer": OWNER_TOKENS,
        "documents": OWNER_DOCUMENTS,
        "split": EXPECTED_SPLIT,
        "file_abi": FILE_ABI,
        "layer_manifests": {
            str(layer): {
                "path": f"layer_{layer:03d}/layer_manifest.json",
                "sha256": sha256_file(layer_dir(root, layer) / "layer_manifest.json"),
            }
            for layer in SELECTED_LAYERS
        },
        "construction_exclusions": CONSTRUCTION_EXCLUSIONS,
        "recovery_receipt": receipt,
    }
    atomic_json(root / "capture_manifest.json", root_manifest)
    return validate_recovered_capture(root, verify_hashes=False, project_root=project)


def validate_recovered_capture(
    capture_dir: str | Path,
    *,
    verify_hashes: bool = True,
    project_root: str | Path | None = None,
) -> dict:
    root = Path(capture_dir)
    _require_payload_layout(root, with_manifests=True)
    manifest = _read_json(root / "capture_manifest.json")
    if (
        manifest.get("schema") != RECOVERED_CAPTURE_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("evidence_mode") != RECOVERY_EVIDENCE_MODE
        or manifest.get("original_capture_run_uuid") is not None
        or manifest.get("selected_layers") != list(SELECTED_LAYERS)
        or manifest.get("tokens_per_layer") != OWNER_TOKENS
        or manifest.get("documents") != OWNER_DOCUMENTS
        or manifest.get("split") != EXPECTED_SPLIT
        or manifest.get("file_abi") != FILE_ABI
        or manifest.get("construction_exclusions") != CONSTRUCTION_EXCLUSIONS
    ):
        raise ValueError("recovered capture root contract differs")
    try:
        uuid.UUID(str(manifest.get("capture_run_uuid")))
    except ValueError as exc:
        raise ValueError("recovered capture UUID is invalid") from exc
    plan_path = root / "document_plan.json"
    plan = load_document_plan(plan_path)
    if manifest.get("document_plan") != {
        "path": "document_plan.json",
        "sha256": sha256_file(plan_path),
        "fingerprint": plan["plan_fingerprint"],
    }:
        raise ValueError("recovered capture document plan differs")
    files: dict[str, dict] = {}
    layers: dict[int, dict] = {}
    for layer in SELECTED_LAYERS:
        path = layer_dir(root, layer) / "layer_manifest.json"
        item = _read_json(path)
        layers[layer] = item
        files[str(layer)] = item.get("files")
    project = (
        Path(__file__).resolve().parents[1]
        if project_root is None
        else Path(project_root).resolve()
    )
    receipt = _validate_receipt(
        manifest.get("recovery_receipt"),
        plan=plan,
        files=files,
        project_root=project,
    )
    receipt_sha = receipt["receipt_sha256"]
    expected_audit = expected_document_audit_sha256(plan)
    audits = receipt["runtime_binding"]["router_audits"]
    for layer, item in layers.items():
        path = layer_dir(root, layer) / "layer_manifest.json"
        reference = manifest.get("layer_manifests", {}).get(str(layer), {})
        if reference != {
            "path": f"layer_{layer:03d}/layer_manifest.json",
            "sha256": sha256_file(path),
        }:
            raise ValueError(f"layer {layer}: recovered root manifest binding differs")
        expected_scalars = {
            "schema": RECOVERED_LAYER_SCHEMA,
            "evidence_mode": RECOVERY_EVIDENCE_MODE,
            "capture_run_uuid": manifest["capture_run_uuid"],
            "original_capture_run_uuid": None,
            "recovery_receipt_sha256": receipt_sha,
            "document_plan_fingerprint": plan["plan_fingerprint"],
            "layer": layer,
            "tokens": OWNER_TOKENS,
            "documents_verified": OWNER_DOCUMENTS,
            "document_audit_sha256": expected_audit,
            "hidden": HIDDEN,
            "topk": TOPK,
            "num_experts": NUM_EXPERTS,
            "routing_source": ROUTING_SOURCE,
            "topk_weights_semantics": TOPK_WEIGHT_SEMANTICS,
            "routed_scaling_factor": ROUTED_SCALING_FACTOR,
            "role_rows": {
                str(key): value
                for key, value in sorted(expected_role_rows(plan).items())
            },
        }
        for key, expected in expected_scalars.items():
            if item.get(key) != expected:
                raise ValueError(f"layer {layer}: recovered {key} differs")
        audit_binding = item.get("router_audit_binding")
        if audit_binding != {
            "source": "bound successful DCP4 smoke",
            "audit": audits[str(layer)],
        }:
            raise ValueError(f"layer {layer}: recovered router audit differs")
        validate_router_audit(audit_binding["audit"], layer=layer)
        _validate_unavailable_route_evidence(
            item.get("route_evidence"), layer=layer, router_audit=audit_binding["audit"]
        )
        directory = layer_dir(root, layer)
        for name, abi in FILE_ABI.items():
            payload = directory / name
            identity = _identity(payload)
            expected_bytes = OWNER_TOKENS * int(abi["bytes_per_row"])
            record = item.get("files", {}).get(name, {})
            expected_shape = (
                [OWNER_TOKENS, HIDDEN]
                if name.startswith("hidden")
                else [OWNER_TOKENS, TOPK]
                if name.startswith("topk")
                else [OWNER_TOKENS]
            )
            if (
                identity[2] != expected_bytes
                or record.get("bytes") != expected_bytes
                or record.get("dtype") != abi["dtype"]
                or record.get("shape") != expected_shape
            ):
                raise ValueError(f"layer {layer}: recovered {name} ABI differs")
            if verify_hashes and record.get("sha256") != _hash_file(payload):
                raise ValueError(f"layer {layer}: recovered {name} hash differs")
        _validate_document_vectors(directory, plan)
        routed, gate_stats = _validate_routes(directory)
        if routed != item.get("routed_counts"):
            raise ValueError(f"layer {layer}: recovered routed counts differ")
        for key, value in gate_stats.items():
            if not math.isclose(
                value, float(item.get(key, math.nan)), rel_tol=1e-11, abs_tol=1e-6
            ):
                raise ValueError(f"layer {layer}: recovered {key} differs")
        if verify_hashes:
            hidden_sha, hidden_validation = _hidden_record(
                directory / "hidden.bf16.bin"
            )
            if (
                hidden_sha != item["files"]["hidden.bf16.bin"]["sha256"]
                or hidden_validation != item.get("hidden_validation")
                or hidden_validation["nonfinite_words"] != 0
            ):
                raise ValueError(f"layer {layer}: recovered hidden validation differs")
    return manifest
