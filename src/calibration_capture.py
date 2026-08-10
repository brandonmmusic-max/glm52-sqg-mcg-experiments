"""On-disk ABI and fail-closed validation for GLM-5.2 calibration capture."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Iterator
import uuid

import numpy as np

from .calibration_plan import (
    EXPECTED_SPLIT,
    OWNER_DOCUMENTS,
    OWNER_TOKENS,
    ROLE_TO_ID,
    atomic_json,
    canonical_json_bytes,
    load_document_plan,
    sha256_file,
    validate_document_plan,
)
from .capture_runtime import (
    RUNTIME_CLASS_SOURCES,
    RUNTIME_ENV,
    RUNTIME_FILES,
    RUNTIME_IMAGE_ID,
    RUNTIME_IMAGE_REFERENCE,
    RUNTIME_PYTHON,
)


CAPTURE_SCHEMA = "glm52-fresh-sqg-calibration-capture-v1"
LAYER_SCHEMA = "glm52-fresh-sqg-layer-capture-v1"
SELECTED_LAYERS = (6, 28, 52, 77)
ROUTED_LAYERS = tuple(range(3, 78))
# Exact all-MCG teacher load order under the hash-sealed r33 EXL3 source and
# its historical 48-layer fused budget.  Layer 9 has a non-two-tier payload;
# after 6/7/8 the next 45 eligible layers are 10..54.
EXPECTED_FUSED_LAYERS = (6, 7, 8, *range(10, 55))
HIDDEN = 6_144
NUM_EXPERTS = 256
TOPK = 8
ROUTED_SCALING_FACTOR = 2.5
ATTENTION_BACKEND = "B12X_MLA_SPARSE"
KV_CACHE_DTYPE = "fp8"
EFFECTIVE_KV_CACHE_DTYPE = "fp8_ds_mla"
INDEX_TOPK_PATTERN = (
    "FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"
    "FSSSFSSSFSSSFSSS"
)
REFERENCE_ROUTE_RTOL = 2e-5
REFERENCE_ROUTE_ATOL = 2e-6
REFERENCE_ROUTE_SELECTION_ATOL = 2e-6
RAW_EFFECTIVE_GATE_AGGREGATE_RTOL = 1e-11
# Each effective gate is rounded independently by the float32 multiply before
# both raw and effective payloads are accumulated in float64.  Their aggregate
# sums therefore close only to the accumulated float32 product-rounding error.
RAW_EFFECTIVE_GATE_AGGREGATE_ATOL = 5e-4
REFERENCE_ROUTE_SCHEMA = "glm52-fused-live-route-admissibility-v2"
REFERENCE_ROUTE_DESCRIPTION = (
    "independent torch.sigmoid(actual router_logits); preserve the exact live "
    "top8 IDs only when max-unselected minus min-selected biased score is at "
    "most 2e-6; gather independent unbiased scores on those live IDs; "
    "normalize; apply sole audited router-scale times runner-output-scale "
    "product 2.5"
)

FILE_ABI = {
    "hidden.bf16.bin": {"bytes_per_row": HIDDEN * 2, "dtype": "bfloat16-le"},
    "topk_ids.u8.bin": {"bytes_per_row": TOPK, "dtype": "uint8"},
    "topk_weights.f32le.bin": {
        "bytes_per_row": TOPK * 4,
        "dtype": "float32-le",
    },
    "doc_epochs.u32le.bin": {"bytes_per_row": 4, "dtype": "uint32-le"},
    "token_positions.u16le.bin": {"bytes_per_row": 2, "dtype": "uint16-le"},
    "role_ids.u8.bin": {"bytes_per_row": 1, "dtype": "uint8"},
}
CONSTRUCTION_EXCLUSIONS = {
    "mcg_payload_bytes": 0,
    "mcg_transform_vectors": 0,
    "mcg_scale_vectors": 0,
    "mcg_permutations": 0,
    "mcg_encoder_seeds": 0,
}
ROUTING_SOURCE = (
    "exact IDs from live router.select_experts after original call; "
    "effective weights are its exact float32 return multiplied only by "
    "the audited live runner output scale"
)
TOPK_WEIGHT_SEMANTICS = "effective applied float32 expert-output multipliers"
CAPTURE_CODE_SCHEMA = "glm52-fresh-sqg-capture-code-manifest-v1"
CAPTURE_CODE_FILES = (
    "run_capture_container.sh",
    "capture_calibration.py",
    "seal_teacher_model.py",
    "src/__init__.py",
    "src/calibration_plan.py",
    "src/calibration_capture.py",
    "src/capture_runtime.py",
    "src/glm52_capture_worker.py",
    "src/teacher_identity.py",
)
SMOKE_SCHEMA = "glm52-fresh-sqg-dcp4-row-ownership-smoke-v1"
JIT_CACHE_BINDING_SCHEMA = "glm52-fresh-sqg-jit-cache-binding-v1"
JIT_CACHE_BINDING_FILE = ".fresh_sqg_capture_cache_binding.json"
FULL_CAPTURE_PREFLIGHT_SCHEMA = "glm52-fresh-sqg-full-capture-preflight-v1"
KV_CACHE_INTERLEAVE_RESOLUTION = (
    "vLLM VllmConfig.validate_block_size: under DCP, legacy "
    "dcp_kv_cache_interleave_size > 1 overrides a differing deprecated "
    "cp_kv_cache_interleave_size"
)
TEACHER_IDENTITY_SEAL_SHA256 = (
    "e256a8c5c6e3734e47da61ecb7649bee596091178981ded0bbc00b69c50b766a"
)
TEACHER_IDENTITY_RECEIPT_SHA256 = (
    "eb88fcd2bf66b0cfef195a232d9b81efcb107c400381b22c913a2c738413479e"
)
TEACHER_IDENTITY_VALIDATION = {
    "schema": "glm52-r33-teacher-identity-validation-v1",
    "receipt_sha256": TEACHER_IDENTITY_RECEIPT_SHA256,
    "seal_sha256": TEACHER_IDENTITY_SEAL_SHA256,
    "verification_mode": "full",
    "all_file_bytes_sha256_validated": True,
    "non_payload_file_bytes_sha256_validated": True,
    "payload_count": 156,
    "loader_sidecar_count": 75,
    "loader_identity_count": 8,
    "total_file_count": 239,
    "payload_bytes": 342_683_459_652,
    "total_bytes": 343_070_719_678,
    "index_weight_count": 187_580,
    "index_declared_tensor_bytes": 342_661_251_548,
}


def effective_kv_cache_interleave_size(
    *,
    decode_context_parallel_size: int,
    dcp_kv_cache_interleave_size_raw: int,
    cp_kv_cache_interleave_size_raw: int,
) -> int:
    """Resolve the effective CP interleave using the pinned vLLM semantics."""

    values = (
        decode_context_parallel_size,
        dcp_kv_cache_interleave_size_raw,
        cp_kv_cache_interleave_size_raw,
    )
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("KV-cache interleave inputs must be positive integers")
    if (
        decode_context_parallel_size > 1
        and dcp_kv_cache_interleave_size_raw > 1
        and cp_kv_cache_interleave_size_raw
        != dcp_kv_cache_interleave_size_raw
    ):
        return dcp_kv_cache_interleave_size_raw
    return cp_kv_cache_interleave_size_raw


def validate_teacher_identity_evidence(value: object) -> dict:
    """Require the exact full-byte validation record for the frozen teacher."""

    if value != TEACHER_IDENTITY_VALIDATION:
        raise ValueError("full teacher-model identity validation evidence differs")
    return dict(value)


def capture_code_evidence(project_root: str | Path) -> dict:
    """Hash the exact local command/driver/worker allowlist at runtime."""

    root = Path(project_root).resolve()
    files: dict[str, dict[str, object]] = {}
    for relative in CAPTURE_CODE_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"capture code allowlist entry is absent/unsafe: {relative}")
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest_sha256 = hashlib.sha256(canonical_json_bytes(files)).hexdigest()
    return {
        "schema": CAPTURE_CODE_SCHEMA,
        "allowlist": list(CAPTURE_CODE_FILES),
        "files": files,
        "manifest_sha256": manifest_sha256,
    }


def validate_capture_code_evidence(
    value: object,
    *,
    project_root: str | Path | None = None,
) -> dict:
    """Validate form and, when available, exact current executable bytes."""

    if not isinstance(value, dict):
        raise ValueError("capture code manifest is absent")
    if (
        set(value) != {"schema", "allowlist", "files", "manifest_sha256"}
        or value.get("schema") != CAPTURE_CODE_SCHEMA
        or value.get("allowlist") != list(CAPTURE_CODE_FILES)
        or not isinstance(value.get("files"), dict)
        or set(value["files"]) != set(CAPTURE_CODE_FILES)
    ):
        raise ValueError("capture code manifest schema/allowlist differs")
    for relative in CAPTURE_CODE_FILES:
        record = value["files"].get(relative)
        if (
            not isinstance(record, dict)
            or set(record) != {"bytes", "sha256"}
            or type(record.get("bytes")) is not int
            or int(record["bytes"]) <= 0
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in record["sha256"])
        ):
            raise ValueError(f"capture code manifest entry differs: {relative}")
    observed = hashlib.sha256(canonical_json_bytes(value["files"])).hexdigest()
    if value.get("manifest_sha256") != observed:
        raise ValueError("capture code manifest digest differs")
    if project_root is not None and value != capture_code_evidence(project_root):
        raise ValueError("capture code manifest differs from current executable bytes")
    return dict(value)


def validate_fused_layer_audit(value: object) -> dict:
    """Validate the actual post-load all-MCG teacher dispatch population."""

    if not isinstance(value, dict):
        raise ValueError("post-load EXL3 fused-layer audit is absent")
    expected_nonfused = sorted(set(ROUTED_LAYERS) - set(EXPECTED_FUSED_LAYERS))
    expected = {
        "schema": "glm52-r33-all-mcg-teacher-fused-layer-audit-v1",
        "routed_layers": list(ROUTED_LAYERS),
        "fused_layers": list(EXPECTED_FUSED_LAYERS),
        "nonfused_layers": expected_nonfused,
        "fused_count": 48,
        "configured_budget": 48,
        "observed_budget_counters": [48],
        "reserved_layers": [],
        "selected_layer_modes": {
            "6": "fused",
            "28": "fused",
            "52": "fused",
            "77": "nonfused",
        },
        "teacher_codebook": "mcg",
    }
    if value != expected:
        raise ValueError(f"post-load EXL3 fused-layer audit differs: {value}")
    return dict(value)


def layer_dir(capture_dir: str | Path, layer: int) -> Path:
    if int(layer) not in SELECTED_LAYERS:
        raise ValueError(f"layer {layer} is outside the frozen pilot")
    return Path(capture_dir) / f"layer_{int(layer):03d}"


def document_audit_bytes(document: dict, observed_rows: int) -> bytes:
    payload = {
        "document_sha256": document["document_sha256"],
        "epoch": int(document["epoch"]),
        "observed_rows": int(observed_rows),
        "role_id": int(document["role_id"]),
        "tokens": int(document["tokens"]),
    }
    return canonical_json_bytes(payload) + b"\n"


def expected_document_audit_sha256(plan: dict) -> str:
    validate_document_plan(plan)
    digest = hashlib.sha256()
    for document in plan["documents"]:
        digest.update(document_audit_bytes(document, int(document["tokens"])))
    return digest.hexdigest()


def expected_role_rows(plan: dict) -> dict[int, int]:
    validate_document_plan(plan)
    return {
        ROLE_TO_ID[role]: int(values["tokens"])
        for role, values in EXPECTED_SPLIT.items()
    }


def validate_reference_route_check(value: object, *, rows: int) -> dict:
    """Return canonical evidence for the all-row independent routing gate."""

    if not isinstance(value, dict):
        raise ValueError("independent route-check evidence is absent")
    expected_scalars = {
        "schema": REFERENCE_ROUTE_SCHEMA,
        "checked_rows": int(rows),
        "selection_violation_rows": 0,
        "weight_mismatch_rows": 0,
        "selection_atol": REFERENCE_ROUTE_SELECTION_ATOL,
        "weight_rtol": REFERENCE_ROUTE_RTOL,
        "weight_atol": REFERENCE_ROUTE_ATOL,
        "coverage": "all captured rows",
        "reference": REFERENCE_ROUTE_DESCRIPTION,
    }
    for key, expected in expected_scalars.items():
        observed = value.get(key)
        if isinstance(expected, float):
            if not math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=0.0
            ):
                raise ValueError(f"independent route check {key} differs")
        elif observed != expected:
            raise ValueError(f"independent route check {key} differs")
    max_abs = float(value.get("max_abs_weight_error", math.nan))
    # Every normalized nonnegative applied gate is at most the row sum 2.5.
    largest_permitted = REFERENCE_ROUTE_ATOL + (
        REFERENCE_ROUTE_RTOL * ROUTED_SCALING_FACTOR
    )
    if (
        not math.isfinite(max_abs)
        or max_abs < 0.0
        or max_abs > largest_permitted
    ):
        raise ValueError("independent route check maximum error is invalid")
    ambiguous_rows = value.get("boundary_ambiguous_rows")
    if (
        not isinstance(ambiguous_rows, int)
        or isinstance(ambiguous_rows, bool)
        or ambiguous_rows < 0
        or ambiguous_rows > int(rows)
    ):
        raise ValueError(
            "independent route check boundary_ambiguous_rows is invalid"
        )
    max_selection_violation = float(
        value.get("max_selection_violation", math.nan)
    )
    if (
        not math.isfinite(max_selection_violation)
        or max_selection_violation < 0.0
        or max_selection_violation > REFERENCE_ROUTE_SELECTION_ATOL
    ):
        raise ValueError(
            "independent route check maximum selection violation is invalid"
        )
    if ambiguous_rows == 0 and max_selection_violation != 0.0:
        raise ValueError(
            "independent route check selection evidence is inconsistent"
        )
    return {
        **expected_scalars,
        "boundary_ambiguous_rows": ambiguous_rows,
        "max_selection_violation": max_selection_violation,
        "max_abs_weight_error": max_abs,
    }


def validate_raw_route_weight_evidence(value: object, *, applied: dict) -> dict:
    """Validate the hash/statistics of the exact unscaled router return."""

    if not isinstance(value, dict):
        raise ValueError("raw router-return weight evidence is absent")
    raw_sha = value.get("sha256")
    if (
        not isinstance(raw_sha, str)
        or len(raw_sha) != 64
        or any(character not in "0123456789abcdef" for character in raw_sha)
    ):
        raise ValueError("raw router-return weight SHA256 is invalid")
    expected_bytes = OWNER_TOKENS * TOPK * 4
    if (
        value.get("source")
        != "exact float32 weights returned by original live router.select_experts"
        or value.get("dtype") != "float32-le"
        or value.get("shape") != [OWNER_TOKENS, TOPK]
        or int(value.get("bytes", -1)) != expected_bytes
        or value.get("payload_stored") is not False
        or value.get("effective_payload") != "topk_weights.f32le.bin"
    ):
        raise ValueError("raw router-return weight ABI evidence differs")
    router_scale = float(value.get("router_return_scale", math.nan))
    runner_scale = float(value.get("runner_output_scale", math.nan))
    product = float(value.get("effective_scale_product", math.nan))
    if (
        not math.isfinite(router_scale)
        or not math.isfinite(runner_scale)
        or router_scale <= 0.0
        or runner_scale <= 0.0
        or not math.isclose(
            router_scale * runner_scale,
            ROUTED_SCALING_FACTOR,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            product, ROUTED_SCALING_FACTOR, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError("router/runner scaling placement is not a sole 2.5 product")
    raw_sum = float(value.get("sum", math.nan))
    raw_sq_sum = float(value.get("sq_sum", math.nan))
    raw_row_min = float(value.get("row_sum_min", math.nan))
    raw_row_max = float(value.get("row_sum_max", math.nan))
    for number in (raw_sum, raw_sq_sum, raw_row_min, raw_row_max):
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError("raw router-return weight statistics are invalid")
    if not math.isclose(
        raw_row_min, router_scale, rel_tol=2e-6, abs_tol=2e-6
    ) or not math.isclose(
        raw_row_max, router_scale, rel_tol=2e-6, abs_tol=2e-6
    ):
        raise ValueError("raw router-return rows do not close to router scale")
    if not math.isclose(
        raw_sum * runner_scale,
        float(applied["gate_sum"]),
        rel_tol=RAW_EFFECTIVE_GATE_AGGREGATE_RTOL,
        abs_tol=RAW_EFFECTIVE_GATE_AGGREGATE_ATOL,
    ) or not math.isclose(
        raw_sq_sum * runner_scale * runner_scale,
        float(applied["gate_sq_sum"]),
        rel_tol=RAW_EFFECTIVE_GATE_AGGREGATE_RTOL,
        abs_tol=RAW_EFFECTIVE_GATE_AGGREGATE_ATOL,
    ):
        raise ValueError("raw/effective gate statistics do not close under runner scale")
    return {
        "source": value["source"],
        "dtype": "float32-le",
        "shape": [OWNER_TOKENS, TOPK],
        "bytes": expected_bytes,
        "sha256": raw_sha,
        "payload_stored": False,
        "effective_payload": "topk_weights.f32le.bin",
        "router_return_scale": router_scale,
        "runner_output_scale": runner_scale,
        "effective_scale_product": product,
        "sum": raw_sum,
        "sq_sum": raw_sq_sum,
        "row_sum_min": raw_row_min,
        "row_sum_max": raw_row_max,
    }


def validate_router_audit(value: object, *, layer: int) -> dict:
    """Validate the live modular-router topology and sole scale placement."""

    if not isinstance(value, dict):
        raise ValueError(f"layer {layer}: live router audit is absent")
    exact = {
        "layer": int(layer),
        "module_class": "MoERunner",
        "router_class": "GroupedTopKRouter",
        "quant_method_class": "Exl3MoEMethod",
        "owner_class": "DeepseekV2MoE",
        "class_sources": RUNTIME_CLASS_SOURCES,
        "quant_method_is_monolithic": False,
        "router_top_k": TOPK,
        "router_global_num_experts": NUM_EXPERTS,
        "router_renormalize": True,
        "router_scoring_func": "sigmoid",
        "router_num_expert_group": 1,
        "router_topk_group": 1,
        "router_num_fused_shared_experts": 0,
        "router_eplb_enabled": False,
        "runner_enable_dbo": False,
        "runner_has_input_transform": False,
        "runner_has_output_transform": False,
        "owner_rocm_aiter_moe_enabled": False,
        "moe_use_ep": False,
        "moe_sequence_parallel": False,
        "apply_router_weight_on_input": False,
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise ValueError(f"layer {layer}: router audit {key} differs")
    for key in (
        "module_name",
    ):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"layer {layer}: router audit {key} is absent")
    if not value["module_name"].endswith(f"layers.{layer}.mlp.experts"):
        raise ValueError(f"layer {layer}: audited another MoE module")
    router_scale = float(value.get("router_routed_scaling_factor", math.nan))
    container_scale = float(
        value.get("expert_container_routed_scaling_factor", math.nan)
    )
    runner_scale = float(value.get("runner_output_scale", math.nan))
    owner_scale = float(
        value.get("owner_configured_routed_scaling_factor", math.nan)
    )
    module_scale = float(value.get("module_output_scale", math.nan))
    effective = float(value.get("effective_routed_scaling_factor", math.nan))
    if (
        not math.isfinite(router_scale)
        or not math.isfinite(container_scale)
        or not math.isfinite(runner_scale)
        or not math.isfinite(owner_scale)
        or not math.isfinite(module_scale)
        or router_scale <= 0.0
        or runner_scale <= 0.0
        or not math.isclose(container_scale, router_scale, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(owner_scale, runner_scale, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(module_scale, runner_scale, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(
            router_scale * runner_scale,
            ROUTED_SCALING_FACTOR,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            effective, ROUTED_SCALING_FACTOR, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError(f"layer {layer}: router audit scaling placement differs")
    return dict(value)


def validate_runtime_provenance_evidence(value: object) -> dict:
    """Validate the exact serving image declaration and mounted source hashes."""

    if not isinstance(value, dict):
        raise ValueError("capture runtime provenance is absent")
    exact = {
        "image_reference": RUNTIME_IMAGE_REFERENCE,
        "image_id_declared": RUNTIME_IMAGE_ID,
        "image_identity_observation": (
            "host docker inspect declaration; independently bound below by exact "
            "mounted execution-file hashes"
        ),
        "python_executable": RUNTIME_PYTHON,
        "environment": RUNTIME_ENV,
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise ValueError(f"capture runtime provenance {key} differs")
    files = value.get("files")
    if not isinstance(files, dict) or set(files) != set(RUNTIME_FILES):
        raise ValueError("capture runtime provenance file set differs")
    for filename, expected_sha256 in RUNTIME_FILES.items():
        evidence = files.get(filename)
        if (
            not isinstance(evidence, dict)
            or evidence.get("sha256") != expected_sha256
            or not isinstance(evidence.get("bytes"), int)
            or int(evidence["bytes"]) <= 0
        ):
            raise ValueError(f"capture runtime provenance differs for {filename}")
    return dict(value)


def write_layer_manifest(
    capture_dir: str | Path,
    layer: int,
    plan: dict,
    worker_result: dict,
    *,
    run_uuid: str,
) -> dict:
    """Validate a worker result against disk and write its immutable manifest."""

    validate_document_plan(plan)
    layer = int(layer)
    directory = layer_dir(capture_dir, layer)
    if int(worker_result.get("tokens", -1)) != OWNER_TOKENS:
        raise ValueError(f"layer {layer}: worker token count differs")
    if int(worker_result.get("documents_verified", -1)) != OWNER_DOCUMENTS:
        raise ValueError(f"layer {layer}: worker document count differs")
    expected_audit = expected_document_audit_sha256(plan)
    if worker_result.get("document_audit_sha256") != expected_audit:
        raise ValueError(f"layer {layer}: one-document/request audit does not close")
    role_rows = {int(key): int(value) for key, value in worker_result["role_rows"].items()}
    if role_rows != expected_role_rows(plan):
        raise ValueError(f"layer {layer}: captured split token totals differ")
    routed = [int(value) for value in worker_result["routed_counts"]]
    if len(routed) != NUM_EXPERTS or sum(routed) != OWNER_TOKENS * TOPK:
        raise ValueError(f"layer {layer}: routed expert counts do not close")

    reference_route_check = validate_reference_route_check(
        worker_result.get("reference_route_check"), rows=OWNER_TOKENS
    )
    applied_stats = {
        key: float(worker_result[key])
        for key in (
            "gate_sum",
            "gate_sq_sum",
            "gate_row_sum_min",
            "gate_row_sum_max",
        )
    }
    raw_route_weights = validate_raw_route_weight_evidence(
        worker_result.get("raw_router_return_weights"), applied=applied_stats
    )
    router_audit = validate_router_audit(
        worker_result.get("router_audit"), layer=layer
    )
    if (
        raw_route_weights["router_return_scale"]
        != float(router_audit["router_routed_scaling_factor"])
        or raw_route_weights["runner_output_scale"]
        != float(router_audit["runner_output_scale"])
    ):
        raise ValueError(f"layer {layer}: raw gate evidence and router audit differ")

    files: dict[str, dict] = {}
    for name, abi in FILE_ABI.items():
        path = directory / name
        expected_bytes = OWNER_TOKENS * int(abi["bytes_per_row"])
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError(f"layer {layer}: {name} size does not close")
        reported = worker_result.get("files", {}).get(name)
        if not isinstance(reported, dict):
            raise ValueError(f"layer {layer}: worker omitted {name} evidence")
        digest = sha256_file(path)
        if reported.get("sha256") != digest or int(reported.get("bytes", -1)) != expected_bytes:
            raise ValueError(f"layer {layer}: {name} worker digest/size differs")
        files[name] = {
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

    manifest = {
        "schema": LAYER_SCHEMA,
        "capture_run_uuid": run_uuid,
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
        "role_rows": {str(key): value for key, value in sorted(role_rows.items())},
        "routed_counts": routed,
        **applied_stats,
        "router_audit": router_audit,
        "raw_router_return_weights": raw_route_weights,
        "reference_route_check": reference_route_check,
        "files": files,
    }
    atomic_json(directory / "layer_manifest.json", manifest)
    return manifest


def write_capture_manifest(
    capture_dir: str | Path,
    plan_path: str | Path,
    plan: dict,
    layer_manifests: list[dict],
    *,
    run_uuid: str,
    runtime: dict,
) -> dict:
    """Write the sole completion sentinel after every layer has sealed."""

    validate_document_plan(plan)
    if sorted(int(item["layer"]) for item in layer_manifests) != list(SELECTED_LAYERS):
        raise ValueError("cannot seal capture without all four frozen pilot layers")
    root = Path(capture_dir)
    plan_copy = root / "document_plan.json"
    if Path(plan_path).resolve() != plan_copy.resolve():
        atomic_json(plan_copy, plan)
    elif not plan_copy.is_file():
        raise ValueError("capture-local document plan is absent")
    if load_document_plan(plan_copy) != plan:
        raise ValueError("capture-local document plan differs")
    manifests = {}
    for item in layer_manifests:
        layer = int(item["layer"])
        path = layer_dir(root, layer) / "layer_manifest.json"
        manifests[str(layer)] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
    value = {
        "schema": CAPTURE_SCHEMA,
        "complete": True,
        "capture_run_uuid": run_uuid,
        "document_plan": {
            "path": "document_plan.json",
            "sha256": sha256_file(plan_copy),
            "fingerprint": plan["plan_fingerprint"],
        },
        "selected_layers": list(SELECTED_LAYERS),
        "tokens_per_layer": OWNER_TOKENS,
        "documents": OWNER_DOCUMENTS,
        "split": EXPECTED_SPLIT,
        "file_abi": FILE_ABI,
        "runtime": runtime,
        "layer_manifests": manifests,
        "construction_exclusions": CONSTRUCTION_EXCLUSIONS,
    }
    atomic_json(root / "capture_manifest.json", value)
    return value


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def validate_capture_runtime_record(
    value: object,
    *,
    project_root: str | Path | None = None,
    require_full_preflight: bool = False,
    archived_capture_code: dict | None = None,
) -> dict:
    """Validate the complete fixed TP4/DCP4 runtime and code provenance."""

    if not isinstance(value, dict):
        raise ValueError("capture runtime evidence is absent")
    expected = {
        "engine_seed": 0,
        "tensor_parallel_size": 4,
        "decode_context_parallel_size": 4,
        "dcp_comm_backend": "a2a",
        "dcp_kv_cache_interleave_size_raw": 64,
        "cp_kv_cache_interleave_size_raw": 1,
        "effective_kv_cache_interleave_size": 64,
        "kv_cache_interleave_resolution": KV_CACHE_INTERLEAVE_RESOLUTION,
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "max_num_seqs": 1,
        "max_model_len": 4_352,
        "max_num_batched_tokens": 2_048,
        "gpu_memory_utilization": 0.90,
        "kv_cache_memory_bytes": 268_435_456,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "expert_parallel": False,
        "moe_backend": "b12x",
        "sequence_parallel_moe": False,
        "speculative_decoding": False,
        "max_tokens": 1,
        "model_dtype": "bfloat16",
        "quantization": "exl3",
        "load_format": "safetensors",
        "attention_backend": ATTENTION_BACKEND,
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "effective_kv_cache_dtype": EFFECTIVE_KV_CACHE_DTYPE,
        "kv_fp8_rope": False,
        "hf_overrides": {
            "use_index_cache": True,
            "index_topk_pattern": INDEX_TOPK_PATTERN,
        },
        "disable_custom_all_reduce": True,
        "async_scheduling": False,
        "trust_remote_code": True,
        "dcp_row_ownership_gate": (
            "rank zero must observe exactly every document token at every selected "
            "MoE layer under endpoint DCP4, accumulated across scheduler chunks"
        ),
        "endpoint_deviations": {
            "max_model_len": {
                "capture": 4_352,
                "saved_kld": 2_560,
                "reason": "owner corpus permits intact documents through 4096 tokens",
            },
            "all_other_scheduler_cache_controls_match_saved_kld": True,
        },
        "llm_extra": {},
    }
    variable_keys = {
        "runtime_provenance",
        "worker_runtime_audit",
        "worker_hf_contract",
        "all_rank_fused_layer_audits",
        "capture_code",
        "full_capture_preflight",
        "teacher_identity",
        "vllm_version",
    }
    if set(value) != set(expected) | variable_keys:
        raise ValueError("capture runtime field set differs")
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"capture runtime {key} differs")

    worker_expected = {
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "decode_context_parallel_size": 4,
        "dcp_comm_backend": "a2a",
        "dcp_kv_cache_interleave_size_raw": 64,
        "cp_kv_cache_interleave_size_raw": 1,
        "effective_kv_cache_interleave_size": 64,
        "kv_cache_interleave_resolution": KV_CACHE_INTERLEAVE_RESOLUTION,
        "enable_expert_parallel": False,
        "enable_eplb": False,
        "moe_backend": "b12x",
        "use_sequence_parallel_moe": False,
        "enable_dbo": False,
        "disable_custom_all_reduce": True,
        "max_num_seqs": 1,
        "max_model_len": 4_352,
        "max_num_batched_tokens": 2_048,
        "gpu_memory_utilization": 0.90,
        "kv_cache_memory_bytes": 268_435_456,
        "async_scheduling": False,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "speculative_config": False,
        "enforce_eager": True,
        "quantization": "exl3",
        "load_format": "safetensors",
        "attention_backend": ATTENTION_BACKEND,
        "kv_cache_dtype": EFFECTIVE_KV_CACHE_DTYPE,
        "python_executable": RUNTIME_PYTHON,
        "runtime_image_id_declared": RUNTIME_IMAGE_ID,
        "exl3_environment": RUNTIME_ENV,
    }
    if value.get("worker_runtime_audit") != worker_expected:
        raise ValueError("worker runtime audit differs")
    for label, record in (
        ("capture runtime", value),
        ("worker runtime audit", value["worker_runtime_audit"]),
    ):
        effective = effective_kv_cache_interleave_size(
            decode_context_parallel_size=record[
                "decode_context_parallel_size"
            ],
            dcp_kv_cache_interleave_size_raw=record[
                "dcp_kv_cache_interleave_size_raw"
            ],
            cp_kv_cache_interleave_size_raw=record[
                "cp_kv_cache_interleave_size_raw"
            ],
        )
        if effective != record["effective_kv_cache_interleave_size"]:
            raise ValueError(f"{label} effective KV-cache interleave differs")
    validate_runtime_provenance_evidence(value.get("runtime_provenance"))

    worker_hf = value.get("worker_hf_contract")
    if worker_hf != {
        "hidden_size": HIDDEN,
        "moe_intermediate_size": 2_048,
        "n_routed_experts": NUM_EXPERTS,
        "num_experts_per_tok": TOPK,
        "routed_scaling_factor": ROUTED_SCALING_FACTOR,
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "hidden_act": "silu",
        "num_hidden_layers": 78,
        "use_index_cache": True,
        "index_topk_pattern": INDEX_TOPK_PATTERN,
    }:
        raise ValueError("worker GLM architecture audit differs")

    all_rank_fused = value.get("all_rank_fused_layer_audits")
    if not isinstance(all_rank_fused, dict) or set(all_rank_fused) != {
        "0",
        "1",
        "2",
        "3",
    }:
        raise ValueError("post-load fused-layer audit does not cover TP ranks 0..3")
    for rank in ("0", "1", "2", "3"):
        try:
            validate_fused_layer_audit(all_rank_fused[rank])
        except ValueError as exc:
            raise ValueError(f"TP rank {rank}: {exc}") from exc

    version = value.get("vllm_version")
    if not isinstance(version, str) or not version or version == "unknown":
        raise ValueError("vLLM package version evidence is absent")
    if archived_capture_code is None:
        root = (
            Path(__file__).resolve().parents[1]
            if project_root is None
            else Path(project_root).resolve()
        )
        validate_capture_code_evidence(value.get("capture_code"), project_root=root)
    else:
        validate_capture_code_evidence(archived_capture_code)
        if value.get("capture_code") != archived_capture_code:
            raise ValueError("capture runtime archived code manifest differs")
    validate_teacher_identity_evidence(value.get("teacher_identity"))
    preflight = value.get("full_capture_preflight")
    if require_full_preflight:
        validate_full_capture_preflight_record(
            preflight,
            capture_code=value["capture_code"],
        )
    elif preflight is not None:
        raise ValueError("DCP4 smoke runtime must not claim a full-capture preflight")
    return dict(value)


def validate_smoke_run_token(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("DCP4 smoke run token is absent")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError("DCP4 smoke run token is invalid") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("DCP4 smoke run token is not a canonical UUID4")
    return value


def validate_dcp4_smoke_evidence(
    smoke_dir: str | Path,
    *,
    plan_path: str | Path,
    project_root: str | Path,
    archived_capture_code: dict | None = None,
) -> dict:
    """Strictly bind a one-document smoke to current code, plan, and disk."""

    root = Path(smoke_dir).resolve()
    project = Path(project_root).resolve()
    if not root.is_dir():
        raise ValueError("DCP4 smoke directory is absent")
    evidence_path = root / "dcp4_smoke_evidence.json"
    if not evidence_path.is_file() or evidence_path.is_symlink():
        raise ValueError("DCP4 smoke evidence is absent/unsafe")
    value = _read_json(evidence_path)
    expected_fields = {
        "schema",
        "passed",
        "completion_sentinel_created",
        "smoke_run_token",
        "plan_fingerprint",
        "document",
        "selected_layers",
        "expected_rows_per_layer",
        "chunked_prefill_proof",
        "status",
        "abort",
        "runtime",
        "router_audits",
    }
    if set(value) != expected_fields:
        raise ValueError("DCP4 smoke evidence field set differs")
    if (
        value.get("schema") != SMOKE_SCHEMA
        or value.get("passed") is not True
        or value.get("completion_sentinel_created") is not False
        or value.get("selected_layers") != list(SELECTED_LAYERS)
    ):
        raise ValueError("DCP4 smoke root contract differs")
    validate_smoke_run_token(value.get("smoke_run_token"))

    plan = load_document_plan(plan_path)
    first_document = plan["documents"][0]
    rows = int(first_document["tokens"])
    if (
        value.get("plan_fingerprint") != plan["plan_fingerprint"]
        or value.get("document") != first_document
        or type(value.get("expected_rows_per_layer")) is not int
        or int(value["expected_rows_per_layer"]) != rows
    ):
        raise ValueError("DCP4 smoke plan/first-document binding differs")
    local_plan = root / "document_plan.json"
    if (
        not local_plan.is_file()
        or local_plan.is_symlink()
        or load_document_plan(local_plan) != plan
        or sha256_file(local_plan) != sha256_file(plan_path)
    ):
        raise ValueError("DCP4 smoke local document plan differs")

    counts = {str(layer): rows for layer in SELECTED_LAYERS}
    if value.get("chunked_prefill_proof") != {
        "document_tokens": rows,
        "max_num_batched_tokens": 2_048,
        "requires_multiple_scheduler_chunks": True,
        "selected_layer_counts": counts,
    } or rows <= 2_048:
        raise ValueError("DCP4 smoke does not prove cross-chunk row accumulation")
    if value.get("status") != {
        "rank": 0,
        "capturing": True,
        "active_epoch": None,
        "documents_verified": 1,
        "counts": counts,
    }:
        raise ValueError("DCP4 smoke rank-zero status differs")
    if value.get("abort") != {
        "rank": 0,
        "capturing": True,
        "partial_payloads_retained": True,
        "documents_verified": 1,
    }:
        raise ValueError("DCP4 smoke verified-abort result differs")

    validate_capture_runtime_record(
        value.get("runtime"),
        project_root=project,
        archived_capture_code=archived_capture_code,
    )
    router_audits = value.get("router_audits")
    if not isinstance(router_audits, dict) or set(router_audits) != {
        str(layer) for layer in SELECTED_LAYERS
    }:
        raise ValueError("DCP4 smoke router-audit layer set differs")
    for layer in SELECTED_LAYERS:
        validate_router_audit(router_audits[str(layer)], layer=layer)

    expected_root_entries = {
        "dcp4_smoke_evidence.json",
        "document_plan.json",
        *(f"layer_{layer:03d}" for layer in SELECTED_LAYERS),
    }
    if {path.name for path in root.iterdir()} != expected_root_entries:
        raise ValueError("DCP4 smoke directory contains unexpected/missing entries")
    for layer in SELECTED_LAYERS:
        directory = root / f"layer_{layer:03d}"
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"DCP4 smoke layer directory is absent/unsafe: {layer}")
        expected_names = {f"{name}.partial" for name in FILE_ABI}
        children = {path.name: path for path in directory.iterdir()}
        if set(children) != expected_names:
            raise ValueError(f"DCP4 smoke layer {layer} partial-file set differs")
        for name, abi in FILE_ABI.items():
            partial = children[f"{name}.partial"]
            expected_bytes = rows * int(abi["bytes_per_row"])
            if (
                not partial.is_file()
                or partial.is_symlink()
                or partial.stat().st_size != expected_bytes
            ):
                raise ValueError(
                    f"DCP4 smoke layer {layer} {name} partial size differs"
                )
    return value


def _jit_cache_binding_value(
    jit_cache: Path,
    smoke_dir: Path,
    *,
    plan_path: Path,
    project_root: Path,
    smoke: dict,
) -> dict:
    plan = load_document_plan(plan_path)
    stat = jit_cache.stat()
    cache_inventory = _jit_cache_inventory(jit_cache)
    value: dict[str, object] = {
        "schema": JIT_CACHE_BINDING_SCHEMA,
        "complete": True,
        "jit_cache": {
            "path": str(jit_cache),
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
        },
        "smoke": {
            "directory": str(smoke_dir),
            "evidence_sha256": sha256_file(
                smoke_dir / "dcp4_smoke_evidence.json"
            ),
            "run_token": smoke["smoke_run_token"],
        },
        "plan": {
            "sha256": sha256_file(plan_path),
            "fingerprint": plan["plan_fingerprint"],
            "first_document": plan["documents"][0],
        },
        "runtime": {
            "image_reference": RUNTIME_IMAGE_REFERENCE,
            "image_id": RUNTIME_IMAGE_ID,
            "capture_code": capture_code_evidence(project_root),
        },
        "cache_inventory": cache_inventory,
    }
    value["binding_id"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value


def _jit_cache_tree(jit_cache: Path) -> Iterator[tuple[Path, str, os.stat_result]]:
    """Walk every cache entry explicitly and fail closed on traversal errors."""

    root = Path(jit_cache)
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise ValueError(f"cannot stat dedicated JIT cache root: {root}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("dedicated JIT cache root is not a safe directory")

    pending: list[tuple[Path, str]] = [(root, "")]
    while pending:
        directory, prefix = pending.pop()
        try:
            with os.scandir(directory) as scanner:
                entries = sorted(scanner, key=lambda entry: entry.name)
        except OSError as exc:
            label = prefix or "."
            raise ValueError(
                f"cannot enumerate dedicated JIT cache directory: {label}"
            ) from exc

        subdirectories: list[tuple[Path, str]] = []
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ValueError(
                    f"cannot stat dedicated JIT cache entry: {relative}"
                ) from exc
            yield Path(entry.path), relative, entry_stat
            if stat.S_ISDIR(entry_stat.st_mode):
                subdirectories.append((Path(entry.path), relative))
        pending.extend(reversed(subdirectories))


def _jit_cache_inventory(jit_cache: Path) -> dict:
    """Hash the complete post-smoke cache tree, excluding only its stamp."""

    directories: list[str] = []
    files: dict[str, dict[str, object]] = {}
    for path, relative, entry_stat in _jit_cache_tree(jit_cache):
        if relative == JIT_CACHE_BINDING_FILE:
            continue
        mode = entry_stat.st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"dedicated JIT cache contains a symlink: {relative}")
        if stat.S_ISDIR(mode):
            directories.append(relative)
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"dedicated JIT cache contains a special file: {relative}")
        try:
            digest = sha256_file(path)
        except OSError as exc:
            raise ValueError(
                f"cannot read dedicated JIT cache file: {relative}"
            ) from exc
        files[relative] = {
            "bytes": entry_stat.st_size,
            "sha256": digest,
        }
    body = {"directories": directories, "files": files}
    return {
        **body,
        "inventory_sha256": hashlib.sha256(
            canonical_json_bytes(body)
        ).hexdigest(),
    }


def normalize_jit_cache_permissions(jit_cache: str | Path) -> dict:
    """Make a root-generated cache host-readable without accepting unsafe nodes.

    This is intended for the sealed, networkless post-smoke helper container.
    It only adds owner read/write and world read/search bits; existing execute
    bits on compiled artifacts are preserved.
    """

    root = Path(jit_cache)
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise ValueError(f"cannot stat dedicated JIT cache root: {root}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("dedicated JIT cache root is not a safe directory")
    try:
        root.chmod(stat.S_IMODE(root_stat.st_mode) | 0o755)
    except OSError as exc:
        raise ValueError("cannot normalize dedicated JIT cache root") from exc

    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as scanner:
                entries = sorted(scanner, key=lambda entry: entry.name)
        except OSError as exc:
            raise ValueError(
                f"cannot enumerate dedicated JIT cache directory: {directory}"
            ) from exc
        subdirectories: list[Path] = []
        for entry in entries:
            path = Path(entry.path)
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ValueError(
                    f"cannot stat dedicated JIT cache entry: {path}"
                ) from exc
            mode = entry_stat.st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"dedicated JIT cache contains a symlink: {path}")
            if stat.S_ISDIR(mode):
                try:
                    path.chmod(stat.S_IMODE(mode) | 0o755)
                except OSError as exc:
                    raise ValueError(
                        f"cannot normalize dedicated JIT cache directory: {path}"
                    ) from exc
                subdirectories.append(path)
            elif stat.S_ISREG(mode):
                try:
                    path.chmod(stat.S_IMODE(mode) | 0o644)
                except OSError as exc:
                    raise ValueError(
                        f"cannot normalize dedicated JIT cache file: {path}"
                    ) from exc
            else:
                raise ValueError(
                    f"dedicated JIT cache contains a special file: {path}"
                )
        pending.extend(reversed(subdirectories))
    return _jit_cache_inventory(root)


def write_jit_cache_binding(
    jit_cache: str | Path,
    smoke_dir: str | Path,
    *,
    plan_path: str | Path,
    project_root: str | Path,
    expected_inventory_sha256: str | None = None,
) -> dict:
    """Stamp the exact successful smoke/cache association after container exit."""

    cache = Path(jit_cache).resolve()
    smoke_root = Path(smoke_dir).resolve()
    plan = Path(plan_path).resolve()
    project = Path(project_root).resolve()
    if not cache.is_dir():
        raise ValueError("dedicated JIT cache directory is absent")
    stamp = cache / JIT_CACHE_BINDING_FILE
    if stamp.exists():
        raise ValueError("dedicated JIT cache is already stamped")
    smoke = validate_dcp4_smoke_evidence(
        smoke_root, plan_path=plan, project_root=project
    )
    value = _jit_cache_binding_value(
        cache,
        smoke_root,
        plan_path=plan,
        project_root=project,
        smoke=smoke,
    )
    if (
        expected_inventory_sha256 is not None
        and value["cache_inventory"]["inventory_sha256"]
        != expected_inventory_sha256
    ):
        raise ValueError("host/container dedicated JIT cache inventories differ")
    atomic_json(stamp, value)
    return value


def validate_jit_cache_binding(
    jit_cache: str | Path,
    smoke_dir: str | Path,
    *,
    plan_path: str | Path,
    project_root: str | Path,
) -> dict:
    """Require full capture to reuse the exact smoke-populated cache inode."""

    cache = Path(jit_cache).resolve()
    smoke_root = Path(smoke_dir).resolve()
    plan = Path(plan_path).resolve()
    project = Path(project_root).resolve()
    smoke = validate_dcp4_smoke_evidence(
        smoke_root, plan_path=plan, project_root=project
    )
    stamp = cache / JIT_CACHE_BINDING_FILE
    if not stamp.is_file() or stamp.is_symlink():
        raise ValueError("dedicated JIT cache binding stamp is absent/unsafe")
    observed = _read_json(stamp)
    expected = _jit_cache_binding_value(
        cache,
        smoke_root,
        plan_path=plan,
        project_root=project,
        smoke=smoke,
    )
    if observed != expected:
        raise ValueError("dedicated JIT cache binding differs")
    return observed


def _validated_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a canonical SHA256")
    return value


def validate_full_capture_preflight_record(
    value: object,
    *,
    plan: dict | None = None,
    capture_code: dict | None = None,
) -> dict:
    """Validate the portable receipt sealed into a completed full capture."""

    if not isinstance(value, dict):
        raise ValueError("full-capture preflight receipt is absent")
    expected_fields = {
        "schema",
        "smoke",
        "jit_cache",
        "plan",
        "runtime",
        "receipt_sha256",
    }
    if set(value) != expected_fields or value.get("schema") != FULL_CAPTURE_PREFLIGHT_SCHEMA:
        raise ValueError("full-capture preflight receipt field set/schema differs")
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value.get("receipt_sha256") != hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest():
        raise ValueError("full-capture preflight receipt digest differs")

    smoke = value.get("smoke")
    if not isinstance(smoke, dict) or set(smoke) != {
        "evidence_sha256",
        "run_token",
    }:
        raise ValueError("full-capture smoke receipt differs")
    _validated_sha256(smoke.get("evidence_sha256"), label="smoke evidence SHA256")
    validate_smoke_run_token(smoke.get("run_token"))

    cache = value.get("jit_cache")
    if not isinstance(cache, dict) or set(cache) != {
        "binding_id",
        "binding_stamp_sha256",
        "inventory_sha256",
        "device",
        "inode",
    }:
        raise ValueError("full-capture JIT-cache receipt differs")
    for key in ("binding_id", "binding_stamp_sha256", "inventory_sha256"):
        _validated_sha256(cache.get(key), label=f"JIT cache {key}")
    for key in ("device", "inode"):
        if type(cache.get(key)) is not int or int(cache[key]) < 0:
            raise ValueError(f"full-capture JIT-cache {key} differs")

    plan_receipt = value.get("plan")
    if not isinstance(plan_receipt, dict) or set(plan_receipt) != {
        "sha256",
        "fingerprint",
    }:
        raise ValueError("full-capture plan receipt differs")
    _validated_sha256(plan_receipt.get("sha256"), label="preflight plan SHA256")
    _validated_sha256(
        plan_receipt.get("fingerprint"), label="preflight plan fingerprint"
    )
    if plan is not None and plan_receipt.get("fingerprint") != plan[
        "plan_fingerprint"
    ]:
        raise ValueError("full-capture preflight binds another document plan")

    runtime = value.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "image_reference",
        "image_id",
        "capture_code_manifest_sha256",
    }:
        raise ValueError("full-capture preflight runtime receipt differs")
    if (
        runtime.get("image_reference") != RUNTIME_IMAGE_REFERENCE
        or runtime.get("image_id") != RUNTIME_IMAGE_ID
    ):
        raise ValueError("full-capture preflight runtime image differs")
    _validated_sha256(
        runtime.get("capture_code_manifest_sha256"),
        label="preflight capture-code manifest SHA256",
    )
    if capture_code is not None and runtime.get(
        "capture_code_manifest_sha256"
    ) != capture_code.get("manifest_sha256"):
        raise ValueError("full-capture preflight capture-code seal differs")
    return dict(value)


def validate_full_capture_preflight(
    jit_cache: str | Path,
    smoke_dir: str | Path,
    *,
    plan_path: str | Path,
    project_root: str | Path,
    expected_smoke_token: object,
) -> dict:
    """Validate mounted smoke/cache bytes and issue the portable full receipt."""

    cache = Path(jit_cache).resolve()
    smoke_root = Path(smoke_dir).resolve()
    plan_file = Path(plan_path).resolve()
    project = Path(project_root).resolve()
    token = validate_smoke_run_token(expected_smoke_token)
    smoke = validate_dcp4_smoke_evidence(
        smoke_root,
        plan_path=plan_file,
        project_root=project,
    )
    if smoke["smoke_run_token"] != token:
        raise ValueError("full-capture smoke token environment differs from evidence")

    stamp_path = cache / JIT_CACHE_BINDING_FILE
    if not stamp_path.is_file() or stamp_path.is_symlink():
        raise ValueError("dedicated JIT cache binding stamp is absent/unsafe")
    stamp = _read_json(stamp_path)
    stamp_fields = {
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
        set(stamp) != stamp_fields
        or stamp.get("schema") != JIT_CACHE_BINDING_SCHEMA
        or stamp.get("complete") is not True
    ):
        raise ValueError("dedicated JIT cache binding schema/field set differs")
    binding_body = {key: item for key, item in stamp.items() if key != "binding_id"}
    binding_id = hashlib.sha256(canonical_json_bytes(binding_body)).hexdigest()
    if stamp.get("binding_id") != binding_id:
        raise ValueError("dedicated JIT cache binding ID differs")

    cache_stat = cache.stat()
    jit_identity = stamp.get("jit_cache")
    if (
        not isinstance(jit_identity, dict)
        or set(jit_identity) != {"path", "device", "inode"}
        or not isinstance(jit_identity.get("path"), str)
        or not Path(jit_identity["path"]).is_absolute()
        or jit_identity.get("device") != int(cache_stat.st_dev)
        or jit_identity.get("inode") != int(cache_stat.st_ino)
    ):
        raise ValueError("dedicated JIT cache mounted identity differs")
    evidence_sha256 = sha256_file(smoke_root / "dcp4_smoke_evidence.json")
    smoke_binding = stamp.get("smoke")
    if (
        not isinstance(smoke_binding, dict)
        or set(smoke_binding) != {"directory", "evidence_sha256", "run_token"}
        or not isinstance(smoke_binding.get("directory"), str)
        or not Path(smoke_binding["directory"]).is_absolute()
        or smoke_binding.get("evidence_sha256") != evidence_sha256
        or smoke_binding.get("run_token") != token
    ):
        raise ValueError("dedicated JIT cache smoke binding differs")

    plan = load_document_plan(plan_file)
    if stamp.get("plan") != {
        "sha256": sha256_file(plan_file),
        "fingerprint": plan["plan_fingerprint"],
        "first_document": plan["documents"][0],
    }:
        raise ValueError("dedicated JIT cache plan binding differs")
    code = capture_code_evidence(project)
    if stamp.get("runtime") != {
        "image_reference": RUNTIME_IMAGE_REFERENCE,
        "image_id": RUNTIME_IMAGE_ID,
        "capture_code": code,
    }:
        raise ValueError("dedicated JIT cache runtime/code binding differs")
    inventory = _jit_cache_inventory(cache)
    if stamp.get("cache_inventory") != inventory:
        raise ValueError("dedicated JIT cache inventory differs")

    receipt = {
        "schema": FULL_CAPTURE_PREFLIGHT_SCHEMA,
        "smoke": {
            "evidence_sha256": evidence_sha256,
            "run_token": token,
        },
        "jit_cache": {
            "binding_id": binding_id,
            "binding_stamp_sha256": sha256_file(stamp_path),
            "inventory_sha256": inventory["inventory_sha256"],
            "device": int(cache_stat.st_dev),
            "inode": int(cache_stat.st_ino),
        },
        "plan": {
            "sha256": sha256_file(plan_file),
            "fingerprint": plan["plan_fingerprint"],
        },
        "runtime": {
            "image_reference": RUNTIME_IMAGE_REFERENCE,
            "image_id": RUNTIME_IMAGE_ID,
            "capture_code_manifest_sha256": code["manifest_sha256"],
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    return validate_full_capture_preflight_record(
        receipt,
        plan=plan,
        capture_code=code,
    )


def _chunks(total: int, rows: int = 65_536) -> Iterator[tuple[int, int]]:
    for begin in range(0, total, rows):
        yield begin, min(total, begin + rows)


def _validate_document_vectors(directory: Path, plan: dict) -> None:
    epochs = np.memmap(directory / "doc_epochs.u32le.bin", mode="r", dtype="<u4")
    positions = np.memmap(
        directory / "token_positions.u16le.bin", mode="r", dtype="<u2"
    )
    roles = np.memmap(directory / "role_ids.u8.bin", mode="r", dtype="u1")
    offset = 0
    for document in plan["documents"]:
        count = int(document["tokens"])
        end = offset + count
        if not np.all(epochs[offset:end] == int(document["epoch"])):
            raise ValueError(f"{directory}: epoch vector differs at epoch {document['epoch']}")
        if not np.array_equal(
            positions[offset:end], np.arange(count, dtype=np.dtype("<u2"))
        ):
            raise ValueError(
                f"{directory}: token positions differ at epoch {document['epoch']}"
            )
        if not np.all(roles[offset:end] == int(document["role_id"])):
            raise ValueError(f"{directory}: role vector differs at epoch {document['epoch']}")
        offset = end
    if offset != OWNER_TOKENS:
        raise ValueError(f"{directory}: document vectors do not consume every row")


def _validate_routes(directory: Path) -> tuple[list[int], dict[str, float]]:
    ids = np.memmap(
        directory / "topk_ids.u8.bin", mode="r", dtype="u1", shape=(OWNER_TOKENS, TOPK)
    )
    weights = np.memmap(
        directory / "topk_weights.f32le.bin",
        mode="r",
        dtype="<f4",
        shape=(OWNER_TOKENS, TOPK),
    )
    routed = np.zeros(NUM_EXPERTS, dtype=np.int64)
    gate_sum = 0.0
    gate_sq_sum = 0.0
    row_min = math.inf
    row_max = -math.inf
    for begin, end in _chunks(OWNER_TOKENS):
        local_ids = np.asarray(ids[begin:end])
        local_weights = np.asarray(weights[begin:end])
        if np.any(local_ids >= NUM_EXPERTS):
            raise ValueError(f"{directory}: routed expert ID is outside [0,255]")
        if np.any(np.sort(local_ids, axis=1)[:, 1:] == np.sort(local_ids, axis=1)[:, :-1]):
            raise ValueError(f"{directory}: a token routes to one expert twice")
        if not np.isfinite(local_weights).all() or np.any(local_weights < 0):
            raise ValueError(f"{directory}: routed weights are invalid")
        row_sums = local_weights.astype(np.float64).sum(axis=1)
        if not np.allclose(row_sums, ROUTED_SCALING_FACTOR, rtol=2e-6, atol=2e-6):
            raise ValueError(f"{directory}: routed weights are not applied GLM gates")
        routed += np.bincount(local_ids.reshape(-1), minlength=NUM_EXPERTS)
        local64 = local_weights.astype(np.float64)
        gate_sum += float(local64.sum())
        gate_sq_sum += float(np.square(local64).sum())
        row_min = min(row_min, float(row_sums.min()))
        row_max = max(row_max, float(row_sums.max()))
    return routed.tolist(), {
        "gate_sum": gate_sum,
        "gate_sq_sum": gate_sq_sum,
        "gate_row_sum_min": row_min,
        "gate_row_sum_max": row_max,
    }


def validate_capture(capture_dir: str | Path, *, verify_hashes: bool = True) -> dict:
    """Validate schema, closure, document boundaries, routing, and optional hashes."""

    root = Path(capture_dir)
    manifest = _read_json(root / "capture_manifest.json")
    # Recovery is an explicit, honest alternate ABI.  Import lazily to avoid a
    # module cycle while retaining the normal v1 path byte-for-byte below.
    from .calibration_recovery import (
        RECOVERED_CAPTURE_SCHEMA,
        validate_recovered_capture,
    )

    if manifest.get("schema") == RECOVERED_CAPTURE_SCHEMA:
        return validate_recovered_capture(root, verify_hashes=verify_hashes)
    if manifest.get("schema") != CAPTURE_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("capture is absent, incomplete, or uses another schema")
    if (
        manifest.get("selected_layers") != list(SELECTED_LAYERS)
        or int(manifest.get("tokens_per_layer", -1)) != OWNER_TOKENS
        or int(manifest.get("documents", -1)) != OWNER_DOCUMENTS
        or manifest.get("split") != EXPECTED_SPLIT
        or manifest.get("file_abi") != FILE_ABI
        or manifest.get("construction_exclusions") != CONSTRUCTION_EXCLUSIONS
    ):
        raise ValueError("capture root contract differs")
    runtime = validate_capture_runtime_record(
        manifest.get("runtime"),
        require_full_preflight=True,
    )
    if set(manifest.get("layer_manifests", {})) != {
        str(layer) for layer in SELECTED_LAYERS
    }:
        raise ValueError("capture root layer-manifest set differs")
    plan_path = root / str(manifest["document_plan"]["path"])
    if manifest["document_plan"].get("path") != "document_plan.json":
        raise ValueError("capture document-plan path differs")
    plan = load_document_plan(plan_path)
    if (
        manifest["document_plan"].get("fingerprint") != plan["plan_fingerprint"]
        or (verify_hashes and manifest["document_plan"].get("sha256") != sha256_file(plan_path))
    ):
        raise ValueError("capture document plan binding differs")
    preflight = validate_full_capture_preflight_record(
        runtime["full_capture_preflight"],
        plan=plan,
        capture_code=runtime["capture_code"],
    )
    if preflight["plan"]["sha256"] != manifest["document_plan"].get("sha256"):
        raise ValueError("capture preflight document-plan SHA256 differs")

    expected_audit = expected_document_audit_sha256(plan)
    for layer in SELECTED_LAYERS:
        directory = layer_dir(root, layer)
        path = directory / "layer_manifest.json"
        layer_manifest = _read_json(path)
        reference = manifest.get("layer_manifests", {}).get(str(layer), {})
        expected_relative = f"layer_{layer:03d}/layer_manifest.json"
        if reference.get("path") != expected_relative:
            raise ValueError(f"layer {layer}: root manifest path differs")
        if verify_hashes and reference.get("sha256") != sha256_file(path):
            raise ValueError(f"layer {layer}: layer-manifest hash differs")
        if (
            layer_manifest.get("schema") != LAYER_SCHEMA
            or int(layer_manifest.get("layer", -1)) != layer
            or int(layer_manifest.get("tokens", -1)) != OWNER_TOKENS
            or int(layer_manifest.get("documents_verified", -1)) != OWNER_DOCUMENTS
            or layer_manifest.get("document_audit_sha256") != expected_audit
            or layer_manifest.get("document_plan_fingerprint") != plan["plan_fingerprint"]
            or layer_manifest.get("capture_run_uuid")
            != manifest.get("capture_run_uuid")
            or int(layer_manifest.get("hidden", -1)) != HIDDEN
            or int(layer_manifest.get("topk", -1)) != TOPK
            or int(layer_manifest.get("num_experts", -1)) != NUM_EXPERTS
            or float(layer_manifest.get("routed_scaling_factor", math.nan))
            != ROUTED_SCALING_FACTOR
            or layer_manifest.get("routing_source") != ROUTING_SOURCE
            or layer_manifest.get("topk_weights_semantics")
            != TOPK_WEIGHT_SEMANTICS
            or layer_manifest.get("role_rows")
            != {
                str(key): value
                for key, value in sorted(expected_role_rows(plan).items())
            }
        ):
            raise ValueError(f"layer {layer}: layer manifest contract differs")
        for name, abi in FILE_ABI.items():
            payload = directory / name
            expected_bytes = OWNER_TOKENS * int(abi["bytes_per_row"])
            file_info = layer_manifest.get("files", {}).get(name, {})
            if not payload.is_file() or payload.stat().st_size != expected_bytes:
                raise ValueError(f"layer {layer}: {name} size differs")
            expected_shape = (
                [OWNER_TOKENS, HIDDEN]
                if name.startswith("hidden")
                else [OWNER_TOKENS, TOPK]
                if name.startswith("topk")
                else [OWNER_TOKENS]
            )
            if (
                int(file_info.get("bytes", -1)) != expected_bytes
                or file_info.get("dtype") != abi["dtype"]
                or file_info.get("shape") != expected_shape
            ):
                raise ValueError(f"layer {layer}: {name} ABI manifest differs")
            if verify_hashes and file_info.get("sha256") != sha256_file(payload):
                raise ValueError(f"layer {layer}: {name} hash differs")
        _validate_document_vectors(directory, plan)
        routed, gate_stats = _validate_routes(directory)
        if routed != layer_manifest.get("routed_counts"):
            raise ValueError(f"layer {layer}: routed-count manifest differs")
        for key, value in gate_stats.items():
            if not math.isclose(
                value, float(layer_manifest.get(key, math.nan)), rel_tol=1e-11, abs_tol=1e-6
            ):
                raise ValueError(f"layer {layer}: {key} manifest differs")
        validate_reference_route_check(
            layer_manifest.get("reference_route_check"), rows=OWNER_TOKENS
        )
        raw = validate_raw_route_weight_evidence(
            layer_manifest.get("raw_router_return_weights"), applied=gate_stats
        )
        audit = validate_router_audit(layer_manifest.get("router_audit"), layer=layer)
        if (
            raw["router_return_scale"]
            != float(audit["router_routed_scaling_factor"])
            or raw["runner_output_scale"] != float(audit["runner_output_scale"])
        ):
            raise ValueError(f"layer {layer}: raw gate/router audit binding differs")
    return manifest
