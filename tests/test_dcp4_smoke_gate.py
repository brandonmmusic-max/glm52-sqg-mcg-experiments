from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import stat
import subprocess
import uuid

import pytest

import capture_calibration as driver
import src.calibration_capture as capture_contract
from src.calibration_capture import (
    EXPECTED_FUSED_LAYERS,
    INDEX_TOPK_PATTERN,
    JIT_CACHE_BINDING_FILE,
    KV_CACHE_INTERLEAVE_RESOLUTION,
    ROUTED_LAYERS,
    SELECTED_LAYERS,
    TEACHER_IDENTITY_VALIDATION,
    atomic_json,
    effective_kv_cache_interleave_size,
    normalize_jit_cache_permissions,
    validate_dcp4_smoke_evidence,
    validate_full_capture_preflight,
    validate_full_capture_preflight_record,
    validate_jit_cache_binding,
    write_jit_cache_binding,
)
from src.capture_runtime import (
    RUNTIME_CLASS_SOURCES,
    RUNTIME_ENV,
    RUNTIME_FILES,
    RUNTIME_IMAGE_ID,
    RUNTIME_IMAGE_REFERENCE,
    RUNTIME_PYTHON,
)


PROJECT = Path(__file__).resolve().parents[1]
PLAN = PROJECT / "evidence" / "document_plan.json"


def _fused_audit() -> dict:
    fused = list(EXPECTED_FUSED_LAYERS)
    return {
        "schema": "glm52-r33-all-mcg-teacher-fused-layer-audit-v1",
        "routed_layers": list(ROUTED_LAYERS),
        "fused_layers": fused,
        "nonfused_layers": sorted(set(ROUTED_LAYERS) - set(fused)),
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


def _worker_runtime() -> dict:
    return {
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
        "attention_backend": "B12X_MLA_SPARSE",
        "kv_cache_dtype": "fp8_ds_mla",
        "python_executable": RUNTIME_PYTHON,
        "runtime_image_id_declared": RUNTIME_IMAGE_ID,
        "exl3_environment": RUNTIME_ENV,
    }


def _hf_contract() -> dict:
    return {
        "hidden_size": 6_144,
        "moe_intermediate_size": 2_048,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "hidden_act": "silu",
        "num_hidden_layers": 78,
        "use_index_cache": True,
        "index_topk_pattern": INDEX_TOPK_PATTERN,
    }


def _runtime_provenance() -> dict:
    return {
        "image_reference": RUNTIME_IMAGE_REFERENCE,
        "image_id_declared": RUNTIME_IMAGE_ID,
        "image_identity_observation": (
            "host docker inspect declaration; independently bound below by exact "
            "mounted execution-file hashes"
        ),
        "python_executable": RUNTIME_PYTHON,
        "environment": RUNTIME_ENV,
        "files": {
            name: {"sha256": digest, "bytes": 1}
            for name, digest in RUNTIME_FILES.items()
        },
    }


def _router_audit(layer: int) -> dict:
    return {
        "layer": layer,
        "module_name": f"model.layers.{layer}.mlp.experts",
        "module_class": "MoERunner",
        "router_class": "GroupedTopKRouter",
        "quant_method_class": "Exl3MoEMethod",
        "owner_class": "DeepseekV2MoE",
        "class_sources": RUNTIME_CLASS_SOURCES,
        "quant_method_is_monolithic": False,
        "router_top_k": 8,
        "router_global_num_experts": 256,
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
        "router_routed_scaling_factor": 1.0,
        "expert_container_routed_scaling_factor": 1.0,
        "runner_output_scale": 2.5,
        "owner_configured_routed_scaling_factor": 2.5,
        "module_output_scale": 2.5,
        "effective_routed_scaling_factor": 2.5,
    }


@pytest.fixture
def strict_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    tiny_abi = {
        name: {**record, "bytes_per_row": 1}
        for name, record in capture_contract.FILE_ABI.items()
    }
    monkeypatch.setattr(capture_contract, "FILE_ABI", tiny_abi)
    plan = capture_contract.load_document_plan(PLAN)
    document = plan["documents"][0]
    rows = int(document["tokens"])
    root = tmp_path / "smoke"
    root.mkdir()
    shutil.copyfile(PLAN, root / "document_plan.json")
    for layer in SELECTED_LAYERS:
        directory = root / f"layer_{layer:03d}"
        directory.mkdir()
        for name in tiny_abi:
            (directory / f"{name}.partial").write_bytes(b"\0" * rows)

    rank_zero = {"runtime": _worker_runtime(), "hf_contract": _hf_contract()}
    audits = {str(rank): _fused_audit() for rank in range(4)}
    runtime = driver._runtime_record(
        extra={},
        runtime_provenance=_runtime_provenance(),
        rank_zero=rank_zero,
        all_rank_fused_audits=audits,
        teacher_identity=TEACHER_IDENTITY_VALIDATION,
    )
    runtime["vllm_version"] = "test-vllm"
    counts = {str(layer): rows for layer in SELECTED_LAYERS}
    evidence = {
        "schema": capture_contract.SMOKE_SCHEMA,
        "passed": True,
        "completion_sentinel_created": False,
        "smoke_run_token": str(uuid.uuid4()),
        "plan_fingerprint": plan["plan_fingerprint"],
        "document": document,
        "selected_layers": list(SELECTED_LAYERS),
        "expected_rows_per_layer": rows,
        "chunked_prefill_proof": {
            "document_tokens": rows,
            "max_num_batched_tokens": 2_048,
            "requires_multiple_scheduler_chunks": True,
            "selected_layer_counts": counts,
        },
        "status": {
            "rank": 0,
            "capturing": True,
            "active_epoch": None,
            "documents_verified": 1,
            "counts": counts,
        },
        "abort": {
            "rank": 0,
            "capturing": True,
            "partial_payloads_retained": True,
            "documents_verified": 1,
        },
        "runtime": runtime,
        "router_audits": {
            str(layer): _router_audit(layer) for layer in SELECTED_LAYERS
        },
    }
    atomic_json(root / "dcp4_smoke_evidence.json", evidence)
    return {"root": root, "evidence": evidence, "cache": tmp_path / "jit"}


def _rewrite_evidence(root: Path, value: dict) -> None:
    path = root / "dcp4_smoke_evidence.json"
    path.unlink()
    atomic_json(path, value)


def test_strict_smoke_closes_current_plan_runtime_router_code_and_partials(
    strict_smoke: dict,
) -> None:
    observed = validate_dcp4_smoke_evidence(
        strict_smoke["root"], plan_path=PLAN, project_root=PROJECT
    )
    assert observed["chunked_prefill_proof"]["document_tokens"] == 3_586
    assert observed["runtime"]["max_num_batched_tokens"] == 2_048


@pytest.mark.parametrize(
    ("decode_context_parallel_size", "dcp_raw", "cp_raw", "expected"),
    (
        (4, 64, 1, 64),
        (4, 64, 64, 64),
        (4, 1, 32, 32),
        (1, 64, 1, 1),
    ),
)
def test_effective_interleave_exactly_models_pinned_vllm_precedence(
    decode_context_parallel_size: int,
    dcp_raw: int,
    cp_raw: int,
    expected: int,
) -> None:
    assert effective_kv_cache_interleave_size(
        decode_context_parallel_size=decode_context_parallel_size,
        dcp_kv_cache_interleave_size_raw=dcp_raw,
        cp_kv_cache_interleave_size_raw=cp_raw,
    ) == expected


@pytest.mark.parametrize(
    ("field", "drift"),
    (
        ("dcp_kv_cache_interleave_size_raw", 32),
        ("cp_kv_cache_interleave_size_raw", 64),
        ("effective_kv_cache_interleave_size", 1),
        ("kv_cache_interleave_resolution", "unproved"),
    ),
)
def test_strict_smoke_rejects_every_raw_or_effective_interleave_drift(
    strict_smoke: dict,
    field: str,
    drift: object,
) -> None:
    value = copy.deepcopy(strict_smoke["evidence"])
    value["runtime"][field] = drift
    value["runtime"]["worker_runtime_audit"][field] = drift
    _rewrite_evidence(strict_smoke["root"], value)
    with pytest.raises(ValueError, match="interleave|runtime"):
        validate_dcp4_smoke_evidence(
            strict_smoke["root"], plan_path=PLAN, project_root=PROJECT
        )


@pytest.mark.parametrize(
    "category",
    (
        "plan",
        "document",
        "status",
        "abort",
        "chunking",
        "worker_runtime",
        "runtime_file",
        "fused",
        "router",
        "capture_code",
        "teacher_identity",
        "token",
    ),
)
def test_strict_smoke_rejects_each_tamper_class(
    strict_smoke: dict,
    category: str,
) -> None:
    value = copy.deepcopy(strict_smoke["evidence"])
    if category == "plan":
        value["plan_fingerprint"] = "0" * 64
    elif category == "document":
        value["document"]["tokens"] -= 1
    elif category == "status":
        value["status"]["documents_verified"] = 0
    elif category == "abort":
        value["abort"]["partial_payloads_retained"] = False
    elif category == "chunking":
        value["chunked_prefill_proof"]["max_num_batched_tokens"] = 3_586
    elif category == "worker_runtime":
        value["runtime"]["worker_runtime_audit"]["enable_chunked_prefill"] = False
    elif category == "runtime_file":
        name = next(iter(value["runtime"]["runtime_provenance"]["files"]))
        value["runtime"]["runtime_provenance"]["files"][name]["sha256"] = "0" * 64
    elif category == "fused":
        value["runtime"]["all_rank_fused_layer_audits"]["2"][
            "reserved_layers"
        ] = [6]
    elif category == "router":
        value["router_audits"]["28"]["runner_output_scale"] = 1.0
    elif category == "capture_code":
        value["runtime"]["capture_code"]["files"].pop("src/__init__.py")
    elif category == "teacher_identity":
        value["runtime"]["teacher_identity"]["verification_mode"] = "metadata"
    elif category == "token":
        value["smoke_run_token"] = "not-a-token"
    _rewrite_evidence(strict_smoke["root"], value)
    with pytest.raises(ValueError):
        validate_dcp4_smoke_evidence(
            strict_smoke["root"], plan_path=PLAN, project_root=PROJECT
        )


@pytest.mark.parametrize("mutation", ("missing", "extra", "final"))
def test_strict_smoke_rejects_partial_filesystem_drift(
    strict_smoke: dict,
    mutation: str,
) -> None:
    directory = strict_smoke["root"] / "layer_006"
    partial = directory / "hidden.bf16.bin.partial"
    if mutation == "missing":
        partial.unlink()
    elif mutation == "extra":
        (directory / "unexpected.bin").write_bytes(b"")
    else:
        (directory / "hidden.bf16.bin").write_bytes(b"")
    with pytest.raises(ValueError, match="partial-file set"):
        validate_dcp4_smoke_evidence(
            strict_smoke["root"], plan_path=PLAN, project_root=PROJECT
        )


@pytest.mark.parametrize("mutation", ("edit", "add", "remove"))
def test_full_rejects_any_post_smoke_jit_cache_drift(
    strict_smoke: dict,
    mutation: str,
) -> None:
    cache = strict_smoke["cache"]
    generated = cache / "triton" / "kernel.bin"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"compiled")
    write_jit_cache_binding(
        cache, strict_smoke["root"], plan_path=PLAN, project_root=PROJECT
    )
    assert validate_jit_cache_binding(
        cache, strict_smoke["root"], plan_path=PLAN, project_root=PROJECT
    )["cache_inventory"]["files"]["triton/kernel.bin"]["bytes"] == 8

    if mutation == "edit":
        generated.write_bytes(b"tampered")
    elif mutation == "add":
        (cache / "new.bin").write_bytes(b"new")
    else:
        generated.unlink()
    with pytest.raises(ValueError, match="binding differs"):
        validate_jit_cache_binding(
            cache, strict_smoke["root"], plan_path=PLAN, project_root=PROJECT
        )


def test_full_preflight_seals_smoke_token_sha_and_jit_binding(
    strict_smoke: dict,
) -> None:
    cache = strict_smoke["cache"]
    generated = cache / "triton" / "kernel.bin"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"compiled")
    binding = write_jit_cache_binding(
        cache, strict_smoke["root"], plan_path=PLAN, project_root=PROJECT
    )
    token = strict_smoke["evidence"]["smoke_run_token"]
    receipt = validate_full_capture_preflight(
        cache,
        strict_smoke["root"],
        plan_path=PLAN,
        project_root=PROJECT,
        expected_smoke_token=token,
    )
    assert receipt["smoke"]["run_token"] == token
    assert receipt["jit_cache"]["binding_id"] == binding["binding_id"]
    assert receipt["smoke"]["evidence_sha256"]
    assert validate_full_capture_preflight_record(receipt) == receipt

    tampered = copy.deepcopy(receipt)
    tampered["jit_cache"]["binding_id"] = "0" * 64
    with pytest.raises(ValueError, match="receipt digest"):
        validate_full_capture_preflight_record(tampered)


def test_full_preflight_rejects_token_not_derived_from_smoke(
    strict_smoke: dict,
) -> None:
    cache = strict_smoke["cache"]
    cache.mkdir()
    write_jit_cache_binding(
        cache, strict_smoke["root"], plan_path=PLAN, project_root=PROJECT
    )
    with pytest.raises(ValueError, match="token environment"):
        validate_full_capture_preflight(
            cache,
            strict_smoke["root"],
            plan_path=PLAN,
            project_root=PROJECT,
            expected_smoke_token=str(uuid.uuid4()),
        )


def test_direct_full_capture_cli_requires_mounted_preflight_paths(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            "/usr/bin/python3",
            str(PROJECT / "capture_calibration.py"),
            "--capture",
            "--model",
            str(tmp_path / "model"),
            "--owner-manifest",
            str(tmp_path / "owner.json"),
            "--corpus",
            str(tmp_path / "corpus.jsonl"),
            "--plan-file",
            str(PLAN),
            "--capture-dir",
            str(tmp_path / "capture"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--smoke-dir is required for full capture" in result.stderr


def test_jit_cache_inventory_rejects_symlinks(strict_smoke: dict) -> None:
    cache = strict_smoke["cache"]
    cache.mkdir()
    (cache / "unsafe").symlink_to(strict_smoke["root"])
    with pytest.raises(ValueError, match="symlink"):
        write_jit_cache_binding(
            cache, strict_smoke["root"], plan_path=PLAN, project_root=PROJECT
        )


def test_jit_cache_inventory_fails_closed_on_unreadable_subtree(
    strict_smoke: dict,
) -> None:
    cache = strict_smoke["cache"]
    hidden = cache / "root-owned"
    hidden.mkdir(parents=True)
    (hidden / "must-not-be-omitted.bin").write_bytes(b"compiled")
    hidden.chmod(0)
    try:
        with pytest.raises(ValueError, match="cannot enumerate"):
            write_jit_cache_binding(
                cache,
                strict_smoke["root"],
                plan_path=PLAN,
                project_root=PROJECT,
            )
    finally:
        hidden.chmod(0o700)


def test_jit_cache_inventory_rejects_special_files(strict_smoke: dict) -> None:
    cache = strict_smoke["cache"]
    cache.mkdir()
    os.mkfifo(cache / "unsafe-fifo")
    with pytest.raises(ValueError, match="special file"):
        write_jit_cache_binding(
            cache, strict_smoke["root"], plan_path=PLAN, project_root=PROJECT
        )


def test_post_smoke_normalizer_makes_complete_tree_host_readable(
    strict_smoke: dict,
) -> None:
    cache = strict_smoke["cache"]
    hidden = cache / "root-owned" / "nested"
    hidden.mkdir(parents=True)
    payload = hidden / "kernel.bin"
    payload.write_bytes(b"compiled")
    payload.chmod(0o600)
    hidden.chmod(0o700)
    hidden.parent.chmod(0o700)

    inventory = normalize_jit_cache_permissions(cache)

    assert inventory["files"]["root-owned/nested/kernel.bin"]["bytes"] == 8
    assert stat.S_IMODE(hidden.stat().st_mode) & 0o055 == 0o055
    assert stat.S_IMODE(payload.stat().st_mode) & 0o044 == 0o044
    binding = write_jit_cache_binding(
        cache,
        strict_smoke["root"],
        plan_path=PLAN,
        project_root=PROJECT,
        expected_inventory_sha256=inventory["inventory_sha256"],
    )
    assert binding["cache_inventory"] == inventory


@pytest.mark.parametrize("kind", ("symlink", "fifo"))
def test_post_smoke_normalizer_rejects_unsafe_nodes(
    strict_smoke: dict,
    kind: str,
) -> None:
    cache = strict_smoke["cache"]
    cache.mkdir()
    unsafe = cache / "unsafe"
    if kind == "symlink":
        unsafe.symlink_to(strict_smoke["root"])
    else:
        os.mkfifo(unsafe)
    with pytest.raises(ValueError, match="symlink|special file"):
        normalize_jit_cache_permissions(cache)


def test_binding_refuses_container_host_inventory_disagreement(
    strict_smoke: dict,
) -> None:
    cache = strict_smoke["cache"]
    cache.mkdir()
    (cache / "kernel.bin").write_bytes(b"compiled")
    with pytest.raises(ValueError, match="inventories differ"):
        write_jit_cache_binding(
            cache,
            strict_smoke["root"],
            plan_path=PLAN,
            project_root=PROJECT,
            expected_inventory_sha256="0" * 64,
        )
    assert not (cache / JIT_CACHE_BINDING_FILE).exists()


def test_launcher_rejects_nonempty_smoke_cache_and_path_overlap(
    tmp_path: Path,
) -> None:
    launcher = PROJECT / "run_capture_container.sh"
    capture_parent = tmp_path / "capture"
    cache = tmp_path / "cache"
    capture_parent.mkdir()
    cache.mkdir()
    (cache / "stale").write_bytes(b"x")
    result = subprocess.run(
        [str(launcher), "smoke"],
        env={
            "PATH": "/usr/bin:/bin",
            "FRESH_CAPTURE_PARENT": str(capture_parent),
            "FRESH_JIT_CACHE": str(cache),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "new empty dedicated JIT cache" in result.stderr

    nested_cache = capture_parent / "nested-cache"
    nested_cache.mkdir()
    result = subprocess.run(
        [str(launcher), "smoke"],
        env={
            "PATH": "/usr/bin:/bin",
            "FRESH_CAPTURE_PARENT": str(capture_parent),
            "FRESH_JIT_CACHE": str(nested_cache),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "no ancestry overlap" in result.stderr
