#!/usr/bin/env python3
"""Prepare, run, or validate the isolated four-layer calibration capture.

No mode edits the source model.  ``--plan`` only tokenizes and seals the fixed
owner corpus.  ``--capture`` is the only mode that loads a model, and it refuses
to write into a nonempty output directory.  ``--status`` is read-only.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
import uuid

from src.calibration_capture import (
    ATTENTION_BACKEND,
    EFFECTIVE_KV_CACHE_DTYPE,
    INDEX_TOPK_PATTERN,
    KV_CACHE_INTERLEAVE_RESOLUTION,
    KV_CACHE_DTYPE,
    SELECTED_LAYERS,
    TEACHER_IDENTITY_SEAL_SHA256,
    capture_code_evidence,
    validate_capture,
    validate_dcp4_smoke_evidence,
    validate_fused_layer_audit,
    validate_full_capture_preflight,
    validate_smoke_run_token,
    validate_teacher_identity_evidence,
    write_capture_manifest,
    write_layer_manifest,
)
from src.calibration_plan import (
    atomic_json,
    build_document_plan,
    load_document_plan,
    load_plan_tokens,
    sha256_file,
    tokenizer_identity,
)
from src.capture_runtime import validate_capture_runtime
from src.calibration_recovery import recover_finalized_capture
from src.teacher_identity import validate_teacher_identity_receipt


ENGINE_SEED = 0
MAX_MODEL_LEN = 4_352
MAX_NUM_BATCHED_TOKENS = 2_048
GPU_MEMORY_UTILIZATION = 0.90
KV_CACHE_MEMORY_BYTES = 268_435_456
TEACHER_IDENTITY_RECEIPT = Path("/work/evidence/teacher_model_identity.json")
PROTECTED_LLM_KEYS = {
    "model",
    "tokenizer",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "decode_context_parallel_size",
    "dcp_comm_backend",
    "dcp_kv_cache_interleave_size",
    "cp_kv_cache_interleave_size",
    "enable_expert_parallel",
    "enable_eplb",
    "moe_backend",
    "enforce_eager",
    "enable_prefix_caching",
    "enable_chunked_prefill",
    "max_model_len",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "kv_cache_memory_bytes",
    "max_num_seqs",
    "speculative_config",
    "worker_extension_cls",
    "seed",
    "dtype",
    "quantization",
    "load_format",
    "attention_backend",
    "kv_cache_dtype",
    "hf_overrides",
    "disable_custom_all_reduce",
    "async_scheduling",
    "trust_remote_code",
}


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}", flush=True)


def _tokenizer(model: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model), trust_remote_code=False)


def prepare_plan(args) -> None:
    tokenizer = _tokenizer(args.model)
    plan = build_document_plan(
        args.owner_manifest,
        args.corpus,
        args.model,
        tokenizer,
    )
    if args.plan_file.exists():
        existing = load_document_plan(args.plan_file)
        if existing != plan:
            raise RuntimeError(
                f"existing plan {args.plan_file} differs; choose a new path explicitly"
            )
        log(f"document plan already exists and closes: {args.plan_file}")
        return
    atomic_json(args.plan_file, plan)
    log(
        f"sealed {plan['documents_total']} documents / {plan['tokens_total']} tokens: "
        f"{args.plan_file} fingerprint={plan['plan_fingerprint']}"
    )


def _validate_model_binding(model: Path, plan: dict, tokenizer: object) -> None:
    source = plan["activation_source"]
    if sha256_file(model / "config.json") != source["config_sha256"]:
        raise RuntimeError("calibration activation-generator config differs from plan")
    if tokenizer_identity(model, tokenizer) != plan["tokenizer"]:
        raise RuntimeError("calibration activation-generator tokenizer differs from plan")
    for filename, key in (
        ("MANIFEST.json", "manifest_json_sha256"),
        ("MANIFEST.sha256", "manifest_sha256_file_sha256"),
    ):
        expected = source.get(key)
        path = model / filename
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            raise RuntimeError(f"calibration activation-generator {filename} differs")


def _make_llm(args):
    here = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = here + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM

    extra = json.loads(args.llm_extra_json)
    if not isinstance(extra, dict):
        raise ValueError("--llm-extra-json must decode to an object")
    forbidden = sorted(PROTECTED_LLM_KEYS.intersection(extra))
    if forbidden:
        raise ValueError(f"--llm-extra-json cannot override capture gates: {forbidden}")
    if extra:
        raise ValueError("--llm-extra-json must be empty for the sealed capture")
    kwargs = {
        "model": str(args.model),
        "tokenizer": str(args.model),
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "decode_context_parallel_size": 4,
        "dcp_comm_backend": "a2a",
        "dcp_kv_cache_interleave_size": 64,
        "enable_expert_parallel": False,
        "enable_eplb": False,
        # The source activation generator is the exact custom EXL3 path used
        # by the serving image.  Exl3MoEMethod remains externally routed even
        # when its R7 expert kernel is fused, so select_experts is observable.
        "moe_backend": "b12x",
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": 1,
        "speculative_config": None,
        "worker_extension_cls": (
            "src.glm52_capture_worker.FreshSQGCaptureWorkerExtension"
        ),
        "seed": ENGINE_SEED,
        "dtype": "bfloat16",
        "quantization": "exl3",
        "load_format": "safetensors",
        "attention_backend": ATTENTION_BACKEND,
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "hf_overrides": {
            "use_index_cache": True,
            "index_topk_pattern": INDEX_TOPK_PATTERN,
        },
        "disable_custom_all_reduce": True,
        "async_scheduling": False,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "kv_cache_memory_bytes": KV_CACHE_MEMORY_BYTES,
        "trust_remote_code": True,
        "disable_log_stats": True,
        "generation_config": "vllm",
    }
    kwargs.update(extra)
    return LLM(**kwargs), extra


def _rank_zero_result(results: list[dict], *, method: str) -> dict:
    capturing = [value for value in results if value.get("capturing")]
    if len(capturing) != 1 or int(capturing[0].get("rank", -1)) != 0:
        raise RuntimeError(f"{method}: expected exactly one TP-rank-zero result: {results}")
    if sorted(int(value.get("rank", -1)) for value in results) != [0, 1, 2, 3]:
        raise RuntimeError(f"{method}: collective result does not cover TP ranks 0..3")
    return capturing[0]


def _all_rank_fused_audits(results: list[dict]) -> dict[str, dict]:
    audits: dict[str, dict] = {}
    for value in results:
        rank = int(value.get("rank", -1))
        if rank in (-1,) or str(rank) in audits:
            raise RuntimeError(f"capture_init: invalid/duplicate rank evidence: {results}")
        audits[str(rank)] = validate_fused_layer_audit(
            value.get("fused_layer_audit")
        )
    if set(audits) != {"0", "1", "2", "3"}:
        raise RuntimeError("capture_init: fused audit does not cover TP ranks 0..3")
    return dict(sorted(audits.items()))


def _runtime_record(
    *,
    extra: dict,
    runtime_provenance: dict,
    rank_zero: dict,
    all_rank_fused_audits: dict[str, dict],
    teacher_identity: dict,
    full_capture_preflight: dict | None = None,
) -> dict:
    return {
        "engine_seed": ENGINE_SEED,
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
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "kv_cache_memory_bytes": KV_CACHE_MEMORY_BYTES,
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
                "capture": MAX_MODEL_LEN,
                "saved_kld": 2_560,
                "reason": "owner corpus permits intact documents through 4096 tokens",
            },
            "all_other_scheduler_cache_controls_match_saved_kld": True,
        },
        "llm_extra": extra,
        "runtime_provenance": runtime_provenance,
        "worker_runtime_audit": rank_zero["runtime"],
        "worker_hf_contract": rank_zero["hf_contract"],
        "all_rank_fused_layer_audits": all_rank_fused_audits,
        "capture_code": capture_code_evidence(Path(__file__).resolve().parent),
        "full_capture_preflight": full_capture_preflight,
        "teacher_identity": validate_teacher_identity_evidence(teacher_identity),
        "vllm_version": _vllm_version(),
    }


def run_capture(args, *, smoke_dcp4: bool = False) -> None:
    plan = load_document_plan(args.plan_file)
    smoke_token = validate_smoke_run_token(
        os.environ.get("FRESH_SQG_SMOKE_RUN_TOKEN")
    )
    if smoke_dcp4:
        full_capture_preflight = None
    else:
        full_capture_preflight = validate_full_capture_preflight(
            args.jit_cache_dir,
            args.smoke_dir,
            plan_path=args.plan_file,
            project_root=Path(__file__).resolve().parent,
            expected_smoke_token=smoke_token,
        )
    runtime_provenance = validate_capture_runtime()
    teacher_identity = validate_teacher_identity_evidence(
        validate_teacher_identity_receipt(
            args.model,
            TEACHER_IDENTITY_RECEIPT,
            expected_seal_sha256=TEACHER_IDENTITY_SEAL_SHA256,
            verify_mode="full",
            workers=4,
        )
    )
    from vllm import SamplingParams

    tokenizer = _tokenizer(args.model)
    _validate_model_binding(args.model, plan, tokenizer)
    token_lists = load_plan_tokens(plan, args.corpus, tokenizer)
    if len(token_lists) != len(plan["documents"]):
        raise AssertionError("plan/token list length differs")

    if args.capture_dir.exists() and any(args.capture_dir.iterdir()):
        raise RuntimeError(
            f"capture directory is not empty: {args.capture_dir}; use a new directory"
        )
    args.capture_dir.mkdir(parents=True, exist_ok=True)
    capture_plan_path = args.capture_dir / "document_plan.json"
    atomic_json(capture_plan_path, plan)
    run_uuid = str(uuid.uuid4())
    log(
        "loading read-only calibration activation generator under exact KLD TP4/DCP4, "
        "eager, one request at a time; production is not addressed by this script"
    )
    llm, extra = _make_llm(args)
    initialized = False
    try:
        init_results = llm.collective_rpc(
            "fresh_sqg_capture_init",
            args=(str(args.capture_dir), plan["plan_fingerprint"]),
        )
        all_rank_fused_audits = _all_rank_fused_audits(init_results)
        rank_zero = _rank_zero_result(init_results, method="capture_init")
        if rank_zero.get("layers") != list(SELECTED_LAYERS):
            raise RuntimeError("capture worker selected-layer set differs")
        initialized = True
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            ignore_eos=True,
            detokenize=False,
        )
        started = time.time()
        cumulative = 0
        documents = list(zip(plan["documents"], token_lists, strict=True))
        if smoke_dcp4:
            documents = documents[:1]
        for index, (document, token_ids) in enumerate(documents):
            begin = llm.collective_rpc(
                "fresh_sqg_begin_document",
                args=(
                    int(document["epoch"]),
                    int(document["role_id"]),
                    str(document["document_sha256"]),
                    int(document["tokens"]),
                ),
            )
            _rank_zero_result(begin, method="begin_document")
            outputs = llm.generate(
                [{"prompt_token_ids": token_ids}],
                sampling_params=sampling,
                use_tqdm=False,
            )
            if len(outputs) != 1:
                raise RuntimeError(f"document epoch {index}: output count differs")
            returned_prompt = getattr(outputs[0], "prompt_token_ids", None)
            if returned_prompt is not None and list(returned_prompt) != token_ids:
                raise RuntimeError(f"document epoch {index}: vLLM prompt IDs differ")
            if not outputs[0].outputs or len(outputs[0].outputs[0].token_ids) != 1:
                raise RuntimeError(f"document epoch {index}: max_tokens=1 contract differs")
            end = llm.collective_rpc("fresh_sqg_end_document")
            end_zero = _rank_zero_result(end, method="end_document")
            cumulative += len(token_ids)
            if int(end_zero.get("cumulative_tokens", -1)) != cumulative:
                raise RuntimeError(f"document epoch {index}: cumulative rows differ")
            if (index + 1) % 25 == 0 or index + 1 == len(documents):
                elapsed = max(time.time() - started, 1e-9)
                log(
                    f"verified {index + 1}/{len(documents)} one-document requests, "
                    f"{cumulative}/{plan['tokens_total']} rows/layer, "
                    f"{cumulative / elapsed:.0f} tok/s"
                )

        if smoke_dcp4:
            smoke_document = plan["documents"][0]
            expected_rows = int(smoke_document["tokens"])
            status_results = llm.collective_rpc("fresh_sqg_capture_status")
            smoke_status = _rank_zero_result(status_results, method="capture_status")
            expected_counts = {
                str(layer): expected_rows for layer in SELECTED_LAYERS
            }
            if smoke_status.get("counts") != expected_counts:
                raise RuntimeError(
                    "DCP4 smoke did not give TP rank zero every MoE row: "
                    f"{smoke_status.get('counts')} != {expected_counts}"
                )
            abort_results = llm.collective_rpc("fresh_sqg_capture_abort")
            abort_zero = _rank_zero_result(abort_results, method="capture_abort")
            initialized = False
            if (args.capture_dir / "capture_manifest.json").exists():
                raise RuntimeError("DCP4 smoke must not create a completion sentinel")
            evidence = {
                "schema": "glm52-fresh-sqg-dcp4-row-ownership-smoke-v1",
                "passed": True,
                "completion_sentinel_created": False,
                "smoke_run_token": os.environ.get("FRESH_SQG_SMOKE_RUN_TOKEN"),
                "plan_fingerprint": plan["plan_fingerprint"],
                "document": smoke_document,
                "selected_layers": list(SELECTED_LAYERS),
                "expected_rows_per_layer": expected_rows,
                "chunked_prefill_proof": {
                    "document_tokens": expected_rows,
                    "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                    "requires_multiple_scheduler_chunks": True,
                    "selected_layer_counts": expected_counts,
                },
                "status": smoke_status,
                "abort": abort_zero,
                "runtime": _runtime_record(
                    extra=extra,
                    runtime_provenance=runtime_provenance,
                    rank_zero=rank_zero,
                    all_rank_fused_audits=all_rank_fused_audits,
                    teacher_identity=teacher_identity,
                    full_capture_preflight=None,
                ),
                "router_audits": rank_zero["router_audits"],
            }
            atomic_json(args.capture_dir / "dcp4_smoke_evidence.json", evidence)
            validate_dcp4_smoke_evidence(
                args.capture_dir,
                plan_path=args.plan_file,
                project_root=Path(__file__).resolve().parent,
            )
            log(
                "DCP4 ROW-OWNERSHIP SMOKE PASSED AND ABORTED: "
                f"{args.capture_dir}"
            )
            return

        final_results = llm.collective_rpc("fresh_sqg_capture_finalize")
        final = _rank_zero_result(final_results, method="capture_finalize")
        initialized = False
        if final.get("plan_fingerprint") != plan["plan_fingerprint"]:
            raise RuntimeError("worker finalized against another document plan")
        layer_manifests = []
        for layer in SELECTED_LAYERS:
            layer_manifests.append(
                write_layer_manifest(
                    args.capture_dir,
                    layer,
                    plan,
                    final["layers"][str(layer)],
                    run_uuid=run_uuid,
                )
            )
        runtime = _runtime_record(
            extra=extra,
            runtime_provenance=runtime_provenance,
            rank_zero=rank_zero,
            all_rank_fused_audits=all_rank_fused_audits,
            teacher_identity=teacher_identity,
            full_capture_preflight=full_capture_preflight,
        )
        write_capture_manifest(
            args.capture_dir,
            capture_plan_path,
            plan,
            layer_manifests,
            run_uuid=run_uuid,
            runtime=runtime,
        )
        # Payload hashes were independently reread while writing layer
        # manifests.  This pass checks row semantics and document boundaries.
        validate_capture(args.capture_dir, verify_hashes=False)
        log(f"CAPTURE SEALED: {args.capture_dir}")
    except Exception:
        if initialized:
            try:
                llm.collective_rpc("fresh_sqg_capture_abort")
            except Exception as abort_error:
                log(f"capture abort RPC also failed: {abort_error!r}")
        raise
    finally:
        del llm
        gc.collect()


def _vllm_version() -> str:
    try:
        import vllm

        return str(vllm.__version__)
    except Exception:
        return "unknown"


def status(args) -> None:
    manifest = validate_capture(args.capture_dir, verify_hashes=not args.skip_hashes)
    print(
        json.dumps(
            {
                "status": "ready",
                "capture_run_uuid": manifest["capture_run_uuid"],
                "selected_layers": manifest["selected_layers"],
                "tokens_per_layer": manifest["tokens_per_layer"],
                "documents": manifest["documents"],
                "evidence_mode": manifest.get("evidence_mode", "live_exact_v1"),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true", help="tokenize/seal only; no model load")
    modes.add_argument("--capture", action="store_true", help="load model and capture")
    modes.add_argument(
        "--smoke-dcp4",
        action="store_true",
        help="capture one document, prove DCP4 row ownership, then abort",
    )
    modes.add_argument("--status", action="store_true", help="read-only capture validation")
    modes.add_argument(
        "--recover-finalized",
        action="store_true",
        help="CPU-only seal of fully promoted payloads after the known host v1 failure",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--owner-manifest", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--jit-cache-dir", type=Path)
    parser.add_argument("--llm-extra-json", default="{}")
    parser.add_argument("--skip-hashes", action="store_true", help="status only")
    args = parser.parse_args()

    for name in ("model", "owner_manifest", "corpus", "plan_file"):
        if not (args.status or args.recover_finalized) and getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if (
        args.capture
        or args.smoke_dcp4
        or args.status
        or args.recover_finalized
    ) and args.capture_dir is None:
        parser.error("--capture-dir is required")
    if args.capture:
        if args.smoke_dir is None:
            parser.error("--smoke-dir is required for full capture")
        if args.jit_cache_dir is None:
            parser.error("--jit-cache-dir is required for full capture")
    if args.recover_finalized:
        if args.smoke_dir is None:
            parser.error("--smoke-dir is required for finalized recovery")
        if args.jit_cache_dir is None:
            parser.error("--jit-cache-dir is required for finalized recovery")
    if args.status:
        status(args)
        return
    for name in ("model", "owner_manifest", "corpus", "plan_file"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.capture_dir is not None:
        args.capture_dir = args.capture_dir.resolve()
    if args.smoke_dir is not None:
        args.smoke_dir = args.smoke_dir.resolve()
    if args.jit_cache_dir is not None:
        args.jit_cache_dir = args.jit_cache_dir.resolve()
    if args.recover_finalized:
        manifest = recover_finalized_capture(
            args.capture_dir,
            smoke_dir=args.smoke_dir,
            jit_cache_dir=args.jit_cache_dir,
            project_root=Path(__file__).resolve().parent,
        )
        log(
            "RECOVERED CAPTURE SEALED WITHOUT MODEL EXECUTION: "
            f"{args.capture_dir} run_uuid={manifest['capture_run_uuid']}"
        )
        return
    if args.plan:
        prepare_plan(args)
    else:
        run_capture(args, smoke_dcp4=args.smoke_dcp4)


if __name__ == "__main__":
    main()
