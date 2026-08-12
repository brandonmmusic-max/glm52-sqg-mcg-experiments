#!/usr/bin/env python3
"""Sealed one-window BF16-reference KLD runner for GLM-5.2 SQG W4A8 on SM120.

Adapted from the sealed PP8/TP1 runner (glm52_fresh_sqg_test/b300_remote/
prefill_kld_pp8_tp1.py). The prompt identity, single 2048-token window,
full-vocabulary FP32 KL(ref||model) convention, and 2,047-position contract are
preserved byte-for-byte; only the engine topology (TP4/DCP4 on four SM120
GPUs) and the attention backend (B12X_MLA_SPARSE, the SM120 production sparse
path) change, and the tail-statistics block required by the local acceptance
contract is added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

# Mirror the branch serve-glm52.sh SM120 engine environment before torch/vllm
# import so the in-process engine matches the served engine's kernel and
# backend selection. Values already exported by the image/compose win.
_V16_ENV_DEFAULTS = {
    "GLOO_SOCKET_IFNAME": "lo",
    "NCCL_SOCKET_IFNAME": "lo",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    "CUTE_DSL_ARCH": "sm_120a",
    "TORCH_CUDA_ARCH_LIST": "12.0a",
    "NCCL_IB_DISABLE": "1",
    "NCCL_P2P_LEVEL": "SYS",
    "NCCL_PROTO": "LL,LL128,Simple",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "SAFETENSORS_FAST_GPU": "1",
    "VLLM_USE_AOT_COMPILE": "1",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
    "VLLM_USE_MEGA_AOT_ARTIFACT": "1",
    "VLLM_USE_FLASHINFER_SAMPLER": "1",
    "VLLM_USE_B12X_FP8_GEMM": "1",
    "VLLM_B12X_ABSORB_BMM": "0",
    "VLLM_USE_B12X_MOE": "1",
    "VLLM_USE_B12X_SPARSE_INDEXER": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_ENABLE_PCIE_ALLREDUCE": "1",
    "VLLM_PCIE_ALLREDUCE_BACKEND": "b12x",
    "B12X_MOE_FORCE_A8": "0",
    "B12X_MOE_FORCE_A16": "0",
    "OMP_NUM_THREADS": "16",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME", "/cache"),
}
for _name, _value in _V16_ENV_DEFAULTS.items():
    os.environ.setdefault(_name, _value)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
for _sub in (
    "vllm",
    "triton",
    "torchinductor",
    "cute-dsl",
    "b12x-cute",
    "deep-gemm",
    "flashinfer",
    "cuda",
):
    _root = os.path.join(os.environ["XDG_CACHE_HOME"], _sub)
    os.makedirs(_root, exist_ok=True)
os.environ.setdefault(
    "TRITON_CACHE_DIR", os.path.join(os.environ["XDG_CACHE_HOME"], "triton")
)
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.join(os.environ["XDG_CACHE_HOME"], "torchinductor"),
)
os.environ.setdefault(
    "CUTE_DSL_CACHE_DIR", os.path.join(os.environ["XDG_CACHE_HOME"], "cute-dsl")
)
os.environ.setdefault(
    "B12X_CUTE_COMPILE_CACHE_DIR",
    os.path.join(os.environ["XDG_CACHE_HOME"], "b12x-cute"),
)
os.environ.setdefault(
    "DG_JIT_CACHE_DIR", os.path.join(os.environ["XDG_CACHE_HOME"], "deep-gemm")
)
os.environ.setdefault(
    "VLLM_CACHE_ROOT", os.path.join(os.environ["XDG_CACHE_HOME"], "vllm")
)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from datasets import load_dataset  # noqa: E402
from safetensors.torch import safe_open  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(32 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def dense_from_flat_prompt_logprobs(
    prompt_logprobs: Any, npos: int, vocab: int
) -> torch.Tensor:
    dense = torch.empty((npos, vocab), dtype=torch.float32)
    if hasattr(prompt_logprobs, "start_indices"):
        for pos in range(npos):
            source = pos + 1
            start = prompt_logprobs.start_indices[source]
            end = prompt_logprobs.end_indices[source]
            ids = torch.tensor(prompt_logprobs.token_ids[start:end], dtype=torch.long)
            values = torch.tensor(
                prompt_logprobs.logprobs[start:end], dtype=torch.float32
            )
            row = torch.full((vocab,), float("-inf"), dtype=torch.float32)
            valid = (ids >= 0) & (ids < vocab)
            row[ids[valid]] = values[valid]
            dense[pos] = row
        return dense
    for pos in range(npos):
        row = torch.full((vocab,), float("-inf"), dtype=torch.float32)
        for token_id, logprob in prompt_logprobs[pos + 1].items():
            token = int(token_id)
            if 0 <= token < vocab:
                row[token] = float(logprob.logprob)
        dense[pos] = row
    return dense


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--reference-logits", required=True)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--decode-context-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--load-format", default="safetensors")
    parser.add_argument("--max-model-len", type=int, default=2560)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--dcp-comm-backend", default="a2a",
    )
    parser.add_argument("--dcp-kv-cache-interleave-size", type=int, default=64)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=268435456)
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        default=128,
        help="Pin the KV block count: the sealed 2,560-token window needs ~40 "
        "blocks of 64 tokens; pinning sidesteps profile-peak budget math on "
        "GPUs that carry foreign desktop/embedder allocations.",
    )
    parser.add_argument("--quantization", default="exl3")
    parser.add_argument("--attention-backend", default="B12X_MLA_SPARSE")
    parser.add_argument("--trim-fraction", type=float, default=0.005)
    parser.add_argument("--kld-chunk-rows", type=int, default=32)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--image-ref", default=os.environ.get("IMAGE_REF", ""))
    parser.add_argument("--image-id", default=os.environ.get("IMAGE_ID", ""))
    args = parser.parse_args()
    expected_attestation = {1: "tp4dcp1", 4: "tp4dcp4"}.get(
        args.decode_context_parallel_size
    )
    if args.tensor_parallel_size != 4 or expected_attestation is None:
        parser.error("sealed SM120 KLD runner requires TP4 with DCP 1 or 4")
    if args.attention_backend != "B12X_MLA_SPARSE":
        parser.error("sealed SM120 KLD requires B12X_MLA_SPARSE")
    if (
        os.environ.get("VLLM_GLM_SQG_W4A8_TOPOLOGY_ATTESTATION")
        != expected_attestation
    ):
        parser.error(
            f"VLLM_GLM_SQG_W4A8_TOPOLOGY_ATTESTATION={expected_attestation} "
            "must be exported for this DCP size"
        )
    return args


def load_tokens(args: argparse.Namespace, manifest: dict[str, Any]) -> list[int]:
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
        )
    text = "\n\n".join(
        row["text"] for row in dataset if row.get("text") and row["text"].strip()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, trust_remote_code=True
    )
    encoded = tokenizer(
        text[: args.context_length * 5],
        add_special_tokens=False,
        truncation=True,
        max_length=args.context_length,
    )["input_ids"]
    token_ids = encoded[0] if encoded and isinstance(encoded[0], list) else encoded
    expected_count = int(manifest["context_length"])
    if args.context_length != expected_count or len(token_ids) != expected_count:
        raise RuntimeError(
            f"token count differs: context={args.context_length}, "
            f"actual={len(token_ids)}, expected={expected_count}"
        )
    if token_ids[:16] != manifest["token_first16"]:
        raise RuntimeError("token fingerprint differs from BF16 reference")
    if not manifest.get("windows"):
        raise RuntimeError("BF16 reference has no windows")
    return token_ids


def collect_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    try:
        import vllm

        versions["vllm"] = vllm.__version__
    except Exception:
        versions["vllm"] = "unavailable"
    for module_name in ("b12x", "triton", "cutlass", "deep_gemm", "flashinfer"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "present")
        except Exception:
            versions[module_name] = "absent"
    return versions


def kernel_provenance() -> dict[str, Any]:
    site = Path("/opt/venv/lib/python3.12/site-packages")
    files = {
        "glm_trellis_w4a8.py": site
        / "b12x/moe/_shared/kernels/glm_trellis_w4a8.py",
        "glm_sqg_w4a8.py": site / "b12x/moe/glm_sqg_w4a8.py",
        "exl3.py": site / "vllm/model_executor/layers/quantization/exl3.py",
        "w4a16_kernel.py": site / "b12x/moe/_shared/kernels/w4a16/kernel.py",
        "trellis_linear_api.py": site / "b12x/gemm/trellis_linear/api.py",
    }
    hashes = {
        name: (sha256_file(path) if path.is_file() else "missing")
        for name, path in files.items()
    }
    from b12x.moe.glm_sqg_w4a8 import glm_route_packed_w4a8_kernel_contract

    return {
        "sha256": hashes,
        "schedule": glm_route_packed_w4a8_kernel_contract(),
    }


def tail_statistics(values: list[float], trim_fraction: float) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    ordered = sorted(finite)
    count = len(ordered)

    def quantile(fraction: float) -> float:
        if not ordered:
            return float("nan")
        index = min(count - 1, max(0, math.ceil(fraction * count) - 1))
        return ordered[index]

    trim_each_side = int(count * trim_fraction)
    trimmed = (
        ordered[trim_each_side : count - trim_each_side]
        if count > 2 * trim_each_side
        else ordered
    )
    worst = max(1, math.ceil(0.01 * count))
    return {
        "mean": sum(ordered) / count if count else float("nan"),
        "median": (
            (ordered[(count - 1) // 2] + ordered[count // 2]) / 2.0
            if count
            else float("nan")
        ),
        "trimmed_mean": sum(trimmed) / len(trimmed) if trimmed else float("nan"),
        "trim_fraction_per_side": trim_fraction,
        "trimmed_count": len(trimmed),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "cvar_worst_1pct": sum(ordered[-worst:]) / worst if ordered else float("nan"),
        "cvar_worst_1pct_count": worst,
        "max": ordered[-1] if ordered else float("nan"),
        "finite_count": count,
        "nonfinite_count": len(values) - count,
    }


def main() -> None:
    args = parse_args()
    reference = Path(args.reference_logits).resolve()
    manifest_path = reference / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    token_ids = load_tokens(args, manifest)
    token_sha256 = hashlib.sha256(
        json.dumps(list(map(int, token_ids))).encode()
    ).hexdigest()

    kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "decode_context_parallel_size": args.decode_context_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "load_format": args.load_format,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        # Explicit 64-token KV block geometry: the B12X sparse indexer and the
        # fp8_ds_mla page layout are both 64-native. Auto-selection has
        # produced mismatched manager-vs-indexer page geometry before (B300
        # campaign receipt: capture_full_bf16_pp8.py) with OOB paged reads.
        "block_size": 64,
        # Frozen accepted DCP contract (historical run_candidate_kld.sh):
        # a2a combine + 64-token KV ownership groups matching the sparse-MLA
        # page geometry. Interleave 1 stripes adjacent tokens across DCP
        # ranks and corrupts the b12x page/index gather.
        "dcp_comm_backend": args.dcp_comm_backend,
        "dcp_kv_cache_interleave_size": args.dcp_kv_cache_interleave_size,
        "disable_custom_all_reduce": True,
        "async_scheduling": False,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        **(
            {"num_gpu_blocks_override": args.num_gpu_blocks_override}
            if args.num_gpu_blocks_override > 0
            else {}
        ),
        "attention_backend": args.attention_backend,
        # Frozen contract: explicit index cache + the 78-layer pattern.
        "hf_overrides": {
            "use_index_cache": True,
            "index_topk_pattern": "FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS",
        },
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "disable_log_stats": True,
        "max_logprobs": -1,
    }
    if args.tokenizer:
        kwargs["tokenizer"] = args.tokenizer
    if args.quantization.lower() not in {"", "auto", "none", "null"}:
        kwargs["quantization"] = args.quantization
    llm = LLM(**kwargs)

    prompt: TokensPrompt = {"prompt_token_ids": token_ids}
    params = SamplingParams(
        prompt_logprobs=-1,
        flat_logprobs=True,
        max_tokens=1,
        temperature=0.0,
        detokenize=False,
    )
    started = time.monotonic()
    output = llm.generate([prompt], sampling_params=params)[0]

    logits_path = reference / "logits_0.safetensors"
    with safe_open(logits_path, framework="pt", device="cpu") as handle:
        reference_logits = handle.get_tensor("logits")
    expected_shape = list(manifest["windows"][0]["shape"])
    if list(reference_logits.shape) != expected_shape:
        raise RuntimeError("BF16 reference tensor shape differs from manifest")
    vocab = int(reference_logits.shape[-1])
    if vocab != 154_880:
        raise RuntimeError(f"vocabulary must be 154,880, got {vocab}")
    npos = len(token_ids) - 1
    if output.prompt_logprobs is None:
        raise RuntimeError("vLLM returned no prompt logprobs")
    model_logits = dense_from_flat_prompt_logprobs(
        output.prompt_logprobs, npos, vocab
    )
    if list(model_logits.shape) != expected_shape:
        raise RuntimeError(
            f"model logits shape differs: {list(model_logits.shape)} != "
            f"{expected_shape}"
        )
    model_finite = int(torch.isfinite(model_logits).sum())
    model_nonfinite_rows = int(
        (~torch.isfinite(model_logits).all(dim=-1)).sum()
    )

    position_kld: list[float] = []
    chunk_rows = max(1, args.kld_chunk_rows)
    for start in range(0, npos, chunk_rows):
        end = min(npos, start + chunk_rows)
        model_log_probs = F.log_softmax(model_logits[start:end].float(), dim=-1)
        reference_log_probs = F.log_softmax(
            reference_logits[start:end].float(), dim=-1
        )
        values = F.kl_div(
            model_log_probs,
            reference_log_probs,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        position_kld.extend(float(value) for value in values.tolist())
    if len(position_kld) != 2_047:
        raise RuntimeError(f"expected 2,047 KLD positions, got {len(position_kld)}")
    elapsed = time.monotonic() - started

    statistics = tail_statistics(position_kld, args.trim_fraction)
    gpus = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "compute_capability": list(torch.cuda.get_device_capability(index)),
        }
        for index in range(torch.cuda.device_count())
    ]
    result = {
        "schema": "glm52-bf16-reference-kld-sm120-tp4dcp4-v1",
        "complete": True,
        "model": str(Path(args.model).resolve()),
        "model_revision": os.environ.get("MODEL_REVISION_PIN", ""),
        "image_ref": args.image_ref,
        "image_id": args.image_id,
        "runtime": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "decode_context_parallel_size": args.decode_context_parallel_size,
            "pipeline_parallel_size": 1,
            "dtype": args.dtype,
            "kv_cache_dtype": args.kv_cache_dtype,
            "quantization": args.quantization,
            "attention_backend": args.attention_backend,
            "enforce_eager": True,
            "topology_attestation": os.environ.get(
                "VLLM_GLM_SQG_W4A8_TOPOLOGY_ATTESTATION"
            ),
            "dcp_comm_backend": args.dcp_comm_backend,
            "dcp_kv_cache_interleave_size": args.dcp_kv_cache_interleave_size,
            "disable_custom_all_reduce": True,
            "async_scheduling": False,
            "block_size": 64,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_model_len": args.max_model_len,
            "hf_overrides": "use_index_cache=true + 78-layer index_topk_pattern (frozen contract)",
        },
        "reference": {
            "manifest_sha256": sha256_file(manifest_path),
            "logits_sha256": sha256_file(logits_path),
            "path": str(reference),
        },
        "token_first16": list(map(int, token_ids[:16])),
        "token_sequence_sha256": token_sha256,
        "kernel_provenance": kernel_provenance(),
        "versions": collect_versions(),
        "gpus": gpus,
        "env_mirror": {
            name: os.environ.get(name)
            for name in sorted(_V16_ENV_DEFAULTS)
        },
        "model_logits_finite_values": model_finite,
        "model_logits_nonfinite_rows": model_nonfinite_rows,
        "statistics": statistics,
        "mean_kld": statistics["mean"],
        "total_positions": len(position_kld),
        "elapsed_sec": elapsed,
        "position_kld": position_kld,
    }
    atomic_json(Path(args.result_json), result)
    print(
        json.dumps(
            {
                "mean_kld": statistics["mean"],
                "median": statistics["median"],
                "trimmed_mean": statistics["trimmed_mean"],
                "p99": statistics["p99"],
                "max": statistics["max"],
                "nonfinite": statistics["nonfinite_count"],
                "positions": len(position_kld),
            }
        )
    )


if __name__ == "__main__":
    main()
