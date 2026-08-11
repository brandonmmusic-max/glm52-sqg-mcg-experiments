"""Fail-closed provenance contract for the GLM-5.2 capture runtime.

The OCI image identity cannot be discovered reliably from inside an ordinary
container.  The launcher therefore supplies the inspected image ID as an
explicit declaration, while this module independently hashes every serving
override that can change model execution.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from .calibration_plan import sha256_file


RUNTIME_IMAGE_REFERENCE = (
    "voipmonitor/vllm@sha256:"
    "fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82"
)
RUNTIME_IMAGE_ID = (
    "sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82"
)
RUNTIME_PYTHON = "/opt/venv/bin/python"
IMAGE_DECLARATION_ENV = "FRESH_SQG_RUNTIME_IMAGE_ID"
EXL3_RUNTIME_SHA256 = os.environ.get(
    "FRESH_SQG_EXL3_RUNTIME_SHA256",
    "1d6f141b123a243e76c3384727280bd2970cea1c1dc54f413a483b6acf3e34d2",
)
if len(EXL3_RUNTIME_SHA256) != 64 or any(
    character not in "0123456789abcdef" for character in EXL3_RUNTIME_SHA256
):
    raise ValueError("FRESH_SQG_EXL3_RUNTIME_SHA256 must be a lowercase SHA256")

# These are the exact read-only destinations and bytes inspected on the
# serving container ``glm-r33-fixed``.  Hashing the destinations catches a
# missing bind mount as well as a drifted host source.
RUNTIME_FILES = {
    "/opt/venv/lib/python3.12/site-packages/vllm/envs.py": (
        "c4937bc88db9ea5c862d65445421664d3cdd53b01fbd53500197ef17ee2345a9"
    ),
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/"
    "deepseek_v2.py": (
        "17c73d674656ab10b3f2b8b2c93f686738413e207bf75f57d92f63dac89e21e8"
    ),
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
    "quantization/exl3.py": (
        EXL3_RUNTIME_SHA256
    ),
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/model_loader/"
    "utils.py": (
        "7a14fca163114e5d88e7713577532a0161a3f6a7d1fa7a27472cc16f57a8f65b"
    ),
    "/opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/w4a16/"
    "mixed_trellis.py": (
        "788fb678765b5718e29ada7b85d6f3a49b2582bbc85a354ffe25f504bf99861b"
    ),
    "/opt/exllamav3-r7ext/exllamav3_ext.cpython-312-x86_64-linux-gnu.so": (
        "e88bc24d2c292a0b69a7ee27bb701557c16e535c9980ee870c931a2b495033bd"
    ),
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
    "fused_moe/layer.py": (
        "c888133b47507d6dc0a8e74671ef6e6b2c60c53e615e27098e354ad5b961d725"
    ),
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
    "fused_moe/runner/moe_runner.py": (
        "531a271b0f9280dbc26a0638598fec613a6f6a35e9c43ae65238e622493eda1b"
    ),
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
    "fused_moe/config.py": (
        "537612941a90b4f6a7a112bb5a5d802be66dda90725de5d335d25e974a58b366"
    ),
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
    "fused_moe/router/router_factory.py": (
        "37f9e0b0d1b3f76e30189b9458786e480b94fbef162ce8bf57d2ce8a32115294"
    ),
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
    "fused_moe/router/grouped_topk_router.py": (
        "6a541bfdcf9e9a09fc888ea3f4a2e9689181e889e3a3ac9f49716d8d59cca142"
    ),
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
    "fused_moe/router/base_router.py": (
        "47fa2324bcd9b6ebdc138d69ae6b6ecc0f2463d34bda8c5988bdb0a2a76d3fae"
    ),
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
    "fused_moe/router/fused_moe_router.py": (
        "f182f30712983f61815f8df947af597277d660cd2891066fcb13789d55f7353e"
    ),
}

RUNTIME_CLASS_SOURCES = {
    "owner": {
        "class": "DeepseekV2MoE",
        "path": (
            "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/"
            "deepseek_v2.py"
        ),
        "sha256": (
            "17c73d674656ab10b3f2b8b2c93f686738413e207bf75f57d92f63dac89e21e8"
        ),
    },
    "runner": {
        "class": "MoERunner",
        "path": (
            "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
            "fused_moe/runner/moe_runner.py"
        ),
        "sha256": (
            "531a271b0f9280dbc26a0638598fec613a6f6a35e9c43ae65238e622493eda1b"
        ),
    },
    "router": {
        "class": "GroupedTopKRouter",
        "path": (
            "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
            "fused_moe/router/grouped_topk_router.py"
        ),
        "sha256": (
            "6a541bfdcf9e9a09fc888ea3f4a2e9689181e889e3a3ac9f49716d8d59cca142"
        ),
    },
    "quant_method": {
        "class": "Exl3MoEMethod",
        "path": (
            "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
            "quantization/exl3.py"
        ),
        "sha256": (
            EXL3_RUNTIME_SHA256
        ),
    },
}

RUNTIME_ENV = {
    # Exact r33/DCP4 execution controls from the preserved prompt-logit KLD
    # launcher.  These are model-path controls, not optional performance
    # tuning: a capture with a different route/collective/kernel path belongs
    # to a different treatment regime.
    "CUDA_VISIBLE_DEVICES": "3,1,2,0",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    "CUTE_DSL_ARCH": "sm_120a",
    "TORCH_CUDA_ARCH_LIST": "12.0a",
    "FLASHINFER_CUDA_ARCH_LIST": "12.0f",
    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    "OMP_NUM_THREADS": "16",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "SAFETENSORS_FAST_GPU": "1",
    "VLLM_FASTSAFETENSORS_QUEUE_SIZE": "-1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_USE_FLASHINFER_SAMPLER": "1",
    "VLLM_USE_B12X_FP8_GEMM": "1",
    "VLLM_USE_B12X_SPARSE_INDEXER": "1",
    "VLLM_USE_B12X_MOE": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_USE_B12X_DCP_A2A": "1",
    "VLLM_DCP_A2A_MAX_TOKENS": "16",
    "VLLM_DCP_A2A_LARGE_BACKEND": "ag_rs",
    "VLLM_DCP_GLOBAL_TOPK": "1",
    "VLLM_DCP_QUERY_SPLIT": "0",
    "VLLM_B12X_MLA_CKV_GATHER": "1",
    "VLLM_USE_B12X_WO_PROJECTION": "1",
    "VLLM_USE_B12X_MHC": "1",
    "B12X_MLA_SM120_UNIFIED": "1",
    "B12X_DENSE_SPLITK_TURBO": "1",
    "B12X_MOE_FORCE_A16": "1",
    "KV_FP8_ROPE": "0",
    "VLLM_NVFP4_MLA_SCALES_FILE": "",
    "VLLM_NVFP4_MLA_DYNAMIC_SCALE": "0",
    "VLLM_EXL3_EXT_PATH": "/opt/exllamav3-r7ext",
    "VLLM_EXL3_R7_FUSED": "1",
    "VLLM_EXL3_R7_FUSED_LAYERS": "48",
    "VLLM_EXL3_R7_A1_MIN_ROWS": "0",
    "VLLM_EXL3_TRELLIS_MIN_M": "4",
    "VLLM_EXL3_TRELLIS_MAX_M": "32",
    "VLLM_EXL3_TRELLIS_BLOCK_M": "8",
    "VLLM_EXL3_PREFILL_CHUNK": "128",
    "VLLM_EXL3_PREFILL_TRELLIS": "1",
    "VLLM_EXL3_PREFILL_BLOCK_M": "64",
    "VLLM_EXL3_PREFILL_CAPACITY": "2048",
    "VLLM_EXL3_PREFILL_SYMMETRIC_TILE": "0",
    "CUDA_LAUNCH_BLOCKING": "0",
    # Dedicated writable compilation cache shared only by DCP4 smoke/full.
    "VLLM_CACHE_DIR": "/cache/jit/vllm",
    "TRITON_CACHE_DIR": "/cache/jit/triton",
    "TORCH_EXTENSIONS_DIR": "/cache/jit/torch_extensions",
    "TORCHINDUCTOR_CACHE_DIR": "/cache/jit/torchinductor",
    "XDG_CACHE_HOME": "/cache/jit",
}

# Fail closed on alternate spellings or additional endpoint controls in these
# namespaces.  Checking only the expected values is insufficient if another
# alias can override the same DCP or EXL3 decision later in import order.
RUNTIME_ENV_EXCLUSIVE_PREFIXES = (
    "VLLM_DCP_",
    "VLLM_USE_B12X_DCP_",
    "VLLM_EXL3_R7_",
    "VLLM_EXL3_TRELLIS_",
    "VLLM_EXL3_PREFILL_",
)


def validate_capture_runtime(
    *,
    executable: str | None = None,
    environ: dict[str, str] | None = None,
    container_marker: Path = Path("/.dockerenv"),
    runtime_files: dict[str, str] | None = None,
) -> dict:
    """Verify and return immutable runtime evidence before any model load."""

    executable = sys.executable if executable is None else str(executable)
    environ = dict(os.environ) if environ is None else dict(environ)
    runtime_files = RUNTIME_FILES if runtime_files is None else dict(runtime_files)
    if executable != RUNTIME_PYTHON:
        raise RuntimeError(
            f"capture requires {RUNTIME_PYTHON}, got {executable}; host vLLM is invalid"
        )
    if not container_marker.is_file():
        raise RuntimeError("capture requires the hash-sealed OCI runtime")
    if environ.get(IMAGE_DECLARATION_ENV) != RUNTIME_IMAGE_ID:
        raise RuntimeError(
            f"{IMAGE_DECLARATION_ENV} must equal the inspected image ID"
        )
    observed_env = {key: environ.get(key) for key in RUNTIME_ENV}
    if observed_env != RUNTIME_ENV:
        raise RuntimeError(
            f"EXL3 capture environment differs: {observed_env} != {RUNTIME_ENV}"
        )
    unexpected_controls = sorted(
        key
        for key in environ
        if key not in RUNTIME_ENV
        and any(key.startswith(prefix) for prefix in RUNTIME_ENV_EXCLUSIVE_PREFIXES)
    )
    if unexpected_controls:
        raise RuntimeError(
            "capture environment contains unsealed DCP/EXL3 control aliases: "
            f"{unexpected_controls}"
        )

    files = {}
    for filename, expected_sha256 in runtime_files.items():
        path = Path(filename)
        if not path.is_file():
            raise RuntimeError(f"required runtime file is absent: {filename}")
        observed_sha256 = sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"runtime file drift: {filename} {observed_sha256} != {expected_sha256}"
            )
        files[filename] = {
            "sha256": observed_sha256,
            "bytes": path.stat().st_size,
        }

    return {
        "image_reference": RUNTIME_IMAGE_REFERENCE,
        "image_id_declared": environ[IMAGE_DECLARATION_ENV],
        "image_identity_observation": (
            "host docker inspect declaration; independently bound below by exact "
            "mounted execution-file hashes"
        ),
        "python_executable": executable,
        "environment": observed_env,
        "files": files,
    }
