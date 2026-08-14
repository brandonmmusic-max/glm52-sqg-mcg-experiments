"""JIT binding for the B12X-owned K6/MCG small-M CUDA kernel."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_SOURCE_DIR = Path(__file__).resolve().parent / "csrc"
_SOURCE = _SOURCE_DIR / "trellis_k6_small.cu"
_VENDORED_FILES = tuple(sorted((_SOURCE_DIR / "vendor").rglob("*.[ch]*")))


_GLM_K6_DECODE_SMS = {
    # Q/indexer projection on the target stream.
    (2048, 4096): 128,
    # TP4 shared-expert FC1/FC2 run beside the target stream. These are the
    # rank-local dimensions after column/row parallel slicing, not the full
    # 4096/2048-wide shared MLP dimensions. The budgets match the E2E-optimal
    # ExLlama autotuner result; using all 188 SMs serializes the graph branches.
    (6144, 1024): 64,
    (512, 6144): 96,
}


def _default_num_sms(size_k: int, size_n: int, available_sms: int) -> int:
    """Select the measured GLM K6 decode overlap budget when applicable."""
    target = _GLM_K6_DECODE_SMS.get((size_k, size_n))
    return available_sms if target is None else min(available_sms, target)


@lru_cache(maxsize=None)
def _available_sms(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def _extension_name() -> str:
    digest = hashlib.sha256()
    for path in (_SOURCE, *_VENDORED_FILES):
        if path.is_file():
            digest.update(path.relative_to(_SOURCE_DIR).as_posix().encode())
            digest.update(path.read_bytes())
    return f"b12x_trellis_k6_{digest.hexdigest()[:12]}"


@lru_cache(maxsize=1)
def _extension():
    build_directory = os.environ.get("B12X_TRELLIS_BUILD_DIR")
    if build_directory:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    return load(
        name=_extension_name(),
        sources=[str(_SOURCE)],
        extra_include_paths=[str(_SOURCE_DIR)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-gencode=arch=compute_120,code=sm_120",
        ],
        extra_cflags=["-O3"],
        build_directory=build_directory,
        verbose=os.environ.get("B12X_JIT_VERBOSE", "0") == "1",
    )


# Codebook sentinels. SQG payloads carry "SQG1"; MCG is the EXL3 multiply-with-
# carry generator constant. These are compared as unsigned int32.
_SQG_MARKER_SQG1 = 0x53514731
_MCG_MARKER = 0xCBAC1FED
_MCG_CODEBOOK_NAMES = frozenset({"mcg", "exl3_trellis_mcg", "w4a16/exl3_trellis_mcg"})
_SQG_CODEBOOK_NAMES = frozenset(
    {
        "sqg",
        "sqg_xor_cheb_t12",
        "exl3_trellis_sqg_cheb_e4m3",
        "w4a16/exl3_trellis_sqg_cheb_e4m3",
        "w4a8/exl3_trellis_sqg_cheb_e4m3",
    }
)


def _reject_non_mcg_codebook(codebook, trellis: torch.Tensor) -> None:
    """Fail closed unless the caller proves this payload is MCG.

    The K6 small-M CUDA kernel decodes with the MCG generator unconditionally.
    Handing it SQG-encoded bytes yields silently wrong numerics rather than an
    error, so admission is explicit and mandatory.
    """
    if codebook is None:
        raise ValueError(
            "run_k6_mcg requires an explicit codebook= argument; the K6 small-M "
            "kernel is MCG-only and cannot detect a wrong codebook at runtime"
        )
    if isinstance(codebook, int):
        if codebook == _SQG_MARKER_SQG1:
            raise ValueError(
                "refusing to route SQG (marker 0x53514731 'SQG1') K6 weights "
                "through launch_k6_mcg; SQG dense K6 must use the SQG decode path"
            )
        if codebook != _MCG_MARKER:
            raise ValueError(
                f"run_k6_mcg admits only the MCG codebook marker "
                f"0x{_MCG_MARKER:08X}, got 0x{int(codebook) & 0xFFFFFFFF:08X}"
            )
    else:
        name = str(codebook).strip().lower()
        if name in _SQG_CODEBOOK_NAMES or name.startswith("sqg"):
            raise ValueError(
                f"refusing to route SQG codebook {codebook!r} K6 weights through "
                "launch_k6_mcg; SQG dense K6 must use the SQG decode path"
            )
        if name not in _MCG_CODEBOOK_NAMES:
            raise ValueError(
                f"run_k6_mcg admits only MCG codebooks {sorted(_MCG_CODEBOOK_NAMES)}, "
                f"got {codebook!r}"
            )
    # Belt-and-braces: catch an SQG sentinel embedded in the payload even when
    # the caller mislabels it.
    try:
        if trellis is not None and trellis.numel() > 0:
            head = trellis.reshape(-1)[0]
            if head.dtype == torch.int32:
                if (int(head.item()) & 0xFFFFFFFF) == _SQG_MARKER_SQG1:
                    raise ValueError(
                        "trellis payload carries the SQG1 sentinel; refusing "
                        "launch_k6_mcg"
                    )
    except (RuntimeError, ValueError) as exc:
        if "SQG1" in str(exc):
            raise


def run_k6_mcg(
    x: torch.Tensor,
    trellis: torch.Tensor,
    output: torch.Tensor,
    suh: torch.Tensor,
    rotated_input: torch.Tensor,
    svh: torch.Tensor,
    locks: torch.Tensor,
    *,
    codebook=None,
    num_sms: int = 0,
) -> None:
    """Launch the capture-safe K6/MCG kernel on Torch's current stream.

    `codebook` is mandatory and must identify an MCG payload; SQG K6 is
    rejected here on every architecture, not merely where the sm_120 build
    happens to be unavailable.
    """
    _reject_non_mcg_codebook(codebook, trellis)
    capability = torch.cuda.get_device_capability(x.device)
    if capability != (12, 0):
        raise NotImplementedError(
            "Trellis K6 small-M kernel is built for sm_120 only; "
            f"device reports sm_{capability[0]}{capability[1]}"
        )
    if num_sms <= 0:
        device_index = x.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        num_sms = _default_num_sms(
            int(x.shape[1]),
            int(output.shape[1]),
            _available_sms(int(device_index)),
        )
    _extension().launch_k6_mcg(
        x,
        trellis,
        output,
        suh,
        rotated_input,
        svh,
        locks,
        int(num_sms),
    )


__all__ = ["run_k6_mcg"]
