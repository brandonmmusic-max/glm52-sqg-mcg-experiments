"""Research-only bridge from EXL's production encoder to SQG E4M3 tables."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
import re
import sys
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from kquant.sqg_e4m3 import sqg_e4m3_bytes


_EXTENSION_NAME = "kquant_sqg_quantize_ext_v22"
_EXTENSION_PATH_ENV = "KQUANT_SQG_EXTENSION_PATH"
_EXTENSION_SHA256_ENV = "KQUANT_SQG_EXTENSION_SHA256"
_REQUIRE_PREBUILT_ENV = "KQUANT_SQG_REQUIRE_PREBUILT"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_prebuilt_extension(path: Path):
    """Import one exact pybind extension without invoking the JIT builder."""

    spec = importlib.util.spec_from_file_location(_EXTENSION_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot construct an import spec for SQG extension: {path}")
    previous = sys.modules.get(_EXTENSION_NAME)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_EXTENSION_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(_EXTENSION_NAME, None)
        else:
            sys.modules[_EXTENSION_NAME] = previous
        raise
    return module


def _load_prebuilt_extension(raw_path: str, expected_sha256: str):
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        raise ValueError(f"{_EXTENSION_PATH_ENV} must be an absolute path")
    if candidate.is_symlink():
        raise ValueError(f"{_EXTENSION_PATH_ENV} may not name a symlink")
    try:
        path = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"sealed SQG extension does not exist: {candidate}"
        ) from exc
    if not path.is_file():
        raise ValueError(f"sealed SQG extension is not a regular file: {path}")
    allowed_names = {
        f"{_EXTENSION_NAME}{suffix}"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    }
    if path.name not in allowed_names:
        raise ValueError(
            "sealed SQG extension filename must identify "
            f"{_EXTENSION_NAME}: {path.name}"
        )
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError(f"{_EXTENSION_SHA256_ENV} must be 64 lowercase hex digits")
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "sealed SQG extension SHA256 mismatch: "
            f"{observed_sha256} != {expected_sha256}"
        )
    module = _import_prebuilt_extension(path)
    module_path = Path(str(getattr(module, "__file__", ""))).resolve()
    if module_path != path:
        raise RuntimeError(
            f"loaded SQG extension origin differs: {module_path} != {path}"
        )
    if not callable(getattr(module, "quantize_tiles_procedural", None)) or not callable(
        getattr(module, "quantize_tiles_sqg", None)
    ):
        raise RuntimeError(
            "sealed SQG extension lacks required quantization entry points"
        )
    if _sha256_file(path) != expected_sha256:
        raise RuntimeError("sealed SQG extension changed while it was being imported")
    return module


@lru_cache(maxsize=1)
def _extension():
    require_value = os.environ.get(_REQUIRE_PREBUILT_ENV, "")
    if require_value not in ("", "0", "1"):
        raise ValueError(f"{_REQUIRE_PREBUILT_ENV} must be unset, 0, or 1")
    raw_path = os.environ.get(_EXTENSION_PATH_ENV, "")
    expected_sha256 = os.environ.get(_EXTENSION_SHA256_ENV, "")
    if require_value == "1" or raw_path or expected_sha256:
        if not raw_path or not expected_sha256:
            raise RuntimeError(
                f"{_EXTENSION_PATH_ENV} and {_EXTENSION_SHA256_ENV} are both required "
                "for sealed prebuilt SQG loading"
            )
        return _load_prebuilt_extension(raw_path, expected_sha256)

    project = Path(__file__).resolve().parents[1]
    return load(
        name=_EXTENSION_NAME,
        sources=[
            str(project / "kquant/csrc/sqg_quantize.cpp"),
            str(project / "kquant/csrc/sqg_quantize.cu"),
        ],
        extra_include_paths=[str(project / "kquant/csrc")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-Xcudafe",
            "--diag_suppress=177",
            "-Xcudafe",
            "--diag_suppress=20012",
        ],
        verbose=False,
    )


@lru_cache(maxsize=None)
def _sqg_temp_buffers(device: torch.device, bits: int):
    """Allocate the packed traceback buffers consumed by kquant's kernel."""

    multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
    edges = 65536 >> bits
    decisions = edges // (4 if bits == 2 else 2)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    decision_bytes_per_tile = 256 * decisions
    affordable = max(256, int(free_bytes * 0.5) // decision_bytes_per_tile)
    max_batch = min(max(256, 3 * multiprocessors), affordable)
    costs = torch.zeros(
        (max_batch, 2, edges), dtype=torch.float16, device=device
    )
    traceback = torch.empty(
        (max_batch, 256, decisions), dtype=torch.uint8, device=device
    )
    return costs, traceback


def install_sqg_quantizer(quantizer_module) -> None:
    """Teach a loaded EXL encoder module to consume ``sqg_e4m3_lut``.

    The patch is process-local. SQG and explicit MCG/MUL1 controls use
    kquant's CUDA extension with one Viterbi/tail-biting implementation. A
    ``None`` entry in a rate-specific mapping explicitly selects MCG,
    permitting controlled hybrid rate-curve studies. Calls without any
    kquant codebook argument retain the unmodified upstream EXL behavior.
    """

    if getattr(quantizer_module, "_kquant_sqg_installed", False):
        return
    original = quantizer_module.quantize_tiles

    device_luts: dict[tuple[str, int, str], torch.Tensor] = {}
    transposed_sqg_luts: dict[
        tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]
    ] = {}

    def quantize_tiles(tiles: torch.Tensor, quant_args: dict):
        codebook = quant_args.get("sqg_e4m3_lut")
        rate_codebooks = quant_args.get("sqg_e4m3_luts_by_bits")
        mode = quant_args.get("sqg_e4m3_mode")
        if codebook is None and rate_codebooks is None and mode is None:
            return original(tiles, quant_args)
        if len(quant_args["devices"]) != 1:
            raise ValueError("the SQG validation hook currently requires one CUDA device")
        tiles = tiles.contiguous()
        if tiles.dtype != torch.float32 or tiles.ndim != 2 or tiles.shape[1] != 256:
            raise ValueError("SQG tiles must be contiguous FP32 [N, 256]")
        bits = int(quant_args["K"])
        if rate_codebooks is not None:
            if codebook is not None or mode is not None:
                raise ValueError(
                    "rate-specific SQG LUTs cannot be combined with another SQG law"
                )
            if not isinstance(rate_codebooks, Mapping):
                raise TypeError("sqg_e4m3_luts_by_bits must be a mapping")
            if set(rate_codebooks) - {2, 3, 4}:
                raise ValueError("rate-specific SQG LUT keys must be K2, K3, or K4")
            try:
                codebook = rate_codebooks[bits]
            except KeyError as exc:
                raise ValueError(f"missing rate-specific SQG K{bits} LUT") from exc
            if codebook is None:
                output = torch.empty_like(tiles)
                indices = torch.empty_like(tiles, dtype=torch.int16)
                costs, edges = _sqg_temp_buffers(tiles.device, bits)
                _extension().quantize_tiles_procedural(
                    tiles,
                    output,
                    indices,
                    costs,
                    edges,
                    bits,
                    1,
                    int(quant_args.get("tailbite_context", 128)),
                )
                return output, indices
        elif codebook is None:
            if mode != "normal":
                raise ValueError("the supported R44 mode is 'normal'")
            key = (str(tiles.device), bits, mode)
            codebook = device_luts.get(key)
            if codebook is None:
                codebook = sqg_e4m3_bytes(bits, mode, device=tiles.device)
                device_luts[key] = codebook
        output = torch.empty_like(tiles)
        indices = torch.empty_like(tiles, dtype=torch.int16)
        costs, edges = _sqg_temp_buffers(tiles.device, bits)
        lut = codebook.to(device=tiles.device, dtype=torch.uint8).contiguous()
        if bits in (2, 3, 4):
            source_key = codebook.data_ptr() if codebook.is_cuda else id(codebook)
            cache_key = (str(tiles.device), bits, source_key)
            cached = transposed_sqg_luts.get(cache_key)
            if cached is None or cached[0] is not codebook:
                # [predecessor, out-edge-pair, pair-byte] ->
                # [out-edge-pair, predecessor, pair-byte].  Each CUDA thread
                # can then fetch all predecessor labels in one uint2 (K2),
                # one uint4 (K3), or two uint4s (K4), rather than issuing one
                # gather per predecessor.
                predecessors = 1 << bits
                out_edge_pairs = (65536 >> bits) // 2
                lut = (
                    lut.reshape(predecessors, out_edge_pairs, 2)
                    .permute(1, 0, 2)
                    .contiguous()
                    .reshape(-1)
                )
                transposed_sqg_luts[cache_key] = (codebook, lut)
            else:
                lut = cached[1]
        _extension().quantize_tiles_sqg(
            tiles,
            output,
            indices,
            costs,
            edges,
            lut,
            bits,
            int(quant_args.get("tailbite_context", 128)),
        )
        return output, indices

    quantizer_module.quantize_tiles = quantize_tiles
    quantizer_module._kquant_sqg_installed = True
