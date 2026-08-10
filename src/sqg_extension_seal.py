"""Build and validate the production KQuant SQG CUDA extension seal.

The build path is an explicit operator step in the pinned r33 image.  Layer
workers consume only the copied, hash-bound shared object through KQuant's
prebuilt override; they never receive a writable Torch extension cache.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import torch

from .fresh_pipeline_common import (
    atomic_json,
    canonical_sha256,
    code_tree_manifest,
    git_provenance,
    load_json_object,
    sha256_file,
)
from .glm52_fresh_sqg.reference import tensor_sha256


SQG_EXTENSION_SEAL_SCHEMA = "glm52-kquant-sqg-prebuilt-extension-v1"
SQG_EXTENSION_MODULE = "kquant_sqg_quantize_ext_v22"
R33_IMAGE_ID = (
    "sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82"
)
BUILD_CONTRACT = {
    "module_name": SQG_EXTENSION_MODULE,
    "sources": ["sqg_quantize.cpp", "sqg_quantize.cu"],
    "extra_include_path": "kquant/csrc",
    "extra_cflags": ["-O3"],
    "extra_cuda_cflags": [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-Xcudafe",
        "--diag_suppress=177",
        "-Xcudafe",
        "--diag_suppress=20012",
    ],
    "torch_cuda_arch_list": "12.0",
}


def _regular(path: Path, *, label: str) -> Path:
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise ValueError(f"{label} must be an exact non-symlink regular file: {path}")
    return path


def _canonical_id(value: Mapping[str, Any], field: str) -> str:
    expected = value.get(field)
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"{field} is not canonical SHA256")
    body = {key: item for key, item in value.items() if key != field}
    if canonical_sha256(body) != expected:
        raise ValueError(f"canonical {field} binding differs")
    return expected


def _runtime_environment() -> dict[str, object]:
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "torch_hip": torch.version.hip,
        "glibcxx_use_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    }


def _build_environment() -> dict[str, object]:
    def version(*argv: str) -> str:
        completed = subprocess.run(
            list(argv), check=True, capture_output=True, text=True
        )
        return (completed.stdout + completed.stderr).strip()

    value = _runtime_environment()
    value.update(
        {
            "nvcc_version": version("nvcc", "--version"),
            "cxx_version": version(os.environ.get("CXX", "c++"), "--version"),
        }
    )
    return value


def _expected_extension_name() -> str:
    suffix = next(
        (
            item
            for item in importlib.machinery.EXTENSION_SUFFIXES
            if item.startswith(".")
        ),
        ".so",
    )
    return f"{SQG_EXTENSION_MODULE}{suffix}"


def _smoke_extension(module: object) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("SQG extension smoke requires exactly one visible CUDA GPU")
    if torch.cuda.current_device() != 0:
        raise RuntimeError("SQG extension smoke requires visible CUDA device zero")
    sqg = importlib.import_module("kquant.sqg_e4m3")
    quantizer = importlib.import_module("kquant.sqg_quantizer")
    original = quantizer._extension
    quantizer._extension = lambda: module
    quantizer._sqg_temp_buffers.cache_clear()
    probe = type("Probe", (), {"quantize_tiles": lambda *_args: None})()
    quantizer.install_sqg_quantizer(probe)
    records: list[dict[str, object]] = []
    try:
        for bits in (3, 4):
            tiles = torch.linspace(
                -2.0, 2.0, 512, dtype=torch.float32, device="cuda:0"
            ).reshape(2, 256)
            output, indices = probe.quantize_tiles(
                tiles,
                {
                    "K": bits,
                    "devices": ["cuda:0"],
                    "sqg_e4m3_lut": sqg.sqg_xor_cheb_t12_bytes(bits),
                    "tailbite_context": 128,
                },
            )
            torch.cuda.synchronize()
            records.append(
                {
                    "bits": bits,
                    "output_sha256": tensor_sha256(output.cpu()),
                    "indices_sha256": tensor_sha256(indices.cpu()),
                    "finite": bool(torch.isfinite(output).all()),
                }
            )
    finally:
        quantizer._extension = original
        quantizer._sqg_temp_buffers.cache_clear()
    if any(record["finite"] is not True for record in records):
        raise RuntimeError("SQG extension smoke produced a non-finite output")
    receipt: dict[str, object] = {
        "device": "cuda:0",
        "visible_device_count": 1,
        "records": records,
        "passed": True,
    }
    receipt["smoke_id"] = canonical_sha256(receipt)
    return receipt


def validate_sqg_extension_seal(
    path: str | Path,
    *,
    kquant_root: str | Path,
    require_runtime_environment: bool = True,
) -> dict[str, Any]:
    seal_path = _regular(Path(path).absolute(), label="SQG extension seal")
    value = load_json_object(seal_path)
    _canonical_id(value, "seal_id")
    extension = value.get("extension")
    smoke = value.get("smoke")
    if (
        value.get("schema") != SQG_EXTENSION_SEAL_SCHEMA
        or value.get("complete") is not True
        or value.get("module_name") != SQG_EXTENSION_MODULE
        or value.get("build_contract") != BUILD_CONTRACT
        or value.get("kquant") != git_provenance(kquant_root)
        or value.get("csrc_tree")
        != code_tree_manifest(Path(kquant_root).resolve() / "kquant/csrc")
        or value.get("runtime_image_id") != R33_IMAGE_ID
        or value.get("jit_allowed_in_workers") is not False
        or not isinstance(extension, Mapping)
        or not isinstance(smoke, Mapping)
    ):
        raise ValueError("SQG extension construction seal differs")
    extension_path = Path(str(extension.get("path")))
    if (
        not extension_path.is_absolute()
        or extension_path.name != _expected_extension_name()
    ):
        raise ValueError("SQG extension runtime path/name differs")
    _regular(extension_path, label="sealed SQG extension")
    digest = sha256_file(extension_path)
    if (
        extension.get("sha256") != digest
        or extension.get("bytes") != extension_path.stat().st_size
    ):
        raise ValueError("sealed SQG extension bytes differ")
    _canonical_id(smoke, "smoke_id")
    records = smoke.get("records")
    if (
        smoke.get("passed") is not True
        or smoke.get("device") != "cuda:0"
        or smoke.get("visible_device_count") != 1
        or not isinstance(records, list)
        or [record.get("bits") for record in records] != [3, 4]
        or any(
            record.get("finite") is not True
            or not isinstance(record.get("output_sha256"), str)
            or not isinstance(record.get("indices_sha256"), str)
            for record in records
        )
    ):
        raise ValueError("SQG extension K3/K4 smoke receipt differs")
    if require_runtime_environment:
        if os.environ.get("FRESH_SQG_RUNTIME_IMAGE_ID") != R33_IMAGE_ID:
            raise ValueError("SQG worker runtime image binding differs")
        if _runtime_environment() != value.get("runtime_environment"):
            raise ValueError("SQG extension runtime environment differs")
        if os.environ.get("KQUANT_SQG_REQUIRE_PREBUILT") != "1":
            raise ValueError("SQG worker does not forbid KQuant JIT")
        if os.environ.get("KQUANT_SQG_EXTENSION_PATH") != str(extension_path):
            raise ValueError("SQG worker extension override path differs")
        if os.environ.get("KQUANT_SQG_EXTENSION_SHA256") != digest:
            raise ValueError("SQG worker extension override SHA256 differs")
    return value


def build_sqg_extension_seal(
    *,
    kquant_root: str | Path,
    build_root: str | Path,
    output_dir: str | Path,
) -> Path:
    """Compile once in an empty cache, copy, smoke, and seal the exact .so."""

    if os.environ.get("FRESH_SQG_RUNTIME_IMAGE_ID") != R33_IMAGE_ID:
        raise ValueError("extension build must run in the exact pinned r33 image")
    if os.environ.get("TORCH_CUDA_ARCH_LIST") != BUILD_CONTRACT[
        "torch_cuda_arch_list"
    ]:
        raise ValueError("extension build TORCH_CUDA_ARCH_LIST differs")
    for name in (
        "KQUANT_SQG_REQUIRE_PREBUILT",
        "KQUANT_SQG_EXTENSION_PATH",
        "KQUANT_SQG_EXTENSION_SHA256",
    ):
        if os.environ.get(name):
            raise ValueError(f"extension build may not receive {name}")
    root = Path(kquant_root).resolve()
    build = Path(build_root).resolve()
    output = Path(output_dir).resolve()
    if (
        not build.is_dir()
        or build.is_symlink()
        or any(build.iterdir())
        or not output.is_dir()
        or output.is_symlink()
        or any(output.iterdir())
    ):
        raise ValueError("extension build/output directories must be real and empty")
    if build == output or build in output.parents or output in build.parents:
        raise ValueError("extension build and sealed output must be disjoint")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(build)
    sys.path.insert(0, str(root))
    quantizer = importlib.import_module("kquant.sqg_quantizer")
    quantizer._extension.cache_clear()
    module = quantizer._extension()
    built_path = _regular(
        Path(str(module.__file__)).resolve(), label="compiled SQG extension"
    )
    if build not in built_path.parents:
        raise ValueError("compiled SQG extension escaped the dedicated build cache")
    destination = output / _expected_extension_name()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=output
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(built_path, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    digest = sha256_file(destination)
    smoke = _smoke_extension(module)
    build_ninja = built_path.parent / "build.ninja"
    value: dict[str, object] = {
        "schema": SQG_EXTENSION_SEAL_SCHEMA,
        "complete": True,
        "module_name": SQG_EXTENSION_MODULE,
        "runtime_image_id": R33_IMAGE_ID,
        "extension": {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": digest,
        },
        "build_contract": BUILD_CONTRACT,
        "kquant": git_provenance(root),
        "csrc_tree": code_tree_manifest(root / "kquant/csrc"),
        "runtime_environment": _runtime_environment(),
        "compiler_environment": _build_environment(),
        "build_receipt": {
            "torch_extensions_dir": str(build),
            "compiled_origin": str(built_path),
            "build_ninja_sha256": sha256_file(build_ninja),
        },
        "smoke": smoke,
        "jit_allowed_in_workers": False,
    }
    value["seal_id"] = canonical_sha256(value)
    seal_path = output / "sqg-extension-seal.json"
    atomic_json(seal_path, value)
    return seal_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--kquant-root", required=True)
    build.add_argument("--build-root", required=True)
    build.add_argument("--output-dir", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--kquant-root", required=True)
    validate.add_argument("--seal", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_sqg_extension_seal(
            kquant_root=args.kquant_root,
            build_root=args.build_root,
            output_dir=args.output_dir,
        )
        print(json.dumps({"seal": str(result)}, sort_keys=True))
    else:
        result = validate_sqg_extension_seal(
            args.seal, kquant_root=args.kquant_root
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
