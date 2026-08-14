"""Immutable contracts and deterministic helpers for the fresh-SQG pilot.

This module is intentionally free of model-loading and encoder imports.  It is
used by preflight, layer workers, artifact validators, and CPU-only tests so
that every stage agrees on the exact four-layer/tensor inventory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import torch

from .glm52_fresh_sqg.manifest import (
    APPROVED_KQUANT_BACKEND_SHA256,
    APPROVED_KQUANT_STATUS_PORCELAIN,
    APPROVED_KQUANT_STATUS_SHA256,
    APPROVED_KQUANT_TRACKED_DIFF_SHA256,
)
from .pilot_config import selected_layers, validated_sha256_env


PIPELINE_SCHEMA = "glm52-fresh-sqg-four-layer-pipeline-v1"
PREFLIGHT_SCHEMA = "glm52-fresh-sqg-preflight-v1"
LAYER_PREP_SCHEMA = "glm52-fresh-sqg-layer-preparation-v1"
PROFILE_SEARCH_SCHEMA = "glm52-fresh-sqg-shared-profile-selection-v1"
EXPERT_ARTIFACT_SCHEMA = "glm52-fresh-sqg-expert-artifact-v1"
LAYER_ARTIFACT_SCHEMA = "glm52-fresh-sqg-layer-artifact-v1"
HOLDOUT_REPORT_SCHEMA = "glm52-fresh-sqg-holdout-routed-functional-v1"
RUN_SEAL_SCHEMA = "glm52-fresh-sqg-four-layer-run-seal-v1"

SOURCE_SEAL_SCHEMA = "glm52-fresh-sqg-bf16-source-v2"
BIT_CONTRACT_SCHEMA = "glm52-fresh-sqg-frozen-per-tensor-bit-allocation-v1"
BIT_CONTRACT_PURPOSE = "topology-neutral per-tensor K3/K4 experimental control"
FROZEN_BIT_CONTRACT_SHA256 = validated_sha256_env(
    "FRESH_SQG_BIT_CONTRACT_SHA256",
    "1fe5a065ef31c2e4c27589415b87bb77a91f095c55eaf4594807ca22645dab33",
)
CAPTURE_SCHEMA = "glm52-fresh-sqg-calibration-capture-v1"

SELECTED_LAYERS = selected_layers()
NUM_EXPERTS = 256
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ROLES = ("fit", "selection", "holdout")
ROLE_TO_ID = {"fit": 0, "selection": 1, "holdout": 2}
HIDDEN = 6_144
INTERMEDIATE = 2_048
TOPK = 8
HADAMARD_BLOCK = 128
TENSORS_PER_LAYER = NUM_EXPERTS * len(PROJECTIONS)
# 3.0625 bpw budget: 720 K3 + 48 K4 over the 768 routed tensors per layer.
# (720*3 + 48*4) / 768 = 2352 / 768 = 3.0625 exactly.  The 384/384 = 3.5 bpw
# budget of the sealed tree is preserved in glm52_fresh_sqg_test and is NOT
# modified; this copy exists solely to encode the lower-rate artifact.
EXPECTED_K3_PER_LAYER = 720
EXPECTED_K4_PER_LAYER = 48
EXPECTED_BIT_UNITS_PER_LAYER = 2_352
KQUANT_REVISION = "104dd9233f850a3955f4991bea68b07dd34deeb8"
SQG_MARKER = 0x53514731
CUDA_CODEC_SMOKE_SHA256 = (
    "0540c59675d8dd42ab96b9109bb0abc8938263cd073b877dc6e5323417744901"
)

# These strings describe disallowed *artifact inputs*.  The isolated KQuant
# source tree naturally contains compatibility code with some of these words;
# it is code provenance, not a model artifact input.
FORBIDDEN_ARTIFACT_SUFFIXES = (
    ".mcg",
    ".mul1",
)
FORBIDDEN_ARTIFACT_FIELDS = (
    "mcg_transform",
    "mcg_scale",
    "mcg_permutation",
    "mcg_seed",
    "mcg_packed_weight",
    "mcg_decoded_weight",
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while payload := handle.read(chunk_bytes):
            digest.update(payload)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: object, *, overwrite: bool = False) -> None:
    """Durably publish canonical JSON without exposing a partial file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    if destination.exists() and not overwrite:
        existing = destination.read_bytes()
        if existing != payload:
            raise FileExistsError(f"refusing to replace bound artifact: {destination}")
        return
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def tensor_prefix(layer: int, expert: int, projection: str) -> str:
    if layer not in SELECTED_LAYERS:
        raise ValueError(f"layer {layer} is outside the frozen pilot")
    if not 0 <= expert < NUM_EXPERTS:
        raise ValueError("expert must lie in [0,255]")
    if projection not in PROJECTIONS:
        raise ValueError(f"unsupported projection {projection!r}")
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}"


def source_tensor_name(layer: int, expert: int, projection: str) -> str:
    return tensor_prefix(layer, expert, projection) + ".weight"


def permutation_scope(layer: int, expert: int) -> str:
    tensor_prefix(layer, expert, "gate_proj")
    return f"layer-{layer:03d}/expert-{expert:03d}"


def derive_seed(*parts: object, bits: int = 63) -> int:
    if not 1 <= bits <= 63:
        raise ValueError("seed width must lie in [1,63]")
    material = canonical_json_bytes(
        {"domain": "glm52-fresh-sqg-seed-v1", "parts": [str(p) for p in parts]}
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") & (
        (1 << bits) - 1
    )


def rademacher(length: int, *parts: object) -> torch.Tensor:
    """Stable SHA-derived signs, independent of PyTorch RNG implementations."""

    if length <= 0:
        raise ValueError("Rademacher length must be positive")
    seed = hashlib.sha256(
        canonical_json_bytes(
            {"domain": "glm52-fresh-sqg-rademacher-v1", "parts": list(map(str, parts))}
        )
    ).digest()
    output = torch.empty(length, dtype=torch.float32)
    offset = 0
    counter = 0
    while offset < length:
        block = hashlib.sha256(seed + counter.to_bytes(8, "little")).digest()
        for byte in block:
            for bit in range(8):
                if offset == length:
                    break
                output[offset] = 1.0 if ((byte >> bit) & 1) else -1.0
                offset += 1
        counter += 1
    return output


def normalized_quarter_scales(values: Sequence[float]) -> torch.Tensor:
    """BMMLaw's bounded, geometric-mean-one quarter-power scale family."""

    if not values:
        raise ValueError("scale evidence must not be empty")
    cleaned = [max(float(value), 1e-20) for value in values]
    if any(not math.isfinite(value) for value in cleaned):
        raise ValueError("scale evidence must be finite")
    geometric = math.exp(sum(math.log(value) for value in cleaned) / len(cleaned))
    scales = [max(0.5, min(2.0, (geometric / value) ** 0.25)) for value in cleaned]
    normalization = math.exp(sum(math.log(value) for value in scales) / len(scales))
    return torch.tensor(
        [value / normalization for value in scales], dtype=torch.float32
    )


def expand_blocks(values: torch.Tensor, size: int) -> torch.Tensor:
    if values.ndim != 1 or values.numel() != size // HADAMARD_BLOCK:
        raise ValueError("block scale geometry differs")
    return values.float().repeat_interleave(HADAMARD_BLOCK).contiguous()


def file_identity(path: str | Path) -> dict[str, int]:
    stat = Path(path).stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


_IGNORED_CODE_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}


def code_tree_manifest(root: str | Path) -> dict[str, object]:
    """Hash every non-cache regular file below one executable-code tree."""

    directory = Path(root).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files: dict[str, dict[str, object]] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if any(part in _IGNORED_CODE_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"executable code tree may not contain symlinks: {path}")
        if not path.is_file() or path.suffix in {".pyc", ".pyo"}:
            continue
        files[relative.as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if not files:
        raise ValueError(f"executable code tree is empty: {directory}")
    value: dict[str, object] = {
        "root": str(directory),
        "file_count": len(files),
        "files": files,
    }
    value["tree_sha256"] = canonical_sha256(value)
    return value


def exllamav3_provenance(root: str | Path) -> dict[str, object]:
    """Seal the complete ExLlama package and mounted precompiled extension."""

    directory = Path(root).resolve()
    package = directory / "exllamav3"
    if not (package / "ext.py").is_file():
        raise FileNotFoundError(f"ExLlamaV3 package lacks ext.py: {package}")
    spec = importlib.util.find_spec("exllamav3_ext")
    if spec is None or spec.origin is None or spec.loader is None:
        raise RuntimeError(
            "precompiled exllamav3_ext is not importable; mount the sealed "
            "extension directory and prepend it to PYTHONPATH"
        )
    extension = Path(spec.origin).resolve()
    if not extension.is_file() or not any(
        str(extension).endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise RuntimeError(f"exllamav3_ext is not a compiled extension: {extension}")
    pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_entries = [
        Path(entry).resolve() for entry in pythonpath.split(os.pathsep) if entry
    ]
    if not pythonpath_entries or pythonpath_entries[0] != extension.parent:
        raise RuntimeError(
            "the sealed exllamav3 extension directory must be the first "
            "PYTHONPATH entry"
        )
    value: dict[str, object] = {
        "root": str(directory),
        "package_tree": code_tree_manifest(package),
        "precompiled_extension": {
            "origin": str(extension),
            "bytes": extension.stat().st_size,
            "sha256": sha256_file(extension),
            "directory_tree": code_tree_manifest(extension.parent),
        },
        "execution_environment": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "torch_hip": torch.version.hip,
            "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
            "CUDAHOSTCXX": os.environ.get("CUDAHOSTCXX"),
            "PYTHONPATH": pythonpath,
        },
        "extension_pythonpath_precedence": True,
        "jit_compilation_allowed": False,
    }
    value["provenance_id"] = canonical_sha256(value)
    return value


def local_pipeline_code_provenance(project_root: str | Path) -> dict[str, object]:
    """Seal all local Python code that can construct or publish artifacts."""

    root = Path(project_root).resolve()
    src = root / "src"
    writer_package = root / "bmmlaw_r7_encoder"
    src_tree = code_tree_manifest(src)
    writer_tree = code_tree_manifest(writer_package)
    # Tests, caches, BF16 shards, capture payloads and output artifacts are not
    # executable construction code and are deliberately outside this seal.
    value: dict[str, object] = {
        "project_root": str(root),
        "src_tree": src_tree,
        "bmmlaw_r7_encoder_tree": writer_tree,
    }
    value["provenance_id"] = canonical_sha256(value)
    return value


@dataclass(frozen=True)
class LayerBitBudget:
    layer: int
    bit_map: Mapping[str, int]
    contract_sha256: str

    def __post_init__(self) -> None:
        expected = {
            tensor_prefix(self.layer, expert, projection)
            for expert in range(NUM_EXPERTS)
            for projection in PROJECTIONS
        }
        if set(self.bit_map) != expected:
            missing = sorted(expected - set(self.bit_map))
            extra = sorted(set(self.bit_map) - expected)
            raise ValueError(
                f"layer {self.layer}: frozen bit-map domain drift: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        values = tuple(self.bit_map.values())
        if any(type(value) is not int or value not in (3, 4) for value in values):
            raise ValueError("frozen bit-map permits only integer K3/K4")
        if (values.count(3), values.count(4), sum(values)) != (
            EXPECTED_K3_PER_LAYER,
            EXPECTED_K4_PER_LAYER,
            EXPECTED_BIT_UNITS_PER_LAYER,
        ):
            raise ValueError(f"layer {self.layer}: exact "
                f"{EXPECTED_K3_PER_LAYER}/{EXPECTED_K4_PER_LAYER} bit budget drift")

    def bits(self, expert: int, projection: str) -> int:
        return int(self.bit_map[tensor_prefix(self.layer, expert, projection)])

    def sanitized_manifest(self) -> dict[str, object]:
        return {
            "controlled_input": "bit_map_only",
            "layer": self.layer,
            "bit_map": dict(sorted(self.bit_map.items())),
            "K3": EXPECTED_K3_PER_LAYER,
            "K4": EXPECTED_K4_PER_LAYER,
            "bit_units": EXPECTED_BIT_UNITS_PER_LAYER,
            "source_sidecar_identity_used": False,
            "contract_sha256": self.contract_sha256,
        }


def load_bit_contract(path: str | Path) -> dict[int, LayerBitBudget]:
    contract_path = Path(path).resolve()
    value = load_json_object(contract_path)
    contract_hash = sha256_file(contract_path)
    if (
        value.get("schema") != BIT_CONTRACT_SCHEMA
        or value.get("purpose") != BIT_CONTRACT_PURPOSE
        or contract_hash != FROZEN_BIT_CONTRACT_SHA256
    ):
        raise ValueError("frozen bit-allocation identity differs")
    if set(value.get("layers", {})) != {str(layer) for layer in SELECTED_LAYERS}:
        raise ValueError(
            f"frozen bit allocation must contain exactly layers {SELECTED_LAYERS}"
        )
    budgets: dict[int, LayerBitBudget] = {}
    for layer in SELECTED_LAYERS:
        raw = value["layers"][str(layer)]
        if not isinstance(raw, dict) or not isinstance(raw.get("bit_map"), dict):
            raise ValueError(f"layer {layer}: invalid bit-map record")
        # Deliberately ignore source_sidecar_sha256.  It authenticated the
        # one-time extraction, but is not an input to this treatment encoder.
        budget = LayerBitBudget(
            layer=layer,
            bit_map={str(key): int(bits) for key, bits in raw["bit_map"].items()},
            contract_sha256=contract_hash,
        )
        histogram = raw.get("histogram")
        if histogram != {"3": EXPECTED_K3_PER_LAYER, "4": EXPECTED_K4_PER_LAYER}:
            raise ValueError(f"layer {layer}: declared bit histogram differs")
        budgets[layer] = budget
    layer_count = len(SELECTED_LAYERS)
    totals = value.get("totals")
    if totals != {
        "K3": layer_count * EXPECTED_K3_PER_LAYER,
        "K4": layer_count * EXPECTED_K4_PER_LAYER,
        "layers": layer_count,
        "tensors": layer_count * TENSORS_PER_LAYER,
    }:
        raise ValueError("selected-layer bit-allocation totals differ")
    return budgets


def git_provenance(root: str | Path) -> dict[str, object]:
    directory = Path(root).resolve()

    def run(*args: str) -> bytes:
        return subprocess.run(
            ["git", *args],
            cwd=directory,
            check=True,
            capture_output=True,
        ).stdout

    revision = run("rev-parse", "HEAD").decode().strip()
    if revision != KQUANT_REVISION:
        raise ValueError(f"KQuant revision drift: {revision} != {KQUANT_REVISION}")
    status = run("status", "--porcelain=v1", "--untracked-files=all").decode()
    diff = run("diff", "--binary", "HEAD", "--")
    status_sha256 = hashlib.sha256(status.encode()).hexdigest()
    tracked_diff_sha256 = hashlib.sha256(diff).hexdigest()
    backend_sha256 = sha256_file(directory / "kquant" / "exl3_encoder_backend.py")
    if (
        status != APPROVED_KQUANT_STATUS_PORCELAIN
        or status_sha256 != APPROVED_KQUANT_STATUS_SHA256
        or tracked_diff_sha256 != APPROVED_KQUANT_TRACKED_DIFF_SHA256
        or backend_sha256 != APPROVED_KQUANT_BACKEND_SHA256
    ):
        raise ValueError("KQuant tree is not the exact reviewed production encoder")
    untracked: dict[str, str] = {}
    for line in status.splitlines():
        if line.startswith("?? "):
            relative = line[3:]
            candidate = directory / relative
            if candidate.is_file():
                untracked[relative] = sha256_file(candidate)
    return {
        "root": str(directory),
        "revision": revision,
        "backend_sha256": backend_sha256,
        "dirty": bool(status),
        "status_sha256": status_sha256,
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked_file_sha256": dict(sorted(untracked.items())),
    }


def assert_no_forbidden_tensor_names(names: Iterable[str]) -> None:
    bad = sorted(
        name for name in names if name.endswith(FORBIDDEN_ARTIFACT_SUFFIXES)
    )
    if bad:
        raise ValueError(f"legacy marker tensors are forbidden: {bad[:4]}")


def assert_role(role: str, expected: str) -> None:
    if role != expected:
        raise ValueError(f"{expected} stage cannot consume role {role!r}")


def validate_layer(layer: int) -> int:
    layer = int(layer)
    if layer not in SELECTED_LAYERS:
        raise ValueError(f"layer must be one of {SELECTED_LAYERS}")
    return layer


def input_allowlist_manifest(
    *,
    source_seal: str | Path,
    capture_manifest: str | Path,
    bit_contract: str | Path,
    kquant_root: str | Path,
    exllamav3_root: str | Path,
    sqg_extension_seal: str | Path,
) -> dict[str, object]:
    """Declare the complete path classes accepted by the worker CLI."""

    return {
        "model_artifact_inputs": {
            "official_bf16_source_seal": str(Path(source_seal).resolve()),
            "fresh_calibration_capture_manifest": str(
                Path(capture_manifest).resolve()
            ),
            "frozen_bit_map_contract": str(Path(bit_contract).resolve()),
        },
        "code_inputs": {
            "kquant_root": str(Path(kquant_root).resolve()),
            "exllamav3_root": str(Path(exllamav3_root).resolve()),
            "sqg_extension_seal": str(Path(sqg_extension_seal).resolve()),
        },
        "accepted_model_artifact_classes": [
            "official_bf16_tensors",
            "fresh_capture_hidden_routes_weights_document_roles",
            "frozen_per_tensor_K3_K4_assignments",
        ],
        "legacy_mcg_model_artifact_paths_accepted": False,
        "forbidden_artifact_fields": list(FORBIDDEN_ARTIFACT_FIELDS),
        "forbidden_artifact_suffixes": list(FORBIDDEN_ARTIFACT_SUFFIXES),
    }
