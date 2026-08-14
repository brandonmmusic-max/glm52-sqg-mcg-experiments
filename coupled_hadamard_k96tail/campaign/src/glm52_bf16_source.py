"""Fail-closed, streaming access to the sampled official GLM-5.2 BF16 experts.

The provider deliberately knows nothing about the local MCG checkpoint.  It
accepts only the pinned official index/config and the 18 exact official BF16
shards, validates their complete index/header binding without materializing
tensor payloads, and then exposes one fresh expert triplet at a time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import torch

from .pilot_config import selected_layers
from safetensors import safe_open

from .glm52_bf16_manifest import (
    EXPECTED_CONFIG_FILENAME,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_INDEX_BYTES,
    EXPECTED_INDEX_SHA256,
    EXPECTED_INDEX_TENSOR_COUNT,
    EXPECTED_INDEX_TOTAL_BYTES,
    EXPECTED_SHARDS,
    EXPECTED_TOTAL_SHARD_BYTES,
    SOURCE_REPO as SOURCE_REPO,
    SOURCE_REVISION,
    ShardIdentity,
)


SELECTED_LAYERS = selected_layers()
NUM_EXPERTS = 256
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 2048
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

_MAX_SAFETENSORS_HEADER_BYTES = 1 << 30
_FileIdentity = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class _CachedExpert:
    weights: "ExpertWeights"
    tensor_versions: tuple[int, int, int]
    bytes: int


_CUDA_CACHE_LOCK = threading.RLock()
_CUDA_CACHE_NAMESPACE: str | None = None
_CUDA_CACHE: "OrderedDict[tuple[object, ...], _CachedExpert]" = OrderedDict()
_CUDA_CACHE_BYTES = 0
_CUDA_CACHE_HITS = 0
_CUDA_CACHE_MISSES = 0


def _cuda_cache_enabled(device: str | torch.device, dtype: torch.dtype) -> bool:
    try:
        device_type = torch.device(device).type
    except (TypeError, RuntimeError):
        return False
    return bool(
        os.environ.get("B300_PERSISTENT_INPROCESS") == "1"
        and os.environ.get("B300_BF16_CUDA_CACHE") == "1"
        and device_type == "cuda"
        and dtype == torch.bfloat16
    )


def _cache_limit_bytes() -> int:
    raw = os.environ.get("B300_BF16_CUDA_CACHE_GIB", "19.0")
    value = float(raw)
    if not math.isfinite(value) or not 1.0 <= value <= 32.0:
        raise ValueError("B300_BF16_CUDA_CACHE_GIB must lie in [1,32]")
    return int(value * (1 << 30))


def _clear_cuda_cache_locked(*, empty_cuda_allocator: bool) -> None:
    global _CUDA_CACHE_BYTES
    _CUDA_CACHE.clear()
    _CUDA_CACHE_BYTES = 0
    if empty_cuda_allocator and torch.cuda.is_available():
        torch.cuda.empty_cache()


def activate_persistent_cuda_cache(namespace: str) -> dict[str, Any]:
    """Select one resident layer namespace and evict the previous layer once.

    This is called by the persistent worker at an RPC boundary.  Atomic work
    units for the same resident layer reuse exact read-only BF16 CUDA tensors;
    assigning another layer to the slot releases the old layer before loading
    any replacement bytes.
    """

    global _CUDA_CACHE_NAMESPACE
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("persistent BF16 CUDA cache namespace is absent")
    with _CUDA_CACHE_LOCK:
        changed = namespace != _CUDA_CACHE_NAMESPACE
        if changed:
            _clear_cuda_cache_locked(empty_cuda_allocator=True)
            _CUDA_CACHE_NAMESPACE = namespace
        return {
            "activated": True,
            "namespace": _CUDA_CACHE_NAMESPACE,
            "namespace_changed": changed,
            "entries": len(_CUDA_CACHE),
            "bytes": _CUDA_CACHE_BYTES,
            "hits": _CUDA_CACHE_HITS,
            "misses": _CUDA_CACHE_MISSES,
        }


def persistent_cuda_cache_status() -> dict[str, Any]:
    with _CUDA_CACHE_LOCK:
        return {
            "namespace": _CUDA_CACHE_NAMESPACE,
            "entries": len(_CUDA_CACHE),
            "bytes": _CUDA_CACHE_BYTES,
            "hits": _CUDA_CACHE_HITS,
            "misses": _CUDA_CACHE_MISSES,
        }


def deactivate_persistent_cuda_cache(
    *, expected_namespace: str | None = None
) -> dict[str, Any]:
    """Release one persistent slot's live BF16 tensors at a scheduler boundary.

    The supervisor invokes this only while the slot is idle.  The optional
    namespace guard prevents a delayed eviction request from discarding a
    newly reassigned layer's cache.
    """

    global _CUDA_CACHE_NAMESPACE
    with _CUDA_CACHE_LOCK:
        if (
            expected_namespace is not None
            and _CUDA_CACHE_NAMESPACE not in (None, expected_namespace)
        ):
            raise ValueError(
                "persistent BF16 cache namespace changed before eviction: "
                f"{_CUDA_CACHE_NAMESPACE!r} != {expected_namespace!r}"
            )
        previous = _CUDA_CACHE_NAMESPACE
        _clear_cuda_cache_locked(empty_cuda_allocator=True)
        _CUDA_CACHE_NAMESPACE = None
        return {
            "activated": False,
            "evicted_namespace": previous,
            "namespace": None,
            "entries": 0,
            "bytes": 0,
            "hits": _CUDA_CACHE_HITS,
            "misses": _CUDA_CACHE_MISSES,
        }


def sha256_file(path: Path, chunk_bytes: int = 32 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_name(layer: int, expert: int, projection: str) -> str:
    if layer not in SELECTED_LAYERS:
        raise ValueError(f"layer {layer} is outside the frozen pilot")
    if not 0 <= expert < NUM_EXPERTS:
        raise ValueError(f"expert {expert} is outside 0..{NUM_EXPERTS - 1}")
    if projection not in PROJECTIONS:
        raise ValueError(f"unknown projection: {projection}")
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"


def expected_shape(projection: str) -> tuple[int, int]:
    if projection in ("gate_proj", "up_proj"):
        return (INTERMEDIATE_SIZE, HIDDEN_SIZE)
    if projection == "down_proj":
        return (HIDDEN_SIZE, INTERMEDIATE_SIZE)
    raise ValueError(f"unknown projection: {projection}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json_bytes(raw: bytes, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} JSON is not an object")
    return value


def _file_identity(stat: os.stat_result) -> _FileIdentity:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


@dataclass(frozen=True)
class TensorHeader:
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


@dataclass(frozen=True)
class SafeTensorHeader:
    header_sha256: str
    tensors: Mapping[str, TensorHeader]
    file_identity: _FileIdentity


def read_safetensors_header(path: Path) -> SafeTensorHeader:
    """Parse and structurally validate a safetensors header, payload untouched."""

    with path.open("rb") as handle:
        opened_stat = os.fstat(handle.fileno())
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors prefix: {path.name}")
        header_length = struct.unpack("<Q", prefix)[0]
        if (
            header_length > _MAX_SAFETENSORS_HEADER_BYTES
            or header_length > opened_stat.st_size - 8
        ):
            raise ValueError(
                f"implausible safetensors header length in {path.name}: {header_length}"
            )
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"truncated safetensors header: {path.name}")

    header = _load_json_bytes(raw_header, role=f"safetensors header {path.name}")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError(f"invalid safetensors metadata: {path.name}")

    tensors: dict[str, TensorHeader] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, record in header.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ValueError(f"invalid tensor record in {path.name}: {name!r}")
        dtype = record.get("dtype")
        shape_raw = record.get("shape")
        offsets_raw = record.get("data_offsets")
        if not isinstance(dtype, str):
            raise ValueError(f"invalid dtype for {name} in {path.name}")
        if not isinstance(shape_raw, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in shape_raw
        ):
            raise ValueError(f"invalid shape for {name} in {path.name}")
        if (
            not isinstance(offsets_raw, list)
            or len(offsets_raw) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offsets_raw
            )
        ):
            raise ValueError(f"invalid data offsets for {name} in {path.name}")
        start, end = offsets_raw
        if start < 0 or end < start:
            raise ValueError(f"invalid byte range for {name} in {path.name}")
        tensors[name] = TensorHeader(
            dtype=dtype,
            shape=tuple(shape_raw),
            data_offsets=(start, end),
        )
        ranges.append((start, end, name))

    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            relation = "overlap" if start < cursor else "hole"
            raise ValueError(
                f"safetensors data {relation} before {name!r} in {path.name}"
            )
        cursor = end
    if 8 + header_length + cursor != opened_stat.st_size:
        raise ValueError(f"unindexed trailing bytes in {path.name}")

    return SafeTensorHeader(
        header_sha256=hashlib.sha256(raw_header).hexdigest(),
        tensors=MappingProxyType(tensors),
        file_identity=_file_identity(opened_stat),
    )


@dataclass(frozen=True)
class ValidatedShard:
    bytes: int
    sha256: str
    header_sha256: str
    selected_tensor_count: int
    total_tensor_count: int
    file_identity: _FileIdentity


@dataclass(frozen=True)
class SourceValidation:
    index_bytes: int
    index_sha256: str
    config_bytes: int
    config_sha256: str
    weight_map: Mapping[str, str]
    tensor_counts: Mapping[str, int]
    shards: Mapping[str, ValidatedShard]


@dataclass(frozen=True)
class ExpertWeights:
    layer: int
    expert: int
    gate_hf: torch.Tensor
    up_hf: torch.Tensor
    down_hf: torch.Tensor
    tensor_sha256: dict[str, str]
    shard_names: dict[str, str]


@dataclass(frozen=True)
class _SealedSourceCacheEntry:
    source: "BF16ExpertSource"
    index_identity: _FileIdentity
    config_identity: _FileIdentity


_SEALED_SOURCE_CACHE_LOCK = threading.RLock()
_SEALED_SOURCE_CACHE: dict[tuple[object, ...], _SealedSourceCacheEntry] = {}


def _sealed_identity(path: Path, expected: Any) -> _FileIdentity:
    if not isinstance(expected, Mapping):
        raise ValueError(f"sealed BF16 file identity is malformed: {path.name}")
    stat = path.stat()
    observed = {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if observed != dict(expected):
        raise ValueError(f"sealed BF16 file identity drift: {path.name}")
    return _file_identity(stat)


def open_sealed_bf16_source(
    *,
    source_seal: Mapping[str, Any],
    shard_file_identity: Mapping[str, Any],
    cache_namespace: str | None = None,
) -> "BF16ExpertSource":
    """Open a preflight-sealed source with payload hashes reused, not repeated.

    The release preflight already authenticated every shard payload. Paid
    workers bind the exact device/inode/size/mtime tuple and the sealed payload
    digest, then use only cheap stat checks on reuse. Index/config are small and
    are hashed once per persistent process.
    """

    if (
        source_seal.get("repo") != SOURCE_REPO
        or source_seal.get("revision") != SOURCE_REVISION
        or source_seal.get("layers") != list(SELECTED_LAYERS)
        or source_seal.get("complete_index_header_binding_validated") is not True
    ):
        raise ValueError("sealed BF16 source contract differs")
    shards_raw = source_seal.get("shards")
    if not isinstance(shards_raw, Mapping) or set(shards_raw) != set(EXPECTED_SHARDS):
        raise ValueError("sealed BF16 shard domain differs from quartet manifest")
    if set(shard_file_identity) != set(shards_raw):
        raise ValueError("preflight BF16 shard identity domain differs")
    index_record = source_seal.get("index")
    config_record = source_seal.get("config")
    if not isinstance(index_record, Mapping) or not isinstance(config_record, Mapping):
        raise ValueError("sealed BF16 index/config record is malformed")
    index_path = Path(str(index_record.get("path"))).resolve()
    config_path = Path(str(config_record.get("path"))).resolve()
    shard_root = Path(str(source_seal.get("shard_root"))).resolve()
    if not shard_root.is_dir() or index_path.is_symlink() or config_path.is_symlink():
        raise ValueError("sealed BF16 source paths are unsafe")
    shard_identities = tuple(
        (name, _sealed_identity(shard_root / name, shard_file_identity[name]))
        for name in sorted(shards_raw)
    )
    index_stat = _file_identity(index_path.stat())
    config_stat = _file_identity(config_path.stat())
    namespace = cache_namespace or os.environ.get("B300_WEIGHT_CACHE_NAMESPACE", "")
    key = (
        namespace,
        tuple(SELECTED_LAYERS),
        str(index_path),
        index_stat,
        str(config_path),
        config_stat,
        str(shard_root),
        shard_identities,
    )
    if os.environ.get("B300_PERSISTENT_INPROCESS") == "1":
        with _SEALED_SOURCE_CACHE_LOCK:
            cached = _SEALED_SOURCE_CACHE.get(key)
            if cached is not None:
                if _file_identity(index_path.stat()) != cached.index_identity:
                    raise ValueError("sealed BF16 index changed after cached open")
                if _file_identity(config_path.stat()) != cached.config_identity:
                    raise ValueError("sealed BF16 config changed after cached open")
                for shard_name in cached.source.required_shards():
                    cached.source._assert_shard_unchanged(shard_name)
                return cached.source

    index_raw = index_path.read_bytes()
    config_raw = config_path.read_bytes()
    if (
        len(index_raw) != int(index_record.get("bytes", -1))
        or hashlib.sha256(index_raw).hexdigest() != index_record.get("sha256")
        or len(config_raw) != int(config_record.get("bytes", -1))
        or hashlib.sha256(config_raw).hexdigest() != config_record.get("sha256")
    ):
        raise ValueError("sealed BF16 index/config bytes differ")
    index = _load_json_bytes(index_raw, role="official BF16 index")
    config = _load_json_bytes(config_raw, role="official BF16 config")
    _validate_config(config)
    weight_map_raw = index.get("weight_map")
    if not isinstance(weight_map_raw, dict) or not all(
        isinstance(name, str) and isinstance(shard, str)
        for name, shard in weight_map_raw.items()
    ):
        raise ValueError("official BF16 weight map is malformed")
    expected_names = _expected_tensor_names()
    if any(name not in weight_map_raw for name in expected_names):
        raise ValueError("official BF16 index omits a selected expert tensor")
    if {weight_map_raw[name] for name in expected_names} != set(shards_raw):
        raise ValueError("selected BF16 tensors do not bind the sealed shard domain")

    validated_shards: dict[str, ValidatedShard] = {}
    for name, identity in shard_identities:
        record = shards_raw[name]
        expected_record = EXPECTED_SHARDS[name]
        if (
            int(record.get("bytes", -1)) != expected_record.bytes
            or record.get("sha256") != expected_record.sha256
            or record.get("header_sha256") != expected_record.header_sha256
            or int(record.get("total_tensor_count", -1)) != expected_record.tensor_count
        ):
            raise ValueError(f"sealed BF16 shard record differs: {name}")
        validated_shards[name] = ValidatedShard(
            bytes=expected_record.bytes,
            sha256=expected_record.sha256,
            header_sha256=expected_record.header_sha256,
            selected_tensor_count=int(record.get("selected_tensor_count", 0)),
            total_tensor_count=expected_record.tensor_count,
            file_identity=identity,
        )
    validation = SourceValidation(
        index_bytes=len(index_raw),
        index_sha256=hashlib.sha256(index_raw).hexdigest(),
        config_bytes=len(config_raw),
        config_sha256=hashlib.sha256(config_raw).hexdigest(),
        weight_map=MappingProxyType(dict(weight_map_raw)),
        tensor_counts=MappingProxyType(dict(source_seal.get("tensor_counts", {}))),
        shards=MappingProxyType(validated_shards),
    )
    source = BF16ExpertSource.__new__(BF16ExpertSource)
    source.index_path = index_path
    source.shard_root = shard_root
    source.config_path = config_path
    source.validation = validation
    source.weight_map = validation.weight_map
    if os.environ.get("B300_PERSISTENT_INPROCESS") == "1":
        with _SEALED_SOURCE_CACHE_LOCK:
            _SEALED_SOURCE_CACHE[key] = _SealedSourceCacheEntry(
                source=source,
                index_identity=index_stat,
                config_identity=config_stat,
            )
    return source


def _expected_tensor_names() -> set[str]:
    return {
        tensor_name(layer, expert, projection)
        for layer in SELECTED_LAYERS
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    }


def _projection_from_name(name: str) -> str:
    projection = name.rsplit(".", 2)[-2]
    if projection not in PROJECTIONS:
        raise ValueError(f"unrecognized sampled projection in index: {name}")
    return projection


def _validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "model_type": "glm_moe_dsa",
        "hidden_size": HIDDEN_SIZE,
        "moe_intermediate_size": INTERMEDIATE_SIZE,
        "n_routed_experts": NUM_EXPERTS,
        "num_hidden_layers": 78,
    }
    drift = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if drift:
        raise ValueError(f"official config geometry mismatch: {drift}")


def validate_bf16_source(
    *,
    index_path: str | Path,
    shard_root: str | Path,
    config_path: str | Path | None = None,
    source_revision: str = SOURCE_REVISION,
) -> SourceValidation:
    """Validate the complete pinned source identity and selected tensor binding."""

    if source_revision != SOURCE_REVISION:
        raise ValueError(
            f"source revision drift: {source_revision} != {SOURCE_REVISION}"
        )
    index_path = Path(index_path).resolve()
    shard_root = Path(shard_root).resolve()
    config_path = (
        Path(config_path).resolve()
        if config_path is not None
        else index_path.with_name(EXPECTED_CONFIG_FILENAME)
    )
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not shard_root.is_dir():
        raise FileNotFoundError(shard_root)

    index_identity = _file_identity(index_path.stat())
    index_raw = index_path.read_bytes()
    if _file_identity(index_path.stat()) != index_identity:
        raise ValueError("official index changed during validation")
    if len(index_raw) != EXPECTED_INDEX_BYTES:
        raise ValueError(
            f"official index size mismatch: {len(index_raw)} != {EXPECTED_INDEX_BYTES}"
        )
    index_sha256 = hashlib.sha256(index_raw).hexdigest()
    if index_sha256 != EXPECTED_INDEX_SHA256:
        raise ValueError(
            f"official index SHA256 mismatch: {index_sha256} != {EXPECTED_INDEX_SHA256}"
        )
    index = _load_json_bytes(index_raw, role="model index")
    metadata = index.get("metadata")
    if not isinstance(metadata, dict) or (
        metadata.get("total_size") != EXPECTED_INDEX_TOTAL_BYTES
    ):
        raise ValueError("official index total_size mismatch")
    weight_map_raw = index.get("weight_map")
    if not isinstance(weight_map_raw, dict) or any(
        not isinstance(name, str) or not isinstance(shard, str)
        for name, shard in weight_map_raw.items()
    ):
        raise ValueError("official index weight_map is invalid")
    if len(weight_map_raw) != EXPECTED_INDEX_TENSOR_COUNT:
        raise ValueError(
            f"official index tensor count mismatch: {len(weight_map_raw)} != "
            f"{EXPECTED_INDEX_TENSOR_COUNT}"
        )
    weight_map: dict[str, str] = dict(weight_map_raw)

    config_identity = _file_identity(config_path.stat())
    config_raw = config_path.read_bytes()
    if _file_identity(config_path.stat()) != config_identity:
        raise ValueError("official config changed during validation")
    config_sha256 = hashlib.sha256(config_raw).hexdigest()
    if config_sha256 != EXPECTED_CONFIG_SHA256:
        raise ValueError(
            f"official config SHA256 mismatch: {config_sha256} != "
            f"{EXPECTED_CONFIG_SHA256}"
        )
    _validate_config(_load_json_bytes(config_raw, role="model config"))

    expected_names = _expected_tensor_names()
    missing = sorted(expected_names - weight_map.keys())
    if missing:
        raise ValueError(f"BF16 index is missing sampled tensors: {missing[:4]}")
    selected_shards = {weight_map[name] for name in expected_names}
    manifest_shards = set(EXPECTED_SHARDS)
    if selected_shards != manifest_shards:
        missing_shards = sorted(manifest_shards - selected_shards)
        unexpected_shards = sorted(selected_shards - manifest_shards)
        raise ValueError(
            "sampled BF16 shard binding mismatch: "
            f"missing={missing_shards}, unexpected={unexpected_shards}"
        )
    if sum(identity.bytes for identity in EXPECTED_SHARDS.values()) != (
        EXPECTED_TOTAL_SHARD_BYTES
    ):
        raise AssertionError("embedded BF16 shard byte total is inconsistent")

    all_index_names_by_shard: dict[str, set[str]] = {
        shard_name: set() for shard_name in manifest_shards
    }
    for name, shard_name in weight_map.items():
        if shard_name in all_index_names_by_shard:
            all_index_names_by_shard[shard_name].add(name)
    selected_names_by_shard: dict[str, set[str]] = {
        shard_name: set() for shard_name in manifest_shards
    }
    for name in expected_names:
        selected_names_by_shard[weight_map[name]].add(name)

    validated_shards: dict[str, ValidatedShard] = {}
    tensor_counts = {str(layer): 0 for layer in SELECTED_LAYERS}
    for shard_name in sorted(manifest_shards):
        if Path(shard_name).name != shard_name:
            raise ValueError(f"unsafe BF16 shard name in manifest: {shard_name!r}")
        expected_identity: ShardIdentity = EXPECTED_SHARDS[shard_name]
        shard_path = shard_root / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        before = _file_identity(shard_path.stat())
        if before[2] != expected_identity.bytes:
            raise ValueError(
                f"BF16 shard size mismatch: {shard_name}: "
                f"{before[2]} != {expected_identity.bytes}"
            )

        header = read_safetensors_header(shard_path)
        if header.file_identity != before:
            raise ValueError(f"BF16 shard changed during header read: {shard_name}")
        if header.header_sha256 != expected_identity.header_sha256:
            raise ValueError(f"BF16 shard header SHA256 mismatch: {shard_name}")
        if len(header.tensors) != expected_identity.tensor_count:
            raise ValueError(
                f"BF16 shard tensor count mismatch: {shard_name}: "
                f"{len(header.tensors)} != {expected_identity.tensor_count}"
            )

        indexed_names = all_index_names_by_shard[shard_name]
        header_names = set(header.tensors)
        if header_names != indexed_names:
            absent = sorted(indexed_names - header_names)
            extra = sorted(header_names - indexed_names)
            raise ValueError(
                f"complete index/header binding mismatch in {shard_name}: "
                f"missing={absent[:4]}, extra={extra[:4]}"
            )

        for name in sorted(selected_names_by_shard[shard_name]):
            record = header.tensors[name]
            projection = _projection_from_name(name)
            if record.dtype != "BF16":
                raise TypeError(f"{name}: expected BF16, got {record.dtype}")
            shape = expected_shape(projection)
            if record.shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {record.shape}")
            start, end = record.data_offsets
            expected_bytes = math.prod(shape) * 2
            if end - start != expected_bytes:
                raise ValueError(
                    f"{name}: expected {expected_bytes} BF16 payload bytes, "
                    f"got {end - start}"
                )
            layer = name.split(".")[2]
            tensor_counts[layer] += 1

        observed_sha256 = sha256_file(shard_path)
        after = _file_identity(shard_path.stat())
        if after != before:
            raise ValueError(f"BF16 shard changed during hashing: {shard_name}")
        if observed_sha256 != expected_identity.sha256:
            raise ValueError(f"BF16 shard SHA256 mismatch: {shard_name}")
        validated_shards[shard_name] = ValidatedShard(
            bytes=before[2],
            sha256=observed_sha256,
            header_sha256=header.header_sha256,
            selected_tensor_count=len(selected_names_by_shard[shard_name]),
            total_tensor_count=len(header.tensors),
            file_identity=after,
        )

    expected_layer_count = NUM_EXPERTS * len(PROJECTIONS)
    expected_counts = {str(layer): expected_layer_count for layer in SELECTED_LAYERS}
    if tensor_counts != expected_counts:
        raise AssertionError(f"sampled tensor count mismatch: {tensor_counts}")
    if sum(record.bytes for record in validated_shards.values()) != (
        EXPECTED_TOTAL_SHARD_BYTES
    ):
        raise AssertionError("validated BF16 shard byte total is inconsistent")
    if _file_identity(index_path.stat()) != index_identity:
        raise ValueError("official index changed before validation completed")
    if _file_identity(config_path.stat()) != config_identity:
        raise ValueError("official config changed before validation completed")
    for shard_name, record in validated_shards.items():
        if _file_identity((shard_root / shard_name).stat()) != record.file_identity:
            raise ValueError(
                f"BF16 shard changed before validation completed: {shard_name}"
            )

    return SourceValidation(
        index_bytes=len(index_raw),
        index_sha256=index_sha256,
        config_bytes=len(config_raw),
        config_sha256=config_sha256,
        weight_map=MappingProxyType(weight_map),
        tensor_counts=MappingProxyType(tensor_counts),
        shards=MappingProxyType(validated_shards),
    )


class BF16ExpertSource:
    """Validate frozen source shards and load one fresh expert at a time."""

    def __init__(
        self,
        *,
        index_path: str | Path,
        shard_root: str | Path,
        config_path: str | Path | None = None,
        source_revision: str = SOURCE_REVISION,
    ) -> None:
        self.index_path = Path(index_path).resolve()
        self.shard_root = Path(shard_root).resolve()
        self.config_path = (
            Path(config_path).resolve()
            if config_path is not None
            else self.index_path.with_name(EXPECTED_CONFIG_FILENAME)
        )
        self.validation = validate_bf16_source(
            index_path=self.index_path,
            config_path=self.config_path,
            shard_root=self.shard_root,
            source_revision=source_revision,
        )
        self.weight_map = self.validation.weight_map

    def required_shards(self) -> tuple[str, ...]:
        return tuple(sorted(self.validation.shards))

    @staticmethod
    def _tensor_sha256(value: torch.Tensor) -> str:
        raw = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    def _assert_shard_unchanged(self, shard_name: str) -> None:
        shard_path = self.shard_root / shard_name
        try:
            observed = _file_identity(shard_path.stat())
        except FileNotFoundError as exc:
            raise ValueError(
                f"validated BF16 shard disappeared before use: {shard_name}"
            ) from exc
        expected = self.validation.shards[shard_name].file_identity
        if observed != expected:
            raise ValueError(f"validated BF16 shard changed before use: {shard_name}")

    def _load_expert_dtype(
        self,
        layer: int,
        expert: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype,
    ) -> ExpertWeights:
        global _CUDA_CACHE_BYTES, _CUDA_CACHE_HITS, _CUDA_CACHE_MISSES
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("expert source loads permit only preserved BF16 or private FP32")
        cache_enabled = _cuda_cache_enabled(device, dtype)
        cache_key: tuple[object, ...] | None = None
        if cache_enabled:
            namespace = os.environ.get("B300_WEIGHT_CACHE_NAMESPACE")
            if not namespace:
                raise ValueError("persistent CUDA cache requires a namespace")
            requested_device = torch.device(device)
            device_index = (
                torch.cuda.current_device()
                if requested_device.index is None
                else requested_device.index
            )
            cache_key = (
                namespace,
                str(self.index_path),
                self.validation.index_sha256,
                str(self.config_path),
                self.validation.config_sha256,
                str(self.shard_root),
                layer,
                expert,
                str(dtype),
                device_index,
            )
            with _CUDA_CACHE_LOCK:
                if namespace != _CUDA_CACHE_NAMESPACE:
                    raise ValueError(
                        "persistent CUDA cache namespace was not activated at RPC boundary"
                    )
                cached = _CUDA_CACHE.get(cache_key)
                if cached is not None:
                    tensors_cached = (
                        cached.weights.gate_hf,
                        cached.weights.up_hf,
                        cached.weights.down_hf,
                    )
                    observed_versions = tuple(value._version for value in tensors_cached)
                    if observed_versions != cached.tensor_versions:
                        raise RuntimeError(
                            f"cached official BF16 expert L{layer} E{expert} was mutated"
                        )
                    for shard_name in cached.weights.shard_names.values():
                        self._assert_shard_unchanged(shard_name)
                    _CUDA_CACHE.move_to_end(cache_key)
                    _CUDA_CACHE_HITS += 1
                    return ExpertWeights(
                        layer=cached.weights.layer,
                        expert=cached.weights.expert,
                        gate_hf=cached.weights.gate_hf,
                        up_hf=cached.weights.up_hf,
                        down_hf=cached.weights.down_hf,
                        tensor_sha256=dict(cached.weights.tensor_sha256),
                        shard_names=dict(cached.weights.shard_names),
                    )
                _CUDA_CACHE_MISSES += 1
        tensors: dict[str, torch.Tensor] = {}
        hashes: dict[str, str] = {}
        shard_names: dict[str, str] = {}
        for projection in PROJECTIONS:
            name = tensor_name(layer, expert, projection)
            shard_name = self.weight_map[name]
            shard_path = self.shard_root / shard_name
            self._assert_shard_unchanged(shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                value = handle.get_tensor(name)
                if value.dtype != torch.bfloat16:
                    raise TypeError(f"{name}: expected BF16, got {value.dtype}")
                if tuple(value.shape) != expected_shape(projection):
                    raise ValueError(
                        f"{name}: expected {expected_shape(projection)}, "
                        f"got {tuple(value.shape)}"
                    )
                hashes[projection] = self._tensor_sha256(value)
                # Clone while the mapping is open so the returned tensor has
                # no lifetime coupling to the immutable source shard.  The
                # strict bridge path preserves BF16 until its immutable
                # binding has validated the exact official payload.
                tensors[projection] = value.clone().to(device=device, dtype=dtype)
            self._assert_shard_unchanged(shard_name)
            shard_names[projection] = shard_name
        result = ExpertWeights(
            layer=layer,
            expert=expert,
            gate_hf=tensors["gate_proj"],
            up_hf=tensors["up_proj"],
            down_hf=tensors["down_proj"],
            tensor_sha256=hashes,
            shard_names=shard_names,
        )
        if cache_enabled:
            assert cache_key is not None
            cache_bytes = sum(
                value.numel() * value.element_size()
                for value in (result.gate_hf, result.up_hf, result.down_hf)
            )
            cached_result = ExpertWeights(
                layer=result.layer,
                expert=result.expert,
                gate_hf=result.gate_hf,
                up_hf=result.up_hf,
                down_hf=result.down_hf,
                tensor_sha256=dict(result.tensor_sha256),
                shard_names=dict(result.shard_names),
            )
            with _CUDA_CACHE_LOCK:
                _CUDA_CACHE[cache_key] = _CachedExpert(
                    weights=cached_result,
                    tensor_versions=(
                        result.gate_hf._version,
                        result.up_hf._version,
                        result.down_hf._version,
                    ),
                    bytes=cache_bytes,
                )
                _CUDA_CACHE_BYTES += cache_bytes
                limit = _cache_limit_bytes()
                while _CUDA_CACHE_BYTES > limit and len(_CUDA_CACHE) > 1:
                    evicted_key, evicted = _CUDA_CACHE.popitem(last=False)
                    if evicted_key == cache_key:
                        _CUDA_CACHE[evicted_key] = evicted
                        break
                    _CUDA_CACHE_BYTES -= evicted.bytes
                if _CUDA_CACHE_BYTES > limit:
                    raise MemoryError(
                        "one persistent BF16 cache entry exceeds configured cache limit"
                    )
        return result

    def load_expert_bf16(
        self,
        layer: int,
        expert: int,
        *,
        device: str | torch.device = "cpu",
    ) -> ExpertWeights:
        """Load exact official BF16 clones for provenance-bound encoding.

        Conversion to FP32 belongs inside the fresh codec, after its
        :class:`BF16TensorBinding` has checked the BF16 payload hash.  This
        method therefore fails if any future refactor changes the returned
        dtype before that boundary.
        """

        result = self._load_expert_dtype(
            layer,
            expert,
            device=device,
            dtype=torch.bfloat16,
        )
        if any(
            value.dtype != torch.bfloat16
            for value in (result.gate_hf, result.up_hf, result.down_hf)
        ):
            raise AssertionError("strict expert source did not preserve BF16")
        return result

    def load_expert(
        self,
        layer: int,
        expert: int,
        *,
        device: str | torch.device = "cpu",
    ) -> ExpertWeights:
        """Load private FP32 convenience copies for calibration replay."""

        return self._load_expert_dtype(
            layer,
            expert,
            device=device,
            dtype=torch.float32,
        )

    def iter_experts(
        self,
        layer: int,
        *,
        device: str | torch.device = "cpu",
    ) -> Iterator[ExpertWeights]:
        for expert in range(NUM_EXPERTS):
            yield self.load_expert(layer, expert, device=device)
