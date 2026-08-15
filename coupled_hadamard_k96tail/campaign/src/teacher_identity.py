"""Independent, fail-closed identity seal for the capture teacher checkpoint.

The source checkpoint's legacy MANIFEST advertises that payload hashes are not
verified.  This module therefore derives its own exact allowlist from the HF
index and the EXL3 sidecar declaration, hashes every allowed byte, and never
uses MANIFEST.json as an authority.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


RECEIPT_SCHEMA = "glm52-r33-teacher-identity-receipt-v1"
SEAL_SCHEMA = "glm52-r33-teacher-identity-seal-v1"
VALIDATION_SCHEMA = "glm52-r33-teacher-identity-validation-v1"
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"
HASH_ALGORITHM = "sha256"

INDEX_FILE = "model.safetensors.index.json"
IDENTITY_FILES = (
    INDEX_FILE,
    "config.json",
    "quantization_config.json",
    "tier_bitmap.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)
IDENTITY_ROLES = {
    INDEX_FILE: "weight_index",
    "config.json": "model_config",
    "quantization_config.json": "quantization_config",
    "tier_bitmap.json": "tier_bitmap",
    "tokenizer.json": "tokenizer",
    "tokenizer_config.json": "tokenizer_config",
    "chat_template.jinja": "chat_template",
    "generation_config.json": "generation_config",
}
TOKENIZER_ALTERNATIVES = (
    "added_tokens.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.model",
    "vocab.json",
    "vocab.txt",
)
ALTERNATE_WEIGHT_PATTERNS = (
    re.compile(r"^pytorch_model(?:-.*)?\.bin$"),
    re.compile(r"^model(?:-.*)?\.gguf$"),
    re.compile(r"^(?:tf_model\.h5|flax_model\.msgpack)$"),
)
SIDECAR_RE = re.compile(r"^r7-experts-layer-(\d{3})\.json$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _default_payload_files() -> tuple[str, ...]:
    names = ["model-embed.safetensors", "model-head.safetensors"]
    names.extend(f"model-layer-{layer:03d}.safetensors" for layer in range(79))
    names.extend(
        f"r7-experts-layer-{layer:03d}.safetensors"
        for layer in range(3, 78)
    )
    return tuple(sorted(names))


def _default_sidecar_files() -> tuple[str, ...]:
    return tuple(f"r7-experts-layer-{layer:03d}.json" for layer in range(3, 78))


@dataclass(frozen=True)
class TeacherIdentityPolicy:
    """Frozen filename policy; tests may inject a smaller exact fixture."""

    payload_files: tuple[str, ...]
    sidecar_files: tuple[str, ...]
    identity_files: tuple[str, ...] = IDENTITY_FILES


DEFAULT_POLICY = TeacherIdentityPolicy(
    payload_files=_default_payload_files(),
    sidecar_files=_default_sidecar_files(),
)


class TeacherIdentityError(ValueError):
    """The teacher checkpoint or its receipt is unsafe or has drifted."""


@dataclass(frozen=True)
class _Snapshot:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _Digest:
    size: int
    sha256: str
    snapshot: _Snapshot


@dataclass(frozen=True)
class _Inventory:
    payload_files: tuple[str, ...]
    sidecar_files: tuple[str, ...]
    sidecar_shard_hashes: Mapping[str, str]
    file_roles: Mapping[str, str]
    index_weight_count: int
    index_declared_tensor_bytes: int


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _snapshot(value: os.stat_result) -> _Snapshot:
    return _Snapshot(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        size=int(value.st_size),
        mtime_ns=int(value.st_mtime_ns),
        ctime_ns=int(value.st_ctime_ns),
    )


def _absolute_without_resolving(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_safe_root(model_root: str | Path) -> Path:
    root = _absolute_without_resolving(model_root)
    try:
        root_stat = root.lstat()
    except FileNotFoundError as error:
        raise TeacherIdentityError(f"teacher root is absent: {root}") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise TeacherIdentityError(f"teacher root must be a real directory: {root}")
    return root


def _safe_top_level_name(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise TeacherIdentityError(f"{context} is not a safe relative filename")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or len(parsed.parts) != 1
        or parsed.parts[0] in {"", ".", ".."}
        or parsed.as_posix() != value
    ):
        raise TeacherIdentityError(f"{context} is not a safe top-level filename: {value!r}")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise TeacherIdentityError(f"JSON object contains duplicate key {key!r}")
        value[key] = item
    return value


def _read_regular_bytes(path: Path) -> tuple[bytes, _Snapshot]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise TeacherIdentityError(f"cannot safely open regular file: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TeacherIdentityError(f"allowlisted path is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _snapshot(before) != _snapshot(after):
            raise TeacherIdentityError(f"file changed while it was read: {path}")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise TeacherIdentityError(f"short read while reading: {path}")
        snap = _snapshot(after)
    finally:
        os.close(descriptor)
    try:
        current = path.lstat()
    except FileNotFoundError as error:
        raise TeacherIdentityError(f"file disappeared after read: {path}") from error
    if _snapshot(current) != snap or not stat.S_ISREG(current.st_mode):
        raise TeacherIdentityError(f"file identity changed after read: {path}")
    return payload, snap


def _load_json(path: Path) -> object:
    raw, _ = _read_regular_bytes(path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TeacherIdentityError(f"JSON is not UTF-8: {path}") from error
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except TeacherIdentityError:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        raise TeacherIdentityError(f"invalid JSON: {path}: {error}") from error


def _hex64(value: object, *, context: str) -> str:
    if not isinstance(value, str) or HEX64_RE.fullmatch(value) is None:
        raise TeacherIdentityError(f"{context} is not lowercase SHA256")
    return value


def _validate_policy(policy: TeacherIdentityPolicy) -> None:
    for label, values in (
        ("payload", policy.payload_files),
        ("sidecar", policy.sidecar_files),
        ("identity", policy.identity_files),
    ):
        if not values or len(set(values)) != len(values):
            raise TeacherIdentityError(f"{label} policy is empty or has duplicates")
        for value in values:
            _safe_top_level_name(value, context=f"{label} policy entry")
    if set(policy.payload_files) & set(policy.sidecar_files):
        raise TeacherIdentityError("payload and sidecar policies overlap")
    if set(policy.payload_files) & set(policy.identity_files):
        raise TeacherIdentityError("payload and identity policies overlap")
    if set(policy.sidecar_files) & set(policy.identity_files):
        raise TeacherIdentityError("sidecar and identity policies overlap")
    if set(policy.identity_files) != set(IDENTITY_FILES):
        raise TeacherIdentityError("teacher loader identity policy differs from v1")


def _scan_for_ambiguity(root: Path, policy: TeacherIdentityPolicy) -> None:
    observed_payloads: set[str] = set()
    observed_sidecars: set[str] = set()
    observed_identity: set[str] = set()
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        for name in tuple(directories) + tuple(files):
            path = directory_path / name
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                raise TeacherIdentityError(f"teacher tree contains symlink: {relative}")
        for name in files:
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            if name.endswith(".safetensors"):
                observed_payloads.add(relative)
            if len(PurePosixPath(relative).parts) == 1:
                if SIDECAR_RE.fullmatch(name):
                    observed_sidecars.add(name)
                if name in set(policy.identity_files) | set(TOKENIZER_ALTERNATIVES):
                    observed_identity.add(name)
                if name.endswith(".py"):
                    raise TeacherIdentityError(
                        f"unsealed top-level remote-code candidate is present: {name}"
                    )
                if any(pattern.fullmatch(name) for pattern in ALTERNATE_WEIGHT_PATTERNS):
                    raise TeacherIdentityError(
                        f"unsealed alternate weight payload is present: {name}"
                    )
    expected_payloads = set(policy.payload_files)
    if observed_payloads != expected_payloads:
        raise TeacherIdentityError(
            "safetensors allowlist differs; "
            f"missing={sorted(expected_payloads - observed_payloads)} "
            f"unreferenced={sorted(observed_payloads - expected_payloads)}"
        )
    expected_sidecars = set(policy.sidecar_files)
    if observed_sidecars != expected_sidecars:
        raise TeacherIdentityError(
            "R7 loader sidecar allowlist differs; "
            f"missing={sorted(expected_sidecars - observed_sidecars)} "
            f"unreferenced={sorted(observed_sidecars - expected_sidecars)}"
        )
    if observed_identity != set(policy.identity_files):
        raise TeacherIdentityError(
            "loader/tokenizer identity allowlist differs; "
            f"missing={sorted(set(policy.identity_files) - observed_identity)} "
            f"ambiguous={sorted(observed_identity - set(policy.identity_files))}"
        )


def _require_object(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TeacherIdentityError(f"{context} must be a JSON object")
    return value


def _inspect_model(root: Path, policy: TeacherIdentityPolicy) -> _Inventory:
    _validate_policy(policy)
    _scan_for_ambiguity(root, policy)

    index = _require_object(_load_json(root / INDEX_FILE), context=INDEX_FILE)
    if set(index) != {"metadata", "weight_map"}:
        raise TeacherIdentityError("weight index must contain exactly metadata and weight_map")
    metadata = _require_object(index["metadata"], context="weight index metadata")
    if set(metadata) != {"total_size"}:
        raise TeacherIdentityError("weight index metadata schema differs")
    declared_bytes = metadata["total_size"]
    if type(declared_bytes) is not int or declared_bytes <= 0:
        raise TeacherIdentityError("weight index total_size is not a positive JSON integer")
    weight_map = _require_object(index["weight_map"], context="weight index weight_map")
    if not weight_map:
        raise TeacherIdentityError("weight index weight_map is empty")
    referenced: set[str] = set()
    for tensor_name, filename in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise TeacherIdentityError("weight index contains an invalid tensor name")
        referenced.add(
            _safe_top_level_name(filename, context=f"index mapping for {tensor_name}")
        )
    if referenced != set(policy.payload_files):
        raise TeacherIdentityError(
            "index-referenced payload allowlist differs; "
            f"missing={sorted(set(policy.payload_files) - referenced)} "
            f"unexpected={sorted(referenced - set(policy.payload_files))}"
        )

    config = _require_object(_load_json(root / "config.json"), context="config.json")
    quantization = _require_object(
        _load_json(root / "quantization_config.json"),
        context="quantization_config.json",
    )
    if config.get("quantization_config") != quantization:
        raise TeacherIdentityError(
            "config.json embedded quantization_config differs from quantization_config.json"
        )
    tail = _require_object(config.get("hybrid_tr3_tail"), context="hybrid_tr3_tail")
    if tail.get("tier_bitmap") != "tier_bitmap.json":
        raise TeacherIdentityError("config does not bind the exact tier_bitmap.json")
    routed = _require_object(
        quantization.get("r7_routed_experts"),
        context="r7_routed_experts",
    )
    manifests = routed.get("bit_map_manifests")
    if not isinstance(manifests, list):
        raise TeacherIdentityError("bit_map_manifests must be a JSON list")
    manifest_names = tuple(
        _safe_top_level_name(item, context="bit_map_manifests entry")
        for item in manifests
    )
    if manifest_names != policy.sidecar_files or len(set(manifest_names)) != len(manifest_names):
        raise TeacherIdentityError("quantization config sidecar allowlist/order differs")

    # Parse every JSON loader identity file strictly, even when its internal
    # schema is owned by Transformers rather than this receipt.
    for name in policy.identity_files:
        if name.endswith(".json") and name not in {
            INDEX_FILE,
            "config.json",
            "quantization_config.json",
        }:
            _load_json(root / name)

    sidecar_shard_hashes: dict[str, str] = {}
    for sidecar_name in policy.sidecar_files:
        match = SIDECAR_RE.fullmatch(sidecar_name)
        if match is None:
            raise TeacherIdentityError(f"invalid R7 sidecar policy name: {sidecar_name}")
        layer = int(match.group(1))
        value = _require_object(_load_json(root / sidecar_name), context=sidecar_name)
        if type(value.get("layer")) is not int or value.get("layer") != layer:
            raise TeacherIdentityError(f"R7 sidecar layer differs: {sidecar_name}")
        if value.get("schema_version") != 2:
            raise TeacherIdentityError(f"R7 sidecar schema version differs: {sidecar_name}")
        shard = _safe_top_level_name(
            value.get("shard"), context=f"{sidecar_name} shard"
        )
        expected_shard = sidecar_name.removesuffix(".json") + ".safetensors"
        if shard != expected_shard or shard not in referenced:
            raise TeacherIdentityError(f"R7 sidecar shard binding differs: {sidecar_name}")
        sidecar_shard_hashes[shard] = _hex64(
            value.get("shard_sha256"), context=f"{sidecar_name} shard_sha256"
        )

    roles: dict[str, str] = {}
    for name in policy.payload_files:
        roles[name] = "indexed_payload"
    for name in policy.sidecar_files:
        roles[name] = "r7_loader_sidecar"
    for name in policy.identity_files:
        roles[name] = IDENTITY_ROLES[name]
    return _Inventory(
        payload_files=tuple(sorted(referenced)),
        sidecar_files=manifest_names,
        sidecar_shard_hashes=sidecar_shard_hashes,
        file_roles=roles,
        index_weight_count=len(weight_map),
        index_declared_tensor_bytes=int(declared_bytes),
    )


def _hash_regular_file(root: Path, name: str) -> _Digest:
    path = root / name
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise TeacherIdentityError(f"cannot safely hash file {name}: {error}") from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TeacherIdentityError(f"allowlisted path is not regular: {name}")
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_SEQUENTIAL"):
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_SEQUENTIAL)
            except OSError:
                pass
        read_bytes = 0
        while True:
            chunk = os.read(descriptor, 16 * 1024 * 1024)
            if not chunk:
                break
            read_bytes += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _snapshot(before) != _snapshot(after) or read_bytes != before.st_size:
            raise TeacherIdentityError(f"file changed or short-read while hashing: {name}")
        snap = _snapshot(after)
    finally:
        os.close(descriptor)
    current = path.lstat()
    if not stat.S_ISREG(current.st_mode) or _snapshot(current) != snap:
        raise TeacherIdentityError(f"file identity changed after hashing: {name}")
    return _Digest(size=snap.size, sha256=digest.hexdigest(), snapshot=snap)


def _hash_files(root: Path, names: Iterable[str], *, workers: int) -> dict[str, _Digest]:
    ordered = tuple(sorted(names))
    if type(workers) is not int or not 1 <= workers <= 32:
        raise TeacherIdentityError("hash workers must be an integer in 1..32")
    if workers == 1:
        values = [_hash_regular_file(root, name) for name in ordered]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="teacher-sha256") as pool:
            values = list(pool.map(lambda name: _hash_regular_file(root, name), ordered))
    results = dict(zip(ordered, values, strict=True))
    for name, value in results.items():
        current = (root / name).lstat()
        if not stat.S_ISREG(current.st_mode) or _snapshot(current) != value.snapshot:
            raise TeacherIdentityError(f"file changed before hash set closed: {name}")
    return results


def _make_seal(
    inventory: _Inventory,
    digests: Mapping[str, _Digest],
    policy: TeacherIdentityPolicy,
) -> dict[str, object]:
    if set(digests) != set(inventory.file_roles):
        raise TeacherIdentityError("internal hash allowlist differs from inventory")
    for shard, expected in inventory.sidecar_shard_hashes.items():
        if digests[shard].sha256 != expected:
            raise TeacherIdentityError(
                f"independent shard SHA256 differs from R7 sidecar claim: {shard}"
            )
    files = [
        {
            "path": name,
            "role": inventory.file_roles[name],
            "bytes": digests[name].size,
            "sha256": digests[name].sha256,
        }
        for name in sorted(digests)
    ]
    payload_bytes = sum(digests[name].size for name in policy.payload_files)
    return {
        "schema": SEAL_SCHEMA,
        "hash_algorithm": HASH_ALGORITHM,
        "canonicalization": CANONICALIZATION,
        "index_filename": INDEX_FILE,
        "index_weight_count": inventory.index_weight_count,
        "index_declared_tensor_bytes": inventory.index_declared_tensor_bytes,
        "payload_count": len(policy.payload_files),
        "loader_sidecar_count": len(policy.sidecar_files),
        "loader_identity_count": len(policy.identity_files),
        "total_file_count": len(files),
        "payload_bytes": payload_bytes,
        "total_bytes": sum(item["bytes"] for item in files),
        "payload_files": list(policy.payload_files),
        "loader_sidecars": list(policy.sidecar_files),
        "loader_identity_files": list(policy.identity_files),
        "files": files,
    }


def _atomic_create_once(path: str | Path, payload: bytes) -> Path:
    target = _absolute_without_resolving(path)
    parent = target.parent
    parent_stat = parent.lstat()
    if not stat.S_ISDIR(parent_stat.st_mode) or stat.S_ISLNK(parent_stat.st_mode):
        raise TeacherIdentityError(f"output parent is not a real directory: {parent}")
    if target.exists() or target.is_symlink():
        raise TeacherIdentityError(f"refusing to overwrite existing receipt/evidence: {target}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise TeacherIdentityError(f"output appeared concurrently: {target}") from error
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def build_teacher_identity_receipt(
    model_root: str | Path,
    receipt_path: str | Path,
    *,
    workers: int = 4,
    policy: TeacherIdentityPolicy = DEFAULT_POLICY,
) -> dict[str, object]:
    """Hash the exact teacher allowlist and atomically create one receipt."""

    target = _absolute_without_resolving(receipt_path)
    if target.exists() or target.is_symlink():
        raise TeacherIdentityError(f"refusing to overwrite existing receipt: {target}")
    root = _require_safe_root(model_root)
    inventory = _inspect_model(root, policy)
    digests = _hash_files(root, inventory.file_roles, workers=workers)
    _scan_for_ambiguity(root, policy)
    seal = _make_seal(inventory, digests, policy)
    seal_sha256 = hashlib.sha256(canonical_json_bytes(seal)).hexdigest()
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "seal": seal,
        "seal_sha256": seal_sha256,
    }
    _atomic_create_once(target, canonical_json_bytes(receipt) + b"\n")
    return validate_teacher_identity_receipt(
        root,
        target,
        expected_seal_sha256=seal_sha256,
        verify_mode="metadata",
        workers=workers,
        policy=policy,
    )


_SEAL_KEYS = {
    "schema",
    "hash_algorithm",
    "canonicalization",
    "index_filename",
    "index_weight_count",
    "index_declared_tensor_bytes",
    "payload_count",
    "loader_sidecar_count",
    "loader_identity_count",
    "total_file_count",
    "payload_bytes",
    "total_bytes",
    "payload_files",
    "loader_sidecars",
    "loader_identity_files",
    "files",
}


def _validate_receipt_shape(
    receipt: object,
    policy: TeacherIdentityPolicy,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    receipt_object = _require_object(receipt, context="teacher identity receipt")
    if set(receipt_object) != {"schema", "seal", "seal_sha256"}:
        raise TeacherIdentityError("teacher identity receipt fields differ")
    if receipt_object.get("schema") != RECEIPT_SCHEMA:
        raise TeacherIdentityError("teacher identity receipt schema differs")
    seal = _require_object(receipt_object.get("seal"), context="teacher identity seal")
    if set(seal) != _SEAL_KEYS:
        raise TeacherIdentityError("teacher identity seal fields differ")
    if (
        seal.get("schema") != SEAL_SCHEMA
        or seal.get("hash_algorithm") != HASH_ALGORITHM
        or seal.get("canonicalization") != CANONICALIZATION
        or seal.get("index_filename") != INDEX_FILE
    ):
        raise TeacherIdentityError("teacher identity seal constants differ")
    observed_seal_sha = hashlib.sha256(canonical_json_bytes(seal)).hexdigest()
    if receipt_object.get("seal_sha256") != observed_seal_sha:
        raise TeacherIdentityError("teacher identity canonical seal digest differs")
    for key in (
        "index_weight_count",
        "index_declared_tensor_bytes",
        "payload_count",
        "loader_sidecar_count",
        "loader_identity_count",
        "total_file_count",
        "payload_bytes",
        "total_bytes",
    ):
        if type(seal.get(key)) is not int or seal[key] <= 0:
            raise TeacherIdentityError(f"teacher identity seal integer differs: {key}")
    if seal.get("payload_files") != list(policy.payload_files):
        raise TeacherIdentityError("receipt payload allowlist differs from policy")
    if seal.get("loader_sidecars") != list(policy.sidecar_files):
        raise TeacherIdentityError("receipt sidecar allowlist differs from policy")
    if seal.get("loader_identity_files") != list(policy.identity_files):
        raise TeacherIdentityError("receipt loader identity allowlist differs from policy")
    if seal.get("payload_count") != len(policy.payload_files):
        raise TeacherIdentityError("receipt payload count differs")
    if seal.get("loader_sidecar_count") != len(policy.sidecar_files):
        raise TeacherIdentityError("receipt sidecar count differs")
    if seal.get("loader_identity_count") != len(policy.identity_files):
        raise TeacherIdentityError("receipt loader identity count differs")

    raw_files = seal.get("files")
    if not isinstance(raw_files, list):
        raise TeacherIdentityError("receipt files must be a JSON list")
    files: dict[str, dict[str, Any]] = {}
    paths: list[str] = []
    expected_names = set(policy.payload_files) | set(policy.sidecar_files) | set(
        policy.identity_files
    )
    for raw in raw_files:
        item = _require_object(raw, context="receipt file entry")
        if set(item) != {"path", "role", "bytes", "sha256"}:
            raise TeacherIdentityError("receipt file entry fields differ")
        name = _safe_top_level_name(item.get("path"), context="receipt file path")
        if name in files:
            raise TeacherIdentityError(f"receipt contains duplicate file entry: {name}")
        if type(item.get("bytes")) is not int or item["bytes"] <= 0:
            raise TeacherIdentityError(f"receipt file byte count differs: {name}")
        _hex64(item.get("sha256"), context=f"receipt {name} SHA256")
        expected_role = (
            "indexed_payload"
            if name in policy.payload_files
            else "r7_loader_sidecar"
            if name in policy.sidecar_files
            else IDENTITY_ROLES.get(name)
        )
        if item.get("role") != expected_role:
            raise TeacherIdentityError(f"receipt file role differs: {name}")
        files[name] = item
        paths.append(name)
    if paths != sorted(paths) or set(files) != expected_names:
        raise TeacherIdentityError("receipt file entry allowlist/order differs")
    if seal.get("total_file_count") != len(files):
        raise TeacherIdentityError("receipt total file count differs")
    if seal.get("total_bytes") != sum(item["bytes"] for item in files.values()):
        raise TeacherIdentityError("receipt total byte count differs")
    if seal.get("payload_bytes") != sum(
        files[name]["bytes"] for name in policy.payload_files
    ):
        raise TeacherIdentityError("receipt payload byte count differs")
    return seal, files


def _receipt_bytes_and_object(receipt_path: str | Path) -> tuple[bytes, object]:
    path = _absolute_without_resolving(receipt_path)
    raw, _ = _read_regular_bytes(path)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except TeacherIdentityError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise TeacherIdentityError(f"invalid teacher identity receipt: {path}") from error
    return raw, value


def _validation_record(
    *,
    receipt_sha256: str,
    seal: Mapping[str, Any],
    verify_mode: str,
) -> dict[str, object]:
    full = verify_mode == "full"
    return {
        "schema": VALIDATION_SCHEMA,
        "receipt_sha256": receipt_sha256,
        "seal_sha256": hashlib.sha256(canonical_json_bytes(seal)).hexdigest(),
        "verification_mode": verify_mode,
        "all_file_bytes_sha256_validated": full,
        "non_payload_file_bytes_sha256_validated": True,
        "payload_count": seal["payload_count"],
        "loader_sidecar_count": seal["loader_sidecar_count"],
        "loader_identity_count": seal["loader_identity_count"],
        "total_file_count": seal["total_file_count"],
        "payload_bytes": seal["payload_bytes"],
        "total_bytes": seal["total_bytes"],
        "index_weight_count": seal["index_weight_count"],
        "index_declared_tensor_bytes": seal["index_declared_tensor_bytes"],
    }


def validate_teacher_identity_receipt(
    model_root: str | Path,
    receipt_path: str | Path,
    *,
    expected_seal_sha256: str | None = None,
    verify_mode: str = "full",
    workers: int = 4,
    policy: TeacherIdentityPolicy = DEFAULT_POLICY,
) -> dict[str, object]:
    """Validate the receipt and current checkpoint, failing on any drift.

    ``full`` independently hashes all payload and loader bytes. ``metadata``
    still hashes all non-payload loader/sidecar bytes and validates every
    payload path and size; it is intended only for the in-container check after
    the launcher's full preflight has produced a bound attestation.
    """

    if verify_mode not in {"full", "metadata"}:
        raise TeacherIdentityError("verify_mode must be 'full' or 'metadata'")
    _validate_policy(policy)
    if expected_seal_sha256 is not None:
        expected_seal_sha256 = _hex64(
            expected_seal_sha256, context="expected teacher seal SHA256"
        )
    receipt_raw, receipt_object = _receipt_bytes_and_object(receipt_path)
    seal, files = _validate_receipt_shape(receipt_object, policy)
    if expected_seal_sha256 is not None and (
        receipt_object["seal_sha256"] != expected_seal_sha256
    ):
        raise TeacherIdentityError("teacher identity seal differs from frozen expectation")

    root = _require_safe_root(model_root)
    inventory = _inspect_model(root, policy)
    if inventory.index_weight_count != seal["index_weight_count"]:
        raise TeacherIdentityError("current weight index tensor count differs from receipt")
    if inventory.index_declared_tensor_bytes != seal["index_declared_tensor_bytes"]:
        raise TeacherIdentityError("current weight index total_size differs from receipt")
    for name, item in files.items():
        try:
            current = (root / name).lstat()
        except FileNotFoundError as error:
            raise TeacherIdentityError(f"receipt file is absent: {name}") from error
        if (
            not stat.S_ISREG(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or current.st_size != item["bytes"]
        ):
            raise TeacherIdentityError(f"receipt file type/size differs: {name}")
    for shard, claimed_sha in inventory.sidecar_shard_hashes.items():
        if files[shard]["sha256"] != claimed_sha:
            raise TeacherIdentityError(
                f"receipt and R7 sidecar shard SHA256 disagree: {shard}"
            )

    names_to_hash: Sequence[str]
    if verify_mode == "full":
        names_to_hash = tuple(files)
    else:
        names_to_hash = tuple(
            name for name in files if name not in set(policy.payload_files)
        )
    observed = _hash_files(root, names_to_hash, workers=workers)
    for name, digest in observed.items():
        if digest.size != files[name]["bytes"] or digest.sha256 != files[name]["sha256"]:
            raise TeacherIdentityError(f"teacher file SHA256 differs from receipt: {name}")
    _scan_for_ambiguity(root, policy)
    return _validation_record(
        receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        seal=seal,
        verify_mode=verify_mode,
    )


def validate_teacher_identity_attestation(
    attestation_path: str | Path,
    model_root: str | Path,
    receipt_path: str | Path,
    *,
    expected_seal_sha256: str,
    workers: int = 4,
    policy: TeacherIdentityPolicy = DEFAULT_POLICY,
) -> dict[str, object]:
    """Bind a launcher-produced full attestation to an in-container check."""

    attestation = _require_object(
        _load_json(_absolute_without_resolving(attestation_path)),
        context="teacher identity preflight attestation",
    )
    metadata = validate_teacher_identity_receipt(
        model_root,
        receipt_path,
        expected_seal_sha256=expected_seal_sha256,
        verify_mode="metadata",
        workers=workers,
        policy=policy,
    )
    expected = dict(metadata)
    expected["verification_mode"] = "full"
    expected["all_file_bytes_sha256_validated"] = True
    if attestation != expected:
        raise TeacherIdentityError(
            "launcher full-hash attestation does not bind to current receipt/metadata"
        )
    return dict(attestation)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="create the immutable receipt once")
    build.add_argument("--model-root", type=Path, required=True)
    build.add_argument("--receipt", type=Path, required=True)
    build.add_argument("--workers", type=int, default=4)
    validate = commands.add_parser("validate", help="fail-closed receipt validation")
    validate.add_argument("--model-root", type=Path, required=True)
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--expected-seal-sha256")
    validate.add_argument("--verify-mode", choices=("full", "metadata"), default="full")
    validate.add_argument("--workers", type=int, default=4)
    validate.add_argument("--attestation-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_teacher_identity_receipt(
                args.model_root,
                args.receipt,
                workers=args.workers,
            )
        else:
            result = validate_teacher_identity_receipt(
                args.model_root,
                args.receipt,
                expected_seal_sha256=args.expected_seal_sha256,
                verify_mode=args.verify_mode,
                workers=args.workers,
            )
            if args.attestation_out is not None:
                if args.verify_mode != "full":
                    raise TeacherIdentityError(
                        "--attestation-out requires --verify-mode full"
                    )
                _atomic_create_once(
                    args.attestation_out,
                    canonical_json_bytes(result) + b"\n",
                )
    except (OSError, TeacherIdentityError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
