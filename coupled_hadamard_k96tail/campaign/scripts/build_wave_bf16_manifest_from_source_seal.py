#!/usr/bin/env python3
"""Derive the legacy import-time BF16 manifest from a sealed wave record.

This reads only the already-published hashes and structural metadata in the
wave source seal.  It never opens or requires the original BF16 shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCHEMA = "glm52-fresh-sqg-bf16-shard-manifest-v1"
SOURCE_SCHEMA = "glm52-fresh-sqg-bf16-source-v2"
REPO = "zai-org/GLM-5.2"
REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
INDEX_SHA256 = "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e"


def require_digest(value: object, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} is not a lowercase SHA256 digest")
    return digest


def atomic_publish(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError(f"existing derived wave manifest differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-seal", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source_seal.resolve(strict=True)
    if args.source_seal.is_symlink() or not source_path.is_file():
        raise ValueError("source seal must be a plain file")
    layers = tuple(args.layers)
    terminal_subset = layers == (75, 76, 77)
    if (
        (len(layers) != 4 and not terminal_subset)
        or tuple(sorted(layers)) != layers
        or len(set(layers)) != len(layers)
        or any(layer < 3 or layer > 78 for layer in layers)
    ):
        raise ValueError(
            "layers must be four unique ascending routed layers or terminal 75,76,77"
        )

    source_raw = source_path.read_bytes()
    source = json.loads(source_raw)
    if (
        not isinstance(source, dict)
        or source.get("schema") != SOURCE_SCHEMA
        or source.get("repo") != REPO
        or source.get("revision") != REVISION
        or source.get("layers")
        != ([74, 75, 76, 77] if terminal_subset else list(layers))
        or not isinstance(source.get("index"), dict)
        or source["index"].get("sha256") != INDEX_SHA256
        or source.get("complete_index_header_binding_validated") is not True
    ):
        raise ValueError("wave source-seal identity or layer binding differs")
    shards = source.get("shards")
    if not isinstance(shards, dict) or not shards or source.get("shard_count") != len(shards):
        raise ValueError("wave source seal has an invalid shard census")

    records: dict[str, dict[str, object]] = {}
    selected_tensor_count = 0
    for name, record in sorted(shards.items()):
        if not isinstance(name, str) or Path(name).name != name or not isinstance(record, dict):
            raise ValueError(f"unsafe wave source-shard record: {name!r}")
        byte_count = int(record.get("bytes", 0))
        tensor_count = int(record.get("total_tensor_count", 0))
        selected_count = int(record.get("selected_tensor_count", 0))
        if byte_count <= 0 or tensor_count <= 0 or not 0 < selected_count <= tensor_count:
            raise ValueError(f"invalid wave source-shard census: {name}")
        records[name] = {
            "bytes": byte_count,
            "sha256": require_digest(record.get("sha256"), f"{name} payload"),
            "header_sha256": require_digest(
                record.get("header_sha256"), f"{name} header"
            ),
            "tensor_count": tensor_count,
        }
        selected_tensor_count += selected_count

    source_layers = tuple(int(layer) for layer in source["layers"])
    source_expected_selected = len(source_layers) * 256 * 3
    expected_selected = len(layers) * 256 * 3
    total_bytes = sum(int(record["bytes"]) for record in records.values())
    if (
        selected_tensor_count != source_expected_selected
        or source.get("target_tensor_count") != source_expected_selected
        or source.get("total_shard_bytes") != total_bytes
    ):
        raise ValueError("wave source-seal tensor or byte census differs")

    manifest = {
        "schema": SCHEMA,
        "repo": REPO,
        "revision": REVISION,
        "index_sha256": INDEX_SHA256,
        "layers": list(layers),
        "selected_tensor_count": expected_selected,
        "shards": records,
        "total_shard_bytes": total_bytes,
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output = args.output.resolve()
    atomic_publish(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "source_seal_sha256": hashlib.sha256(source_raw).hexdigest(),
                "layers": list(layers),
                "shards": len(records),
                "bytes": total_bytes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
