#!/usr/bin/env python3
"""Seal the exact official BF16 shards needed by the active four layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path


INDEX_SHA256 = "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e"
REPO = "zai-org/GLM-5.2"
REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(64 << 20):
            digest.update(block)
    return digest.hexdigest()


def shard_header(path: Path) -> tuple[str, int]:
    with path.open("rb") as handle:
        length_raw = handle.read(8)
        if len(length_raw) != 8:
            raise ValueError(f"{path}: truncated safetensors prefix")
        length = struct.unpack("<Q", length_raw)[0]
        raw = handle.read(length)
    if len(raw) != length:
        raise ValueError(f"{path}: truncated safetensors header")
    parsed = json.loads(raw)
    tensors = [name for name in parsed if name != "__metadata__"]
    return hashlib.sha256(raw).hexdigest(), len(tensors)


def atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index_raw = args.index.read_bytes()
    if hashlib.sha256(index_raw).hexdigest() != INDEX_SHA256:
        raise ValueError("official model index SHA256 differs")
    index = json.loads(index_raw)
    names = {
        name
        for name in index["weight_map"]
        if any(name.startswith(f"model.layers.{layer}.mlp.experts.") for layer in args.layers)
        and name.endswith((".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"))
    }
    if len(names) != 4 * 256 * 3:
        raise ValueError(f"expected 3072 selected tensors, got {len(names)}")
    shard_names = sorted({index["weight_map"][name] for name in names})
    records = {}
    for name in shard_names:
        path = args.shard_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        header_sha256, tensor_count = shard_header(path)
        records[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "header_sha256": header_sha256,
            "tensor_count": tensor_count,
        }
    value = {
        "schema": "glm52-fresh-sqg-bf16-shard-manifest-v1",
        "repo": REPO,
        "revision": REVISION,
        "index_sha256": INDEX_SHA256,
        "layers": list(args.layers),
        "selected_tensor_count": len(names),
        "shards": records,
        "total_shard_bytes": sum(record["bytes"] for record in records.values()),
    }
    atomic_json(args.output.resolve(), value)
    print(json.dumps({"output": str(args.output.resolve()), "layers": list(args.layers), "shards": len(records), "bytes": value["total_shard_bytes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
