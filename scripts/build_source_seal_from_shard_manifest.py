#!/usr/bin/env python3
"""Promote one already-hashed BF16 shard manifest to a pipeline source seal.

The shard-manifest builder reads and hashes every payload once.  This helper
reuses those hashes, but independently checks the pinned index/config,
selected-tensor binding, file sizes, and safetensors headers.  It therefore
avoids a byte-neutral second 80+ GB payload pass before a follow-up pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fresh_pipeline_common import atomic_json
from src.glm52_bf16_manifest import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_INDEX_BYTES,
    EXPECTED_INDEX_SHA256,
    SOURCE_REPO,
    SOURCE_REVISION,
)
from src.glm52_bf16_source import PROJECTIONS, read_safetensors_header


def sha256_bytes(path: Path) -> tuple[int, str]:
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = args.manifest.resolve()
    index_path = args.index.resolve()
    config_path = args.config.resolve()
    shard_root = args.shard_root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "glm52-fresh-sqg-bf16-shard-manifest-v1"
        or manifest.get("repo") != SOURCE_REPO
        or manifest.get("revision") != SOURCE_REVISION
        or manifest.get("index_sha256") != EXPECTED_INDEX_SHA256
        or not isinstance(manifest.get("shards"), dict)
    ):
        raise ValueError("BF16 shard manifest contract differs")
    layers = tuple(int(layer) for layer in manifest.get("layers", ()))
    if len(layers) != 4 or len(set(layers)) != 4:
        raise ValueError("source seal requires exactly four unique layers")

    index_bytes, index_sha256 = sha256_bytes(index_path)
    if index_bytes != EXPECTED_INDEX_BYTES or index_sha256 != EXPECTED_INDEX_SHA256:
        raise ValueError("official index identity differs")
    config_bytes, config_sha256 = sha256_bytes(config_path)
    if config_sha256 != EXPECTED_CONFIG_SHA256:
        raise ValueError("official config identity differs")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("official index has no weight_map")

    selected_names = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
        for layer in layers
        for expert in range(256)
        for projection in PROJECTIONS
    }
    if len(selected_names) != 3_072 or not selected_names.issubset(weight_map):
        raise ValueError("official index does not bind all selected tensors")
    selected_by_shard: dict[str, set[str]] = {}
    indexed_by_shard: dict[str, set[str]] = {}
    for name, shard in weight_map.items():
        indexed_by_shard.setdefault(str(shard), set()).add(str(name))
    for name in selected_names:
        selected_by_shard.setdefault(str(weight_map[name]), set()).add(name)
    if set(selected_by_shard) != set(manifest["shards"]):
        raise ValueError("selected index shards differ from hashed manifest")

    records: dict[str, dict[str, object]] = {}
    for name, record in sorted(manifest["shards"].items()):
        if Path(name).name != name or not isinstance(record, dict):
            raise ValueError(f"unsafe shard manifest record: {name!r}")
        path = shard_root / name
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"shard size changed after hashing: {name}")
        header = read_safetensors_header(path)
        if (
            header.header_sha256 != str(record["header_sha256"])
            or len(header.tensors) != int(record["tensor_count"])
            or set(header.tensors) != indexed_by_shard[name]
        ):
            raise ValueError(f"shard header/index binding differs: {name}")
        records[name] = {
            "bytes": int(record["bytes"]),
            "sha256": str(record["sha256"]),
            "header_sha256": str(record["header_sha256"]),
            "selected_tensor_count": len(selected_by_shard[name]),
            "total_tensor_count": len(header.tensors),
        }

    tensor_counts = {str(layer): 256 * len(PROJECTIONS) for layer in layers}
    seal = {
        "schema": "glm52-fresh-sqg-bf16-source-v2",
        "repo": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "index": {
            "path": str(index_path),
            "bytes": index_bytes,
            "sha256": index_sha256,
        },
        "config": {
            "path": str(config_path),
            "bytes": config_bytes,
            "sha256": config_sha256,
        },
        "shard_root": str(shard_root),
        "layers": list(layers),
        "tensor_counts": tensor_counts,
        "target_tensor_count": sum(tensor_counts.values()),
        "shard_count": len(records),
        "total_shard_bytes": sum(int(record["bytes"]) for record in records.values()),
        "complete_index_header_binding_validated": True,
        "payload_hashes_reused_from": {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "shards": records,
    }
    atomic_json(output, seal)
    print(output)


if __name__ == "__main__":
    main()
