#!/usr/bin/env python3
"""Audit and seal the exact official BF16 source for the fresh SQG pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import glm52_bf16_source as bf16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="official config.json (defaults to the index directory)",
    )
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_json_atomic(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_source_seal(
    *,
    index_path: str | Path,
    shard_root: str | Path,
    config_path: str | Path | None = None,
    source_revision: str = bf16.SOURCE_REVISION,
) -> dict[str, object]:
    """Return a seal only after the hard-coded official identity validates."""

    index_path = Path(index_path).resolve()
    shard_root = Path(shard_root).resolve()
    config_path = (
        Path(config_path).resolve()
        if config_path is not None
        else index_path.with_name("config.json")
    )
    validation = bf16.validate_bf16_source(
        index_path=index_path,
        config_path=config_path,
        shard_root=shard_root,
        source_revision=source_revision,
    )
    shard_records = {
        name: {
            "bytes": record.bytes,
            "sha256": record.sha256,
            "header_sha256": record.header_sha256,
            "selected_tensor_count": record.selected_tensor_count,
            "total_tensor_count": record.total_tensor_count,
        }
        for name, record in sorted(validation.shards.items())
    }
    return {
        "schema": "glm52-fresh-sqg-bf16-source-v2",
        "repo": bf16.SOURCE_REPO,
        "revision": bf16.SOURCE_REVISION,
        "index": {
            "path": str(index_path),
            "bytes": validation.index_bytes,
            "sha256": validation.index_sha256,
        },
        "config": {
            "path": str(config_path),
            "bytes": validation.config_bytes,
            "sha256": validation.config_sha256,
        },
        "shard_root": str(shard_root),
        "layers": list(bf16.SELECTED_LAYERS),
        "tensor_counts": dict(validation.tensor_counts),
        "target_tensor_count": sum(validation.tensor_counts.values()),
        "shard_count": len(shard_records),
        "total_shard_bytes": sum(
            int(record["bytes"]) for record in shard_records.values()
        ),
        "complete_index_header_binding_validated": True,
        "shards": shard_records,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite seal: {output}")
    result = build_source_seal(
        index_path=args.index,
        config_path=args.config,
        shard_root=args.shard_root,
    )
    write_json_atomic(output, result)
    print(output)


if __name__ == "__main__":
    main()
