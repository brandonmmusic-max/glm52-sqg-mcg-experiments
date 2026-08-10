#!/usr/bin/env python3
"""Validate four final SQG shards and assemble one canonical layer artifact."""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SHARD_RANGES = (
    (0, 64),
    (64, 128),
    (128, 192),
    (192, 256),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidate four validated final-treatment SQG shards"
    )
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", required=True)
    return parser


def _publish_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    sha256_file,
) -> str:
    """Publish without replacing any existing destination."""

    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise ValueError(f"canonical destination is not a regular file: {destination}")
        if source.stat().st_ino == destination.stat().st_ino and (
            source.stat().st_dev == destination.stat().st_dev
        ):
            return "existing_hardlink"
        if sha256_file(destination) != expected_sha256:
            raise ValueError(f"refusing to overwrite mismatched file: {destination}")
        return "existing_equal"

    try:
        os.link(source, destination, follow_symlinks=False)
        return "hardlinked"
    except FileExistsError:
        return _publish_file(
            source,
            destination,
            expected_sha256=expected_sha256,
            sha256_file=sha256_file,
        )
    except OSError as exc:
        fallback_errors = {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.EMLINK,
        }
        if hasattr(errno, "EOPNOTSUPP"):
            fallback_errors.add(errno.EOPNOTSUPP)
        if exc.errno not in fallback_errors:
            raise

    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=64 << 20)
            writer.flush()
            os.fsync(writer.fileno())
    except FileExistsError:
        return _publish_file(
            source,
            destination,
            expected_sha256=expected_sha256,
            sha256_file=sha256_file,
        )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if sha256_file(destination) != expected_sha256:
        destination.unlink(missing_ok=True)
        raise ValueError(f"copied file hash differs: {destination}")
    return "copied"


def main() -> int:
    args = _parser().parse_args()

    from scripts.encode_final_shard import _open_fast_sealed_runtime
    from src.fresh_pipeline_artifacts import (
        expert_stem,
        validate_expert_artifact,
    )
    from src.fresh_pipeline_common import sha256_file
    from src.fresh_pipeline_runner import encode_layer

    runtime = _open_fast_sealed_runtime(
        args.preflight,
        layer=args.layer,
        device=args.device,
    )
    shard_root = runtime.layer_root / "expert_shards"
    expected_names = {
        f"experts_{start:03d}_{end:03d}" for start, end in SHARD_RANGES
    }
    if not shard_root.is_dir() or shard_root.is_symlink():
        raise FileNotFoundError(f"expert shard root is absent or unsafe: {shard_root}")
    observed_names = {
        path.name
        for path in shard_root.iterdir()
        if path.name.startswith("experts_") and path.is_dir()
    }
    if observed_names != expected_names:
        raise ValueError(
            "final shard directory set differs: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )

    validated: list[tuple[Path, dict[str, object]]] = []
    for start, end in SHARD_RANGES:
        directory = shard_root / f"experts_{start:03d}_{end:03d}"
        if directory.is_symlink():
            raise ValueError(f"expert shard directory is symlinked: {directory}")
        expected_manifests = {
            f"{expert_stem(args.layer, expert)}.json"
            for expert in range(start, end)
        }
        observed_manifests = {
            path.name
            for path in directory.glob("layer-*-expert-*.json")
            if not path.name.endswith(".json.sha256")
        }
        if observed_manifests != expected_manifests:
            raise ValueError(
                f"{directory.name} expert manifest set differs: "
                f"missing={sorted(expected_manifests - observed_manifests)[:4]}, "
                f"extra={sorted(observed_manifests - expected_manifests)[:4]}"
            )
        for expert in range(start, end):
            manifest_path = directory / f"{expert_stem(args.layer, expert)}.json"
            artifact = validate_expert_artifact(manifest_path)
            if (
                int(artifact.get("layer", -1)) != args.layer
                or int(artifact.get("expert", -1)) != expert
                or artifact.get("purpose") != "final_treatment"
            ):
                raise ValueError(f"expert artifact binding differs: {manifest_path}")
            validated.append((manifest_path, artifact))
    if len(validated) != 256:
        raise RuntimeError("final shard validation did not cover exactly 256 experts")

    runtime.expert_dir.mkdir(parents=True, exist_ok=True)
    publication = {
        "hardlinked": 0,
        "copied": 0,
        "existing_hardlink": 0,
        "existing_equal": 0,
    }
    for manifest_path, artifact in validated:
        members = (
            (manifest_path, sha256_file(manifest_path)),
            (
                manifest_path.with_suffix(".json.sha256"),
                sha256_file(manifest_path.with_suffix(".json.sha256")),
            ),
            (
                manifest_path.with_suffix(".safetensors"),
                str(artifact["shard_sha256"]),
            ),
        )
        for source, expected_sha256 in members:
            outcome = _publish_file(
                source,
                runtime.expert_dir / source.name,
                expected_sha256=expected_sha256,
                sha256_file=sha256_file,
            )
            publication[outcome] += 1

    layer_artifact = encode_layer(runtime)
    layer_manifest_path = (
        runtime.final_dir / f"fresh-sqg-layer-{args.layer:03d}.json"
    )
    print(
        json.dumps(
            {
                "complete": True,
                "layer": args.layer,
                "device": args.device,
                "expert_count": len(validated),
                "source_shard_directories": sorted(expected_names),
                "canonical_expert_dir": str(runtime.expert_dir),
                "published_files": sum(publication.values()),
                "publication": publication,
                "layer_artifact": {
                    "manifest": str(layer_manifest_path),
                    "manifest_sha256": sha256_file(layer_manifest_path),
                    "shard": str(runtime.final_dir / str(layer_artifact["shard"])),
                    "shard_sha256": layer_artifact["shard_sha256"],
                    "sqg_tensor_count": layer_artifact["lineage"][
                        "sqg_tensor_count"
                    ],
                    "mcg_tensor_count": layer_artifact["lineage"][
                        "mcg_tensor_count"
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
