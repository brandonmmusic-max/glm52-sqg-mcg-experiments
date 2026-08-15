#!/usr/bin/env python3
"""Verify local model files against one Hugging Face model revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


def digest_file(path: Path) -> tuple[str, str]:
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            sha256.update(chunk)
            git_sha1.update(chunk)
    return sha256.hexdigest(), git_sha1.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.model_root.resolve()
    excluded = set(args.exclude)
    local: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        name = relative.as_posix()
        # Hugging Face creates resumable-client state under .cache in the
        # upload root. It is operational metadata, not checkpoint content.
        if ".cache" in relative.parts or name in excluded:
            continue
        local[name] = path
    info = HfApi().model_info(
        args.repo,
        revision=args.revision,
        files_metadata=True,
    )
    remote = {sibling.rfilename: sibling for sibling in info.siblings}

    missing: list[str] = []
    mismatches: list[dict[str, Any]] = []
    verified_lfs = 0
    verified_git = 0
    total_bytes = 0

    for name, path in local.items():
        size = path.stat().st_size
        total_bytes += size
        sibling = remote.get(name)
        if sibling is None:
            missing.append(name)
            continue
        sha256, git_sha1 = digest_file(path)
        if sibling.size != size:
            mismatches.append(
                {
                    "path": name,
                    "field": "size",
                    "local": size,
                    "hub": sibling.size,
                }
            )
            continue
        if sibling.lfs is not None:
            verified_lfs += 1
            if sibling.lfs.sha256 != sha256:
                mismatches.append(
                    {
                        "path": name,
                        "field": "sha256",
                        "local": sha256,
                        "hub": sibling.lfs.sha256,
                    }
                )
        else:
            verified_git += 1
            if sibling.blob_id != git_sha1:
                mismatches.append(
                    {
                        "path": name,
                        "field": "git_blob_sha1",
                        "local": git_sha1,
                        "hub": sibling.blob_id,
                    }
                )

    complete = not missing and not mismatches
    receipt = {
        "schema": "glm52-k96tail-hub-file-verification-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo": args.repo,
        "requested_revision": args.revision,
        "resolved_revision": info.sha,
        "model_root": str(root),
        "excluded_paths": sorted(excluded),
        "local_file_count": len(local),
        "local_total_bytes": total_bytes,
        "verified_lfs_file_count": verified_lfs,
        "verified_git_file_count": verified_git,
        "missing_paths": missing,
        "mismatches": mismatches,
        "complete": complete,
    }
    atomic_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
