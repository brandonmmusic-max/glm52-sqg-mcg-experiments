#!/usr/bin/env python3
"""Verify the published K96-tail source snapshot and optional active roots."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "SOURCE_SHA256SUMS"
PROVENANCE = ROOT / "manifests" / "source_provenance.json"
SOURCE_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "build",
        "dist",
        "out",
        "__pycache__",
    }
)
SOURCE_EXCLUDED_SUFFIXES = frozenset(
    {".a", ".bin", ".o", ".pyo", ".pyc", ".safetensors", ".so"}
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def published_files() -> dict[str, Path]:
    return {
        path.relative_to(ROOT).as_posix(): path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path != MANIFEST
        and "__pycache__" not in path.parts
    }


def verify_manifest() -> int:
    if not MANIFEST.is_file() or MANIFEST.is_symlink():
        raise ValueError("SOURCE_SHA256SUMS is absent or unsafe")
    expected: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or relative in expected:
            raise ValueError(f"invalid manifest line: {line!r}")
        expected[relative] = digest
    observed = published_files()
    if set(expected) != set(observed):
        missing = sorted(set(observed) - set(expected))
        extra = sorted(set(expected) - set(observed))
        raise ValueError(f"manifest file set differs: missing={missing}, extra={extra}")
    for relative, path in observed.items():
        actual = sha256(path)
        if actual != expected[relative]:
            raise ValueError(f"manifest hash differs: {relative}")
    return len(observed)


def compare_file(source: Path, active: Path, label: str) -> None:
    if not active.is_file() or active.is_symlink():
        raise ValueError(f"active {label} is absent or unsafe: {active}")
    if sha256(source) != sha256(active):
        raise ValueError(f"active {label} differs: {active}")


def compare_subtree(
    published: Path,
    active: Path,
    label: str,
    *,
    skip: frozenset[str] = frozenset(),
) -> int:
    count = 0
    for source in sorted(published.rglob("*")):
        if not source.is_file() or source.is_symlink():
            continue
        relative = source.relative_to(published)
        if relative.as_posix() in skip:
            continue
        compare_file(source, active / relative, f"{label}/{relative.as_posix()}")
        count += 1
    return count


def verify_overlay_manifest() -> int:
    overlay_root = ROOT / "runtime" / "build-context"
    manifest = overlay_root / "OVERLAY_SHA256SUMS"
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or relative in expected:
            raise ValueError(f"invalid overlay manifest line: {line!r}")
        expected[relative] = digest
    observed = {
        path.relative_to(overlay_root).as_posix(): path
        for path in sorted((overlay_root / "overlay").rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    if set(expected) != set(observed):
        raise ValueError("filtered runtime overlay manifest file set differs")
    for relative, path in observed.items():
        if sha256(path) != expected[relative]:
            raise ValueError(f"filtered runtime overlay hash differs: {relative}")
    return len(observed)


def working_source_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if (
            any(part in SOURCE_EXCLUDED_PARTS for part in relative.parts)
            or any(part.endswith(".egg-info") for part in relative.parts)
            or path.suffix in SOURCE_EXCLUDED_SUFFIXES
        ):
            continue
        result[relative.as_posix()] = path
    return result


def compare_complete_source(
    published: Path, active: Path, label: str, expected_count: int
) -> int:
    published_files = working_source_files(published)
    active_files = working_source_files(active)
    if set(published_files) != set(active_files):
        missing = sorted(set(active_files) - set(published_files))
        extra = sorted(set(published_files) - set(active_files))
        raise ValueError(
            f"{label} working source file set differs: missing={missing}, extra={extra}"
        )
    if len(published_files) != expected_count:
        raise ValueError(f"{label} source count differs from provenance")
    for relative, source in published_files.items():
        compare_file(source, active_files[relative], f"{label}/{relative}")
    return len(published_files)


def git_output(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def verify_patch(
    *,
    name: str,
    active_root: Path,
    expected_revision: str,
    expected_diff_sha256: str,
) -> dict[str, Any]:
    published = ROOT / "patches" / name
    patch = published / "changes.patch"
    changed_root = published / "changed-files"
    revision = git_output(active_root, "rev-parse", "HEAD").decode().strip()
    if revision != expected_revision:
        raise ValueError(f"{name} base revision differs: {revision}")
    diff = git_output(active_root, "diff", "--binary", "HEAD")
    diff_hash = hashlib.sha256(diff).hexdigest()
    if diff_hash != expected_diff_sha256 or patch.read_bytes() != diff:
        raise ValueError(f"{name} tracked diff differs")
    changed = [
        value
        for value in git_output(active_root, "diff", "--name-only", "HEAD")
        .decode()
        .splitlines()
        if value
    ]
    mirrored = sorted(
        path.relative_to(changed_root).as_posix()
        for path in changed_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if sorted(changed) != mirrored:
        raise ValueError(f"{name} changed-file set differs")
    for relative in mirrored:
        compare_file(
            changed_root / relative,
            active_root / relative,
            f"{name} changed file {relative}",
        )
    archive = git_output(active_root, "archive", "--format=tar", "HEAD")
    with tempfile.TemporaryDirectory(prefix=f"k96tail-{name}-") as temporary:
        destination = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
            handle.extractall(destination, filter="data")
        subprocess.run(
            ["git", "apply", "--check", str(patch)],
            cwd=destination,
            check=True,
        )
        subprocess.run(["git", "apply", str(patch)], cwd=destination, check=True)
        for relative in mirrored:
            compare_file(
                changed_root / relative,
                destination / relative,
                f"{name} applied file {relative}",
            )
    return {
        "base_revision": revision,
        "changed_files": len(mirrored),
        "patch_sha256": diff_hash,
        "patch_applies_exactly": True,
    }


def compare_active(args: argparse.Namespace, provenance: dict[str, Any]) -> dict[str, Any]:
    campaign_count = compare_subtree(
        ROOT / "campaign", args.active_project, "campaign"
    )
    if campaign_count != provenance["campaign_source"]["file_count"]:
        raise ValueError("campaign source file count differs from provenance")
    runtime_count = compare_subtree(
        ROOT / "runtime" / "build-context",
        args.active_acceptance / "build-context",
        "runtime/build-context",
        skip=frozenset({"OVERLAY_SHA256SUMS"}),
    )
    compare_file(
        ROOT / "manifests" / "active_runtime_OVERLAY_SHA256SUMS",
        args.active_acceptance / "build-context" / "OVERLAY_SHA256SUMS",
        "active runtime OVERLAY_SHA256SUMS",
    )
    runtime_count += 1
    for relative in ("compose.yaml", "serve.sh"):
        compare_file(
            ROOT / "runtime" / relative,
            args.active_acceptance / relative,
            f"runtime/{relative}",
        )
        runtime_count += 1
    for relative in ("smoke_test.py", "validate_model_codec.py"):
        compare_file(
            ROOT / "runtime" / "scripts" / relative,
            args.active_acceptance / "scripts" / relative,
            f"runtime/scripts/{relative}",
        )
        runtime_count += 1
    if runtime_count != provenance["runtime_source"]["file_count"]:
        raise ValueError("runtime source file count differs from provenance")
    qsrt_source_count = compare_complete_source(
        ROOT / provenance["qsrt"]["complete_working_source_root"],
        args.active_qsrt,
        "qsrt source",
        provenance["qsrt"]["complete_working_source_file_count"],
    )
    kquant_source_count = compare_complete_source(
        ROOT / provenance["kquant"]["complete_working_source_root"],
        args.active_kquant,
        "kquant source",
        provenance["kquant"]["complete_working_source_file_count"],
    )
    qsrt = verify_patch(
        name="qsrt",
        active_root=args.active_qsrt,
        expected_revision=provenance["qsrt"]["base_revision"],
        expected_diff_sha256=provenance["qsrt"]["patch_sha256"],
    )
    kquant = verify_patch(
        name="kquant",
        active_root=args.active_kquant,
        expected_revision=provenance["kquant"]["base_revision"],
        expected_diff_sha256=provenance["kquant"]["patch_sha256"],
    )
    return {
        "campaign_files_equal": campaign_count,
        "kquant": kquant,
        "kquant_working_source_files_equal": kquant_source_count,
        "qsrt": qsrt,
        "qsrt_working_source_files_equal": qsrt_source_count,
        "runtime_files_equal": runtime_count,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--compare-active", action="store_true")
    result.add_argument(
        "--active-project",
        type=Path,
        default=Path("/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625"),
    )
    result.add_argument(
        "--active-acceptance",
        type=Path,
        default=Path(
            "/home/brandonmusic/KLC_SANDBOXES/"
            "glm52_sqg_w4a8_sm120_local_acceptance_20260812"
        ),
    )
    result.add_argument(
        "--active-qsrt",
        type=Path,
        default=Path("/home/brandonmusic/KLC_SANDBOXES/qsrt-glm52-port"),
    )
    result.add_argument(
        "--active-kquant",
        type=Path,
        default=Path(
            "/home/brandonmusic/KLC_SANDBOXES/"
            "glm52_fresh_sqg_3p0625/kquant"
        ),
    )
    return result


def main() -> None:
    args = parser().parse_args()
    provenance = load_json(PROVENANCE)
    result: dict[str, Any] = {
        "complete": True,
        "internal_manifest_files": verify_manifest(),
        "runtime_overlay_files": verify_overlay_manifest(),
        "schema": "glm52-coupled-k96tail-source-sync-verification-v1",
    }
    for name in ("qsrt", "kquant"):
        patch = ROOT / provenance[name]["patch"]
        if sha256(patch) != provenance[name]["patch_sha256"]:
            raise ValueError(f"published {name} patch hash differs")
    if args.compare_active:
        result["active_equivalence"] = compare_active(args, provenance)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
