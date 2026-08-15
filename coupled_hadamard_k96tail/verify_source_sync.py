#!/usr/bin/env python3
"""Verify the published K96-tail source snapshot and optional active roots."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "SOURCE_SHA256SUMS"
PROVENANCE = ROOT / "manifests" / "source_provenance.json"
SNAPSHOT = ROOT / "evidence" / "snapshot_2026-08-15T002639-0400"
FINAL_MECHANICAL = ROOT / "evidence" / "final-mechanical"
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
        and not any(part in SOURCE_EXCLUDED_PARTS for part in path.parts)
        and path.suffix not in {".pyc", ".pyo"}
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
        if (
            relative.as_posix() in skip
            or any(part in SOURCE_EXCLUDED_PARTS for part in relative.parts)
            or source.suffix in {".pyc", ".pyo"}
        ):
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


def read_sha256_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or not relative
            or relative in result
        ):
            raise ValueError(f"invalid SHA-256 manifest line in {path}: {line!r}")
        result[relative] = digest
    return result


def read_exact_archive(archive: Path, expected_sha256: str) -> bytes:
    if sha256(archive) != expected_sha256:
        raise ValueError(f"archive hash differs: {archive}")
    with gzip.open(archive, "rb") as handle:
        return handle.read()


def read_tar_archive(
    archive: Path,
    expected_archive_sha256: str,
    inner_manifest: Path,
    expected_manifest_sha256: str,
) -> dict[str, bytes]:
    if sha256(archive) != expected_archive_sha256:
        raise ValueError(f"tar archive hash differs: {archive}")
    if sha256(inner_manifest) != expected_manifest_sha256:
        raise ValueError(f"inner manifest hash differs: {inner_manifest}")
    expected = read_sha256_manifest(inner_manifest)
    observed: dict[str, bytes] = {}
    with tarfile.open(archive, mode="r:gz") as handle:
        for member in handle.getmembers():
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or member.name in observed
            ):
                raise ValueError(f"unsafe tar member: {member.name}")
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError(f"tar member has no bytes: {member.name}")
            observed[member.name] = stream.read()
    if set(observed) != set(expected):
        raise ValueError(f"tar member set differs: {archive}")
    for name, payload in observed.items():
        if hashlib.sha256(payload).hexdigest() != expected[name]:
            raise ValueError(f"tar member hash differs: {archive}:{name}")
    return observed


def json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON archive member is not an object: {label}")
    return value


def verify_final_mechanical() -> dict[str, Any]:
    receipt = load_json(FINAL_MECHANICAL / "MECHANICAL_EVIDENCE.json")
    if (
        receipt.get("schema")
        != "glm52-coupled-k96tail-final-mechanical-evidence-v1"
        or receipt.get("complete") is not True
        or receipt.get("mechanical_layer_count") != 75
        or receipt.get("passing_runtime_oracles") != 75
        or receipt.get("exact_k96_score_encode_parity_receipts") != 74
    ):
        raise ValueError("final mechanical summary differs")

    layer_archive = receipt["archives"]["layers"]
    layers = read_tar_archive(
        FINAL_MECHANICAL / layer_archive["path"],
        layer_archive["sha256"],
        FINAL_MECHANICAL / layer_archive["inner_manifest"],
        layer_archive["inner_manifest_sha256"],
    )
    parity_archive = receipt["archives"]["k96_parity"]
    parity = read_tar_archive(
        FINAL_MECHANICAL / parity_archive["path"],
        parity_archive["sha256"],
        FINAL_MECHANICAL / parity_archive["inner_manifest"],
        parity_archive["inner_manifest_sha256"],
    )
    if len(layers) != 225 or len(parity) != 74:
        raise ValueError("final mechanical archive counts differ")

    for layer in range(3, 78):
        padded = f"{layer:03d}"
        manifest = json_bytes(
            layers[f"r7-experts-layer-{padded}.json"], f"layer {layer} manifest"
        )
        quality = json_bytes(
            layers[f"r7-experts-layer-{padded}.quality.json"],
            f"layer {layer} quality",
        )
        oracle = json_bytes(
            layers[f"runtime-oracle-layer-{padded}.json"],
            f"layer {layer} oracle",
        )
        census = (
            {"k3": 720, "k4": 48, "total": 768}
            if layer == 3
            else {"k3": 672, "k4": 96, "total": 768}
        )
        bpw = 3.0625 if layer == 3 else 3.125
        if not (
            manifest.get("complete") is True
            and manifest.get("layer") == layer
            and manifest.get("bit_census") == census
            and manifest.get("bits_per_weight") == bpw
            and quality.get("complete") is True
            and quality.get("layer") == layer
            and oracle.get("complete") is True
            and oracle.get("pass") is True
            and oracle.get("layer") == layer
            and oracle.get("bit_census") == census
            and oracle.get("bits_per_weight") == bpw
        ):
            raise ValueError(f"final mechanical layer evidence differs: {layer}")

    for layer in range(4, 78):
        name = f"layer-{layer:03d}-k096-score-encode-parity.json"
        value = json_bytes(parity[name], f"layer {layer} parity")
        if not (
            value.get("complete") is True
            and value.get("all_exact") is True
            and value.get("experts_checked") == 256
            and value.get("projection_payloads_checked") == 768
        ):
            raise ValueError(f"final K96 parity differs: {layer}")

    exception = load_json(FINAL_MECHANICAL / "layer-003-k48-exception.json")
    if not (
        exception.get("complete") is True
        and exception.get("layer") == 3
        and exception.get("k96_score_encode_parity_receipt") is None
        and exception.get("parity_disposition")
        == "not_applicable_layer3_is_k48_not_k96"
    ):
        raise ValueError("layer-3 K48 exception differs")
    assembly = load_json(FINAL_MECHANICAL / receipt["assembly"]["path"])
    codec = load_json(FINAL_MECHANICAL / receipt["codec"]["path"])
    if (
        sha256(FINAL_MECHANICAL / receipt["assembly"]["path"])
        != receipt["assembly"]["sha256"]
        or assembly.get("complete") is not True
        or assembly.get("manifest_id") != receipt["assembly"]["manifest_id"]
    ):
        raise ValueError("assembly manifest binding differs")
    if (
        sha256(FINAL_MECHANICAL / receipt["codec"]["path"])
        != receipt["codec"]["sha256"]
        or codec.get("pass") is not True
        or codec.get("per_layer_census_ok") is not True
        or codec.get("coupled_layers") != list(range(3, 78))
        or codec.get("mtp_layer78_preserved") is not True
    ):
        raise ValueError("model codec receipt binding differs")
    return {
        "exact_k96_parity_receipts": len(parity),
        "layer_3_k48_exception": True,
        "mechanical_layer_files": len(layers),
        "passing_runtime_oracles": 75,
        "model_codec_receipt": "pass",
    }


def verify_exact_runtime_archives() -> dict[str, Any]:
    runtime_root = ROOT / "runtime" / "exact-ii-r11"
    provenance = load_json(runtime_root / "PROVENANCE.json")
    benchmark = provenance["quality_runner"]["benchmark"]
    benchmark_bytes = read_exact_archive(
        runtime_root / benchmark["exact_source_archive"],
        benchmark["exact_source_archive_sha256"],
    )
    if hashlib.sha256(benchmark_bytes).hexdigest() != benchmark["script_sha256"]:
        raise ValueError("measured benchmark decompressed hash differs")
    derivative = runtime_root / benchmark["ascii_derivative"]
    if (
        sha256(derivative) != benchmark["ascii_derivative_sha256"]
        or benchmark["ascii_derivative_is_byte_identical"] is not False
        or derivative.read_bytes() == benchmark_bytes
    ):
        raise ValueError("benchmark ASCII derivative binding differs")

    exact_sources = provenance["archived_exact_sources"]
    for record in exact_sources:
        payload = read_exact_archive(
            runtime_root / record["path"], record["archive_sha256"]
        )
        if hashlib.sha256(payload).hexdigest() != record["decompressed_sha256"]:
            raise ValueError(f"archived exact source differs: {record['path']}")

    qualification = (
        ROOT
        / "evidence/final-exact-ii-r11/qualification-5x"
        / "exact-r11-tp4dcp4mtp3-20260815T084427Z"
    )
    archive_provenance = load_json(qualification / "ARCHIVE_PROVENANCE.json")
    measured = archive_provenance["measured_manifest"]
    measured_manifest = read_exact_archive(
        qualification / measured["archive"], measured["archive_sha256"]
    )
    if hashlib.sha256(measured_manifest).hexdigest() != measured["decompressed_sha256"]:
        raise ValueError("measured quality manifest archive differs")
    for record in archive_provenance["compressed_exact_files"]:
        payload = read_exact_archive(
            qualification / record["archive"], record["archive_sha256"]
        )
        if hashlib.sha256(payload).hexdigest() != record["decompressed_sha256"]:
            raise ValueError(
                f"compressed qualification evidence differs: {record['archive']}"
            )
    publication_manifest = read_sha256_manifest(qualification / "SHA256SUMS")
    observed = {
        path.relative_to(qualification).as_posix(): path
        for path in qualification.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(publication_manifest) != set(observed):
        raise ValueError("qualification publication manifest file set differs")
    for relative, path in observed.items():
        if sha256(path) != publication_manifest[relative]:
            raise ValueError(f"qualification publication hash differs: {relative}")
    return {
        "archived_exact_runtime_sources": len(exact_sources),
        "benchmark_decompressed_sha256": benchmark["script_sha256"],
        "benchmark_derivative_byte_identical": False,
        "qualification_archived_exact_files": len(
            archive_provenance["compressed_exact_files"]
        ),
        "qualification_publication_files": len(publication_manifest),
    }


def verify_compact_evidence() -> dict[str, Any]:
    status = load_json(SNAPSHOT / "status.json")
    ledger = load_json(SNAPSHOT / "score_ledger.json")
    if status["schema"] != "glm52-coupled-k96tail-publication-snapshot-v1":
        raise ValueError("compact evidence status schema differs")
    if status["complete"] is not False:
        raise ValueError("historical compact evidence must not claim completion")
    final = status["final_results"]
    if final["status"] != "historical_snapshot_pre_finalization" or any(
        value is not None for key, value in final.items() if key != "status"
    ):
        raise ValueError("historical snapshot disposition differs")

    receipt_path = SNAPSHOT / ledger["receipt_manifest"]["path"]
    if sha256(receipt_path) != ledger["receipt_manifest"]["sha256"]:
        raise ValueError("score receipt manifest hash differs from ledger")
    receipts = read_sha256_manifest(receipt_path)
    if len(receipts) != 24064 or len(receipts) != ledger["totals"]["atomic_file_count"]:
        raise ValueError("atomic score receipt count differs")
    pattern = re.compile(
        r"^layer_(\d{3})/experts/expert_(\d{3})(\.row-sse\.npz|\.json)$"
    )
    counts: Counter[tuple[int, str]] = Counter()
    pairs: Counter[tuple[int, int]] = Counter()
    for relative in receipts:
        match = pattern.fullmatch(relative)
        if match is None:
            raise ValueError(f"unsafe atomic score receipt path: {relative}")
        layer = int(match.group(1))
        expert = int(match.group(2))
        suffix = match.group(3)
        if layer not in range(4, 51) or expert not in range(256):
            raise ValueError(f"atomic score receipt outside frozen domain: {relative}")
        counts[(layer, suffix)] += 1
        pairs[(layer, expert)] += 1
    if any(counts[(layer, suffix)] != 256 for layer in range(4, 51) for suffix in (".json", ".row-sse.npz")):
        raise ValueError("atomic score layer/type counts differ")
    if len(pairs) != 12032 or any(value != 2 for value in pairs.values()):
        raise ValueError("atomic score JSON/row-SSE pair closure differs")

    if ledger["schema"] != "glm52-k96tail-compact-score-ledger-v1":
        raise ValueError("score ledger schema differs")
    if ledger["frozen_at"] != status["snapshot_at"] or ledger["layer_count"] != 47:
        raise ValueError("score ledger frozen frontier differs")
    if [entry["layer"] for entry in ledger["layers"]] != list(range(4, 51)):
        raise ValueError("score ledger layer set differs")
    for entry in ledger["layers"]:
        if not entry["complete"] or any(
            entry[key] != 256
            for key in (
                "expert_json_count",
                "row_sse_npz_count",
                "expert_row_sse_pair_count",
                "embedded_row_sse_sha256_validated",
            )
        ):
            raise ValueError(f"score ledger layer is incomplete: {entry['layer']}")
        if len(entry["candidate_aggregates"]) != 8:
            raise ValueError(f"score candidate aggregate count differs: {entry['layer']}")

    for layer in range(3, 47):
        padded = f"{layer:03d}"
        manifest = load_json(SNAPSHOT / "layers_003_046" / f"r7-experts-layer-{padded}.json")
        quality = load_json(SNAPSHOT / "layers_003_046" / f"r7-experts-layer-{padded}.quality.json")
        oracle = load_json(SNAPSHOT / "layers_003_046" / f"runtime-oracle-layer-{padded}.json")
        if not manifest["complete"] or manifest["layer"] != layer:
            raise ValueError(f"sealed layer manifest differs: {layer}")
        if not quality["complete"] or quality["layer"] != layer:
            raise ValueError(f"sealed layer quality differs: {layer}")
        if not oracle["complete"] or not oracle["pass"] or oracle["layer"] != layer:
            raise ValueError(f"sealed layer runtime oracle differs: {layer}")
    for layer in range(4, 47):
        padded = f"{layer:03d}"
        parity = load_json(
            SNAPSHOT / "parity_layers_004_046" / f"layer-{padded}-k096-score-encode-parity.json"
        )
        if (
            not parity["complete"]
            or not parity["all_exact"]
            or parity["experts_checked"] != 256
            or parity["projection_payloads_checked"] != 768
        ):
            raise ValueError(f"score/encode parity differs: {layer}")
        allocation = load_json(
            SNAPSHOT / "allocations_layers_004_046" / f"layer_{padded}.kld-route-v1-k096.allocation.json"
        )
        if (
            not allocation["complete"]
            or not allocation["production_eligible"]
            or allocation["histogram"] != {"3": 672, "4": 96}
            or allocation["bpw"] != 3.125
        ):
            raise ValueError(f"guarded allocation differs: {layer}")

    forbidden_payloads = [
        path
        for path in SNAPSHOT.rglob("*")
        if path.is_file() and path.suffix in {".npz", ".safetensors"}
    ]
    if forbidden_payloads:
        raise ValueError(f"forbidden tensor payloads published: {forbidden_payloads}")
    return {
        "atomic_score_files_bound": len(receipts),
        "exact_parity_layers": 43,
        "production_eligible_allocations": 43,
        "score_complete_layers": 47,
        "sealed_runtime_oracles": 44,
    }


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


def compare_compact_evidence_active(args: argparse.Namespace) -> dict[str, int]:
    layer_files = 0
    for layer in range(3, 47):
        padded = f"{layer:03d}"
        for name in (
            f"r7-experts-layer-{padded}.json",
            f"r7-experts-layer-{padded}.quality.json",
            f"runtime-oracle-layer-{padded}.json",
        ):
            compare_file(
                SNAPSHOT / "layers_003_046" / name,
                args.active_layer_root / name,
                f"compact evidence/layers_003_046/{name}",
            )
            layer_files += 1
    parity_files = 0
    allocation_files = 0
    for layer in range(4, 47):
        padded = f"{layer:03d}"
        parity_name = f"layer-{padded}-k096-score-encode-parity.json"
        compare_file(
            SNAPSHOT / "parity_layers_004_046" / parity_name,
            args.active_parity_root / parity_name,
            f"compact evidence/parity_layers_004_046/{parity_name}",
        )
        parity_files += 1
        for suffix in (
            "tail-v7-k096.allocation.json",
            "kld-route-v1-k096.allocation.json",
        ):
            name = f"layer_{padded}.{suffix}"
            compare_file(
                SNAPSHOT / "allocations_layers_004_046" / name,
                args.active_allocation_root / name,
                f"compact evidence/allocations_layers_004_046/{name}",
            )
            allocation_files += 1

    recipe_files = 0
    recipe_root = SNAPSHOT / "recipe_layers_004_046"
    for source in sorted(recipe_root.rglob("*")):
        if not source.is_file() or source.is_symlink():
            continue
        relative = source.relative_to(recipe_root)
        layer_name, group, *tail = relative.parts
        if group in {"NO_SHORTCUT_COUPLED_RECIPE.json", "runtime_binding.json"}:
            active = args.active_recipe_root / layer_name / group
        elif group == "beta":
            active = args.active_recipe_root / layer_name / "beta" / tail[0]
        elif group in {"bootstrap_profile", "final_profile"}:
            active = (
                args.active_recipe_root
                / layer_name
                / group
                / "w4a8_native_profile_search"
                / tail[0]
            )
        elif group == "final_profile_binding":
            active = (
                args.active_recipe_root
                / "final_profiles"
                / layer_name
                / "w4a8_native_profile_search"
                / tail[0]
            )
        else:
            raise ValueError(f"unmapped compact recipe file: {relative}")
        compare_file(source, active, f"compact evidence/recipe/{relative}")
        recipe_files += 1

    receipt_manifest = read_sha256_manifest(
        SNAPSHOT / "score_receipts_layers_004_050.sha256"
    )
    for relative, expected in receipt_manifest.items():
        active = args.active_score_root / relative
        if not active.is_file() or active.is_symlink() or sha256(active) != expected:
            raise ValueError(f"active atomic score receipt differs: {active}")

    wave_archives = read_sha256_manifest(SNAPSHOT / "wave_archives.sha256")
    for relative, expected in wave_archives.items():
        active = args.active_wave_archive_root / relative
        if not active.is_file() or active.is_symlink() or sha256(active) != expected:
            raise ValueError(f"active candidate metadata archive differs: {active}")
    return {
        "allocation_files_equal": allocation_files,
        "atomic_score_receipts_equal": len(receipt_manifest),
        "layer_manifest_quality_oracle_files_equal": layer_files,
        "parity_files_equal": parity_files,
        "recipe_files_equal": recipe_files,
        "wave_archives_equal": len(wave_archives),
    }


def compare_final_mechanical_active(args: argparse.Namespace) -> dict[str, int]:
    receipt = load_json(FINAL_MECHANICAL / "MECHANICAL_EVIDENCE.json")
    layer_record = receipt["archives"]["layers"]
    layers = read_tar_archive(
        FINAL_MECHANICAL / layer_record["path"],
        layer_record["sha256"],
        FINAL_MECHANICAL / layer_record["inner_manifest"],
        layer_record["inner_manifest_sha256"],
    )
    for name, payload in layers.items():
        active = args.active_layer_root / name
        if not active.is_file() or active.read_bytes() != payload:
            raise ValueError(f"final mechanical active layer file differs: {name}")
    parity_record = receipt["archives"]["k96_parity"]
    parity = read_tar_archive(
        FINAL_MECHANICAL / parity_record["path"],
        parity_record["sha256"],
        FINAL_MECHANICAL / parity_record["inner_manifest"],
        parity_record["inner_manifest_sha256"],
    )
    for name, payload in parity.items():
        active = args.active_parity_root / name
        if not active.is_file() or active.read_bytes() != payload:
            raise ValueError(f"final mechanical active parity file differs: {name}")
    compare_file(
        FINAL_MECHANICAL / "COUPLED_REENCODE_MANIFEST.json",
        args.active_model_root / "COUPLED_REENCODE_MANIFEST.json",
        "final mechanical assembly manifest",
    )
    compare_file(
        FINAL_MECHANICAL / "model_codec_validation.json",
        args.active_codec_receipt,
        "final mechanical codec receipt",
    )
    return {
        "layer_manifest_quality_oracle_files_equal": len(layers),
        "parity_files_equal": len(parity),
        "assembly_manifest_equal": 1,
        "codec_receipt_equal": 1,
    }


def verify_finalized_contract_normalization(
    active_path: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    published_path = ROOT / "reproduction/machine/k96tail-distributed-campaign.json"
    campaign_path = ROOT / "campaign/reproduction/k96tail-distributed-campaign.json"
    compare_file(published_path, campaign_path, "published finalized contracts")
    active = load_json(active_path)
    published = load_json(published_path)
    normalization = provenance["campaign_source"]["finalized_contract_normalization"]
    if sha256(active_path) != normalization["active_historical_sha256"]:
        raise ValueError("active historical contract hash differs")
    allowed = set(normalization["allowed_changed_top_level_keys"])
    for key in set(active) | set(published):
        if key not in allowed and active.get(key) != published.get(key):
            raise ValueError(f"finalized contract changed disallowed key: {key}")
    changed = sorted(
        key for key in set(active) | set(published) if active.get(key) != published.get(key)
    )
    if changed != sorted(allowed):
        raise ValueError(
            f"finalized contract normalization set differs: {changed}"
        )
    final = published["final_results"]
    pending = sorted(key for key, value in final.items() if value is None)
    if pending != ["hub_model_commit", "model_card_hub_revision", "tensor_hub_revision"]:
        raise ValueError(f"finalized contract has non-Hub pending fields: {pending}")
    if not (
        published.get("complete") is False
        and published.get("status") == "research-only"
        and published["phase_completion"]["final_mechanical_evidence_sealed"] is True
        and final["mechanical_target_layer_count"] == 75
        and final["mechanical_runtime_oracle_count"] == 75
        and final["exact_k96_parity_receipt_count"] == 74
        and final["full_model_quality_gate_pass"] is False
    ):
        raise ValueError("finalized publication contract semantics differ")
    return {
        "active_historical_sha256": sha256(active_path),
        "allowed_changed_top_level_keys": changed,
        "normalizations": 1,
        "pending_fields_are_hub_only": True,
    }


def compare_active(args: argparse.Namespace, provenance: dict[str, Any]) -> dict[str, Any]:
    campaign_count = compare_subtree(
        ROOT / "campaign",
        args.active_project,
        "campaign",
        skip=frozenset(
            {
                "docs/K96_COUPLED_DISTRIBUTED_REPRODUCTION_20260814.md",
                "hub/k96tail-staging/README.md",
                "reproduction/README.md",
                "reproduction/k96tail-distributed-campaign.json",
                "scripts/audit_k96tail_campaign.py",
                "tests/archive/test_k96tail_reproduction_contract.active.py.gz",
                "tests/test_k96tail_reproduction_contract.py",
            }
        ),
    )
    if campaign_count != provenance["campaign_source"]["active_equal_file_count"]:
        raise ValueError("campaign source file count differs from provenance")
    active_test_archive = (
        ROOT
        / "campaign/tests/archive"
        / "test_k96tail_reproduction_contract.active.py.gz"
    )
    active_test_bytes = read_exact_archive(
        active_test_archive,
        provenance["campaign_source"]["active_historical_test_archive_sha256"],
    )
    active_test = args.active_project / "tests/test_k96tail_reproduction_contract.py"
    if active_test.read_bytes() != active_test_bytes:
        raise ValueError("archived active historical campaign test differs")
    contract_normalization = verify_finalized_contract_normalization(
        args.active_project / "reproduction/k96tail-distributed-campaign.json",
        provenance,
    )

    orchestration_count = compare_subtree(
        ROOT / "orchestration" / "vast_supervisor",
        args.active_project / "vast_supervisor",
        "orchestration/vast_supervisor",
    )
    for name in (
        "glm52-full-coupled-k96tail-resume.service",
        "glm52-k96tail-hub-wave-prefetch.service",
    ):
        compare_file(
            ROOT / "orchestration" / "systemd" / name,
            args.active_project / "systemd" / name,
            f"orchestration/systemd/{name}",
        )
        orchestration_count += 1
    for name in (
        "glm52-full-coupled-k96tail-no-shortcut-goal019ffa7c.service",
        "glm52-k96tail-distributed-merge-finalize.service",
        "glm52-k96tail-hub-local-persistent.service",
    ):
        compare_file(
            ROOT / "orchestration" / "systemd" / name,
            args.active_user_systemd / name,
            f"orchestration/systemd/{name}",
        )
        orchestration_count += 1
    if orchestration_count != provenance["orchestration_source"]["file_count"]:
        raise ValueError("orchestration source file count differs from provenance")
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
    exact_runtime_count = 0
    for relative in (
        "deploy/accelerate-hf-routed-upload.sh",
        "deploy/wait-and-upload-final-model.sh",
    ):
        compare_file(
            ROOT / "runtime/exact-ii-r11" / relative,
            args.active_exact_runtime / relative,
            f"exact runtime/{relative}",
        )
        exact_runtime_count += 1
    benchmark = load_json(
        ROOT / "runtime/exact-ii-r11/PROVENANCE.json"
    )["quality_runner"]["benchmark"]
    benchmark_bytes = read_exact_archive(
        ROOT / "runtime/exact-ii-r11" / benchmark["exact_source_archive"],
        benchmark["exact_source_archive_sha256"],
    )
    if benchmark_bytes != args.active_benchmark.read_bytes():
        raise ValueError("exact measured benchmark archive differs from active source")
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
        "campaign_historical_test_archive_equal": 1,
        "campaign_publication_normalization": contract_normalization,
        "compact_evidence": compare_compact_evidence_active(args),
        "final_mechanical_evidence": compare_final_mechanical_active(args),
        "exact_runtime_active_files_equal": exact_runtime_count,
        "exact_measured_benchmark_equal": 1,
        "kquant": kquant,
        "kquant_working_source_files_equal": kquant_source_count,
        "qsrt": qsrt,
        "qsrt_working_source_files_equal": qsrt_source_count,
        "orchestration_files_equal": orchestration_count,
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
        "--active-exact-runtime",
        type=Path,
        default=Path("/home/brandonmusic/KLC_SANDBOXES/ii-r11-k96-runtime"),
    )
    result.add_argument(
        "--active-benchmark",
        type=Path,
        default=Path(
            "/home/brandonmusic/KLC_SANDBOXES/"
            "llm-inference-bench-v0.4.29/llm_decode_bench.py"
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
    result.add_argument(
        "--active-layer-root",
        type=Path,
        default=Path(
            "/home/brandonmusic/models/"
            "GLM-5.2-SQG-Coupled-H512-H128-K96Tail-layers"
        ),
    )
    result.add_argument(
        "--active-model-root",
        type=Path,
        default=Path(
            "/home/brandonmusic/models/"
            "GLM-5.2-SQG-Coupled-H512-H128-K96Tail"
        ),
    )
    result.add_argument(
        "--active-codec-receipt",
        type=Path,
        default=Path(
            "/home/brandonmusic/KLC_SANDBOXES/"
            "glm52_sqg_w4a8_sm120_local_acceptance_20260812/RESULTS/"
            "full_coupled_k96tail_no_shortcut_model_codec_quick.json"
        ),
    )
    result.add_argument(
        "--active-parity-root",
        type=Path,
        default=Path(
            "/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625/"
            "evidence/full-coupled-k96tail-no-shortcut"
        ),
    )
    result.add_argument(
        "--active-allocation-root",
        type=Path,
        default=Path(
            "/media/brandonmusic/nvme1n1p3/"
            "glm52-coupled-k96tail-no-shortcut-allocations-v1"
        ),
    )
    result.add_argument(
        "--active-recipe-root",
        type=Path,
        default=Path(
            "/media/brandonmusic/nvme1n1p3/"
            "glm52-coupled-no-shortcut-recipe-v1"
        ),
    )
    result.add_argument(
        "--active-score-root",
        type=Path,
        default=Path(
            "/media/brandonmusic/nvme1n1p3/"
            "glm52-coupled-tail-rate-scores-no-shortcut-v5"
        ),
    )
    result.add_argument(
        "--active-wave-archive-root",
        type=Path,
        default=Path(
            "/home/brandonmusic/models/"
            "GLM-5.2-SQG-Coupled-H512-H128-K96Tail-reproduction/wave-archives"
        ),
    )
    result.add_argument(
        "--active-user-systemd",
        type=Path,
        default=Path("/home/brandonmusic/.config/systemd/user"),
    )
    return result


def main() -> None:
    args = parser().parse_args()
    provenance = load_json(PROVENANCE)
    result: dict[str, Any] = {
        "compact_evidence": verify_compact_evidence(),
        "complete": True,
        "exact_runtime_archives": verify_exact_runtime_archives(),
        "final_mechanical_evidence": verify_final_mechanical(),
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
