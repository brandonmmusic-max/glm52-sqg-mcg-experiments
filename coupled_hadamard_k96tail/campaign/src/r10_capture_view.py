"""Validation for the zero-copy R10-to-SQG calibration view."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


SCHEMA = "glm52-r10-sqg-capture-view-v1"
LAYER_SCHEMA = "glm52-r10-sqg-layer-view-v1"
TOKENS = 1_049_589
DOCUMENTS = 1_773
HIDDEN = 6_144
TOPK = 8
EXPERTS = 256
ROUTED_SCALE = 2.5
FILE_ABI = {
    "hidden.bf16.bin": (HIDDEN * 2, "bfloat16-le", [TOKENS, HIDDEN]),
    "topk_ids.u8.bin": (TOPK, "uint8", [TOKENS, TOPK]),
    "topk_weights.f32le.bin": (TOPK * 4, "float32-le", [TOKENS, TOPK]),
    "doc_epochs.u32le.bin": (4, "uint32-le", [TOKENS]),
    "token_positions.u16le.bin": (2, "uint16-le", [TOKENS]),
    "role_ids.u8.bin": (1, "uint8", [TOKENS]),
}


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _validate_document_vectors(directory: Path, documents: list[dict]) -> None:
    epochs = np.memmap(directory / "doc_epochs.u32le.bin", mode="r", dtype="<u4")
    positions = np.memmap(
        directory / "token_positions.u16le.bin", mode="r", dtype="<u2"
    )
    roles = np.memmap(directory / "role_ids.u8.bin", mode="r", dtype="u1")
    offset = 0
    for document in documents:
        count = int(document["tokens"])
        end = offset + count
        if not np.all(epochs[offset:end] == int(document["epoch"])):
            raise ValueError(f"{directory}: document epoch vector differs")
        if not np.array_equal(positions[offset:end], np.arange(count, dtype="<u2")):
            raise ValueError(f"{directory}: document position vector differs")
        if not np.all(roles[offset:end] == int(document["role_id"])):
            raise ValueError(f"{directory}: document role vector differs")
        offset = end
    if offset != TOKENS:
        raise ValueError(f"{directory}: document vectors do not close")


def _validate_routes(directory: Path, expected_counts: list[int]) -> None:
    ids = np.memmap(
        directory / "topk_ids.u8.bin", mode="r", dtype="u1", shape=(TOKENS, TOPK)
    )
    weights = np.memmap(
        directory / "topk_weights.f32le.bin",
        mode="r",
        dtype="<f4",
        shape=(TOKENS, TOPK),
    )
    counts = np.zeros(EXPERTS, dtype=np.int64)
    for begin in range(0, TOKENS, 65_536):
        end = min(TOKENS, begin + 65_536)
        local_ids = np.asarray(ids[begin:end])
        local_weights = np.asarray(weights[begin:end])
        ordered = np.sort(local_ids, axis=1)
        if np.any(ordered[:, 1:] == ordered[:, :-1]):
            raise ValueError(f"{directory}: duplicate routed expert")
        if not np.isfinite(local_weights).all() or np.any(local_weights < 0):
            raise ValueError(f"{directory}: routed weights are invalid")
        sums = local_weights.astype(np.float64).sum(axis=1)
        if not np.allclose(sums, ROUTED_SCALE, rtol=2e-6, atol=2e-6):
            raise ValueError(f"{directory}: routed weights are not applied gates")
        counts += np.bincount(local_ids.reshape(-1), minlength=EXPERTS)
    if counts.tolist() != expected_counts:
        raise ValueError(f"{directory}: routed counts differ")


def validate_r10_sqg_capture_view(
    root: Path,
    *,
    verify_hashes: bool,
    selected_layers: Sequence[int],
    expected_split: dict,
    load_document_plan: Callable[[Path], dict],
    sha256_file: Callable[[Path], str],
) -> dict:
    """Validate only the active wave while binding the complete 75-layer view."""

    manifest = _json(root / "capture_manifest.json")
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("evidence_mode") != "r10_bf16_exact_view_v1"
        or int(manifest.get("tokens_per_layer", -1)) != TOKENS
        or int(manifest.get("documents", -1)) != DOCUMENTS
        or manifest.get("split") != expected_split
        or manifest.get("zero_copy") is not True
    ):
        raise ValueError("R10 SQG capture-view root contract differs")
    all_layers = [int(value) for value in manifest.get("selected_layers", [])]
    if all_layers != list(range(3, 78)):
        raise ValueError("R10 SQG capture view must contain all routed layers 3..77")
    if not set(int(layer) for layer in selected_layers).issubset(all_layers):
        raise ValueError("active SQG wave is absent from R10 capture view")
    plan_record = manifest.get("document_plan", {})
    plan_path = root / str(plan_record.get("path"))
    plan = load_document_plan(plan_path)
    if (
        plan_record.get("path") != "document_plan.json"
        or plan_record.get("fingerprint") != plan["plan_fingerprint"]
        or (verify_hashes and plan_record.get("sha256") != sha256_file(plan_path))
    ):
        raise ValueError("R10 SQG document-plan binding differs")
    refs = manifest.get("layer_manifests", {})
    if set(refs) != {str(layer) for layer in range(3, 78)}:
        raise ValueError("R10 SQG layer-manifest domain differs")
    source_root = Path(str(manifest.get("source_flat_capture"))).resolve()
    if not source_root.is_dir():
        raise ValueError("R10 source flat capture is absent")

    for layer in selected_layers:
        directory = root / f"layer_{int(layer):03d}"
        layer_path = directory / "layer_manifest.json"
        layer_manifest = _json(layer_path)
        reference = refs[str(int(layer))]
        if (
            reference.get("path") != f"layer_{int(layer):03d}/layer_manifest.json"
            or (verify_hashes and reference.get("sha256") != sha256_file(layer_path))
            or layer_manifest.get("schema") != LAYER_SCHEMA
            or int(layer_manifest.get("layer", -1)) != int(layer)
            or int(layer_manifest.get("tokens", -1)) != TOKENS
            or int(layer_manifest.get("documents_verified", -1)) != DOCUMENTS
            or layer_manifest.get("document_plan_fingerprint")
            != plan["plan_fingerprint"]
            or int(layer_manifest.get("hidden", -1)) != HIDDEN
            or int(layer_manifest.get("topk", -1)) != TOPK
            or int(layer_manifest.get("num_experts", -1)) != EXPERTS
            or not math.isclose(
                float(layer_manifest.get("routed_scaling_factor", math.nan)),
                ROUTED_SCALE,
            )
        ):
            raise ValueError(f"layer {layer}: R10 SQG layer contract differs")
        files = layer_manifest.get("files", {})
        for name, (bytes_per_row, dtype, shape) in FILE_ABI.items():
            payload = directory / name
            record = files.get(name, {})
            expected_bytes = TOKENS * bytes_per_row
            if (
                not payload.is_file()
                or payload.stat().st_size != expected_bytes
                or int(record.get("bytes", -1)) != expected_bytes
                or record.get("dtype") != dtype
                or record.get("shape") != shape
                or (verify_hashes and record.get("sha256") != sha256_file(payload))
            ):
                raise ValueError(f"layer {layer}: {name} ABI differs")
        # The three large files must still be the source payload inodes. This
        # is a cheap, stronger check than trusting copied bytes when hashes are
        # deliberately skipped on the timed path.
        for source_name, view_name in (
            ("x.bin", "hidden.bf16.bin"),
            ("ids.bin", "topk_ids.u8.bin"),
            ("weights.bin", "topk_weights.f32le.bin"),
        ):
            if not os.path.samefile(
                source_root / f"layer_{int(layer):03d}" / source_name,
                directory / view_name,
            ):
                raise ValueError(f"layer {layer}: {view_name} is not zero-copy")
        _validate_document_vectors(directory, plan["documents"])
        _validate_routes(
            directory, [int(value) for value in layer_manifest["routed_counts"]]
        )
    manifest["evidence_mode"] = "r10_bf16_exact_view_v1"
    return manifest
