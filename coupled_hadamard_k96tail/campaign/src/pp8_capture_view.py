"""Fail-closed validation for zero-copy PP8 and MTP78 SQG capture views."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


SCHEMA = "glm52-pp8-sqg-capture-view-v1"
LAYER_SCHEMA = "glm52-pp8-sqg-layer-view-v1"
HIDDEN = 6_144
TOPK = 8
EXPERTS = 256
ROUTED_SCALE = 2.5
FILE_ABI = {
    "hidden.bf16.bin": (HIDDEN * 2, "bfloat16-le"),
    "topk_ids.u8.bin": (TOPK, "uint8"),
    "topk_weights.f32le.bin": (TOPK * 4, "float32-le"),
    "doc_epochs.u32le.bin": (4, "uint32-le"),
    "token_positions.u16le.bin": (2, "uint16-le"),
    "role_ids.u8.bin": (1, "uint8"),
}


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _source_path(root: Path, view_root: Path, relative: str) -> Path:
    if relative.startswith("@view/"):
        return view_root / relative.removeprefix("@view/")
    return root / relative


def _validate_vectors(directory: Path, documents: list[dict], tokens: int) -> None:
    epochs = np.memmap(directory / "doc_epochs.u32le.bin", mode="r", dtype="<u4")
    positions = np.memmap(
        directory / "token_positions.u16le.bin", mode="r", dtype="<u2"
    )
    roles = np.memmap(directory / "role_ids.u8.bin", mode="r", dtype="u1")
    offset = 0
    for document in documents:
        count = int(document["tokens"])
        end = offset + count
        if not bool(np.all(epochs[offset:end] == int(document["epoch"]))):
            raise ValueError(f"{directory}: document epoch vector differs")
        if not np.array_equal(positions[offset:end], np.arange(count, dtype="<u2")):
            raise ValueError(f"{directory}: document position vector differs")
        if not bool(np.all(roles[offset:end] == int(document["role_id"]))):
            raise ValueError(f"{directory}: document role vector differs")
        offset = end
    if offset != tokens:
        raise ValueError(f"{directory}: document vectors do not close")


def _validate_routes(directory: Path, tokens: int, expected_counts: list[int]) -> None:
    ids = np.memmap(
        directory / "topk_ids.u8.bin", mode="r", dtype="u1", shape=(tokens, TOPK)
    )
    weights = np.memmap(
        directory / "topk_weights.f32le.bin",
        mode="r",
        dtype="<f4",
        shape=(tokens, TOPK),
    )
    counts = np.zeros(EXPERTS, dtype=np.int64)
    for begin in range(0, tokens, 65_536):
        end = min(tokens, begin + 65_536)
        local_ids = np.asarray(ids[begin:end])
        local_weights = np.asarray(weights[begin:end])
        ordered = np.sort(local_ids, axis=1)
        if np.any(ordered[:, 1:] == ordered[:, :-1]):
            raise ValueError(f"{directory}: duplicate routed expert")
        if not np.isfinite(local_weights).all() or np.any(local_weights < 0):
            raise ValueError(f"{directory}: routed weights are invalid")
        if not np.allclose(
            local_weights.astype(np.float64).sum(axis=1),
            ROUTED_SCALE,
            rtol=2e-6,
            atol=2e-6,
        ):
            raise ValueError(f"{directory}: routed weights are not applied gates")
        counts += np.bincount(local_ids.reshape(-1), minlength=EXPERTS)
    if counts.tolist() != expected_counts:
        raise ValueError(f"{directory}: routed counts differ")


def validate_pp8_sqg_capture_view(
    root: Path,
    *,
    verify_hashes: bool,
    selected_layers: Sequence[int],
    expected_split: dict,
    load_document_plan: Callable[[Path], dict],
    sha256_file: Callable[[Path], str],
) -> dict:
    manifest = _json(root / "capture_manifest.json")
    evidence = manifest.get("evidence_mode")
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("complete") is not True
        or evidence
        not in {
            "official_bf16_pp8_exact_view_v1",
            "official_bf16_mtp78_exact_view_v1",
        }
        or manifest.get("split") != expected_split
        or manifest.get("zero_copy") is not True
    ):
        raise ValueError("PP8 SQG capture-view root contract differs")
    all_layers = [int(value) for value in manifest.get("selected_layers", [])]
    expected_domain = [78] if evidence.endswith("mtp78_exact_view_v1") else list(range(3, 78))
    if all_layers != expected_domain or not set(map(int, selected_layers)).issubset(all_layers):
        raise ValueError("PP8 SQG capture-view layer domain differs")
    tokens = int(manifest.get("tokens_per_layer", -1))
    documents = int(manifest.get("documents", -1))
    if tokens <= 0 or documents <= 0:
        raise ValueError("PP8 SQG capture-view population is empty")
    plan_record = manifest.get("document_plan", {})
    plan_path = root / str(plan_record.get("path"))
    plan = load_document_plan(plan_path)
    if (
        plan_record.get("path") != "document_plan.json"
        or int(plan.get("tokens_total", -1)) != tokens
        or int(plan.get("documents_total", -1)) != documents
        or plan_record.get("fingerprint") != plan.get("plan_fingerprint")
        or (verify_hashes and plan_record.get("sha256") != sha256_file(plan_path))
    ):
        raise ValueError("PP8 SQG document-plan binding differs")
    source_root = Path(str(manifest.get("source_capture", ""))).resolve()
    if not source_root.is_dir() or source_root.is_symlink():
        raise ValueError("PP8 source capture is absent or unsafe")
    source_manifest = manifest.get("source_manifest", {})
    source_manifest_path = source_root / str(source_manifest.get("path", ""))
    if (
        not source_manifest_path.is_file()
        or source_manifest_path.is_symlink()
        or sha256_file(source_manifest_path) != source_manifest.get("sha256")
    ):
        raise ValueError("PP8 source manifest binding differs")
    refs = manifest.get("layer_manifests", {})
    if set(refs) != {str(layer) for layer in all_layers}:
        raise ValueError("PP8 SQG layer-manifest domain differs")
    for layer in selected_layers:
        layer = int(layer)
        directory = root / f"layer_{layer:03d}"
        layer_path = directory / "layer_manifest.json"
        layer_manifest = _json(layer_path)
        reference = refs[str(layer)]
        if (
            reference.get("path") != f"layer_{layer:03d}/layer_manifest.json"
            or reference.get("sha256") != sha256_file(layer_path)
            or layer_manifest.get("schema") != LAYER_SCHEMA
            or int(layer_manifest.get("layer", -1)) != layer
            or int(layer_manifest.get("tokens", -1)) != tokens
            or int(layer_manifest.get("documents_verified", -1)) != documents
            or layer_manifest.get("document_plan_fingerprint")
            != plan.get("plan_fingerprint")
            or int(layer_manifest.get("hidden", -1)) != HIDDEN
            or int(layer_manifest.get("topk", -1)) != TOPK
            or int(layer_manifest.get("num_experts", -1)) != EXPERTS
            or not math.isclose(
                float(layer_manifest.get("routed_scaling_factor", math.nan)),
                ROUTED_SCALE,
            )
        ):
            raise ValueError(f"layer {layer}: PP8 SQG layer contract differs")
        files = layer_manifest.get("files", {})
        source_files = layer_manifest.get("source_files", {})
        for name, (bytes_per_row, dtype) in FILE_ABI.items():
            payload = directory / name
            record = files.get(name, {})
            expected_bytes = tokens * bytes_per_row
            expected_shape = (
                [tokens, HIDDEN]
                if name.startswith("hidden")
                else [tokens, TOPK]
                if name.startswith("topk")
                else [tokens]
            )
            if (
                not payload.is_file()
                or payload.is_symlink()
                or payload.stat().st_size != expected_bytes
                or int(record.get("bytes", -1)) != expected_bytes
                or record.get("dtype") != dtype
                or record.get("shape") != expected_shape
                or (
                    verify_hashes
                    and record.get("sha256") != sha256_file(payload)
                )
            ):
                raise ValueError(f"layer {layer}: {name} ABI differs")
            source = _source_path(source_root, root, str(source_files.get(name, "")))
            if not source.is_file() or not os.path.samefile(source, payload):
                raise ValueError(f"layer {layer}: {name} is not zero-copy")
        _validate_vectors(directory, plan["documents"], tokens)
        _validate_routes(
            directory,
            tokens,
            [int(value) for value in layer_manifest["routed_counts"]],
        )
    return manifest


__all__ = ["validate_pp8_sqg_capture_view"]
