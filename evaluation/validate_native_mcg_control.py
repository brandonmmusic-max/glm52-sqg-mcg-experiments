#!/usr/bin/env python3
"""Fail closed unless a checkpoint is the sealed all-MCG native control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any

from safetensors import safe_open
import torch


LAYERS = (6, 28, 52, 77)
PROJECTIONS = ("down_proj", "gate_proj", "up_proj")
EXPERTS = 256
MCG_MARKER_U32 = 0xCBAC1FED
MCG_MARKER_I32 = MCG_MARKER_U32 - (1 << 32)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
RECEIPT_SCHEMA = "glm52-r33-teacher-identity-receipt-v1"
RECEIPT_SEAL_SCHEMA = "glm52-r33-teacher-identity-seal-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read exact JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_real_file(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"control input must not be a symlink: {path}")
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise ValueError(f"cannot stat control input: {path}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"control input must be a regular file: {path}")


def _expected_bases(layer: int) -> set[str]:
    return {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        for expert in range(EXPERTS)
        for projection in PROJECTIONS
    }


def _receipt_files(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("teacher receipt schema differs")
    seal = receipt.get("seal")
    if not isinstance(seal, dict) or seal.get("schema") != RECEIPT_SEAL_SCHEMA:
        raise ValueError("teacher receipt seal schema differs")
    entries = seal.get("files")
    if not isinstance(entries, list):
        raise ValueError("teacher receipt files must be a list")
    by_path: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("teacher receipt contains a non-object file entry")
        name = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("bytes")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(digest, str)
            or not SHA256_RE.fullmatch(digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or name in by_path
        ):
            raise ValueError("teacher receipt contains an invalid file entry")
        by_path[name] = entry
    return by_path


def _verify_receipt_file(
    model: Path, name: str, receipt_files: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    entry = receipt_files.get(name)
    if entry is None:
        raise ValueError(f"teacher receipt does not seal required file: {name}")
    path = model / name
    _require_real_file(path)
    size = path.stat().st_size
    if size != entry["bytes"]:
        raise ValueError(
            f"sealed file size differs for {name}: {size} != {entry['bytes']}"
        )
    digest = _sha256_file(path)
    if digest != entry["sha256"]:
        raise ValueError(f"sealed file SHA-256 differs for {name}")
    return {"bytes": size, "sha256": digest, "role": entry.get("role")}


def validate_native_mcg_control(
    model: str | Path,
    *,
    expected_model: str | Path,
    teacher_receipt: str | Path,
    manifest_verified_sha256: str,
) -> dict[str, Any]:
    """Validate selected-layer all-MCG topology and sealed source identity."""

    if not SHA256_RE.fullmatch(manifest_verified_sha256):
        raise ValueError("manifest marker SHA-256 must be 64 lowercase hex digits")
    model_path = Path(model).resolve(strict=True)
    expected_path = Path(expected_model).resolve(strict=True)
    if model_path != expected_path:
        raise ValueError(
            f"control model must be the protected checkpoint: {model_path} != "
            f"{expected_path}"
        )
    if model_path.is_symlink() or not model_path.is_dir():
        raise ValueError("protected checkpoint must be a real directory")

    receipt_path = Path(teacher_receipt).resolve(strict=True)
    _require_real_file(receipt_path)
    receipt = _load_json(receipt_path)
    receipt_files = _receipt_files(receipt)

    marker_path = model_path / ".manifest_verified"
    _require_real_file(marker_path)
    marker_sha256 = _sha256_file(marker_path)
    if marker_sha256 != manifest_verified_sha256:
        raise ValueError("protected checkpoint verification marker differs")

    required_names = [
        "config.json",
        "quantization_config.json",
        "model.safetensors.index.json",
    ]
    for layer in LAYERS:
        required_names.extend(
            [
                f"r7-experts-layer-{layer:03d}.json",
                f"r7-experts-layer-{layer:03d}.safetensors",
            ]
        )
    verified_files = {
        name: _verify_receipt_file(model_path, name, receipt_files)
        for name in required_names
    }

    config = _load_json(model_path / "config.json")
    quant_file = _load_json(model_path / "quantization_config.json")
    embedded_quant = config.get("quantization_config")
    if not isinstance(embedded_quant, dict):
        raise ValueError("config.json lacks quantization_config object")
    for label, quant in (("config", embedded_quant), ("quantization", quant_file)):
        if quant.get("codebook") != "mcg":
            raise ValueError(f"{label} control codebook is not MCG")
        if quant.get("codebook_overrides") not in (None, {}):
            raise ValueError(f"{label} contains codebook overrides")
        if quant.get("codebook_tensor_overrides") not in (None, {}):
            raise ValueError(f"{label} contains tensor codebook overrides")

    index = _load_json(model_path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index lacks a nonempty weight_map")

    layer_reports: list[dict[str, Any]] = []
    for layer in LAYERS:
        bases = _expected_bases(layer)
        shard_name = f"r7-experts-layer-{layer:03d}.safetensors"
        sidecar_name = f"r7-experts-layer-{layer:03d}.json"
        expected_mcg = {f"{base}.mcg" for base in bases}
        expected_trellis = {f"{base}.trellis" for base in bases}
        prefix = f"model.layers.{layer}.mlp.experts."
        indexed_mcg = {key for key in weight_map if key.startswith(prefix) and key.endswith(".mcg")}
        indexed_sqg = {key for key in weight_map if key.startswith(prefix) and key.endswith(".sqg")}
        indexed_trellis = {
            key for key in weight_map if key.startswith(prefix) and key.endswith(".trellis")
        }
        if indexed_sqg:
            raise ValueError(f"layer {layer} index contains SQG markers")
        if indexed_mcg != expected_mcg or indexed_trellis != expected_trellis:
            raise ValueError(f"layer {layer} index marker/trellis census differs")
        if any(weight_map[key] != shard_name for key in expected_mcg | expected_trellis):
            raise ValueError(f"layer {layer} index points selected tensors outside {shard_name}")

        sidecar = _load_json(model_path / sidecar_name)
        if sidecar.get("layer") != layer or sidecar.get("shard") != shard_name:
            raise ValueError(f"layer {layer} sidecar binding differs")
        if sidecar.get("marker") != "CODEX_ROUND7":
            raise ValueError(f"layer {layer} sidecar marker differs")
        bit_map = sidecar.get("bit_map")
        if not isinstance(bit_map, dict) or set(bit_map) != bases:
            raise ValueError(f"layer {layer} sidecar bit map differs")
        payload_hashes = sidecar.get("payload_sha256")
        if not isinstance(payload_hashes, dict):
            raise ValueError(f"layer {layer} sidecar payload hashes are absent")
        if any(key.endswith(".sqg") for key in payload_hashes):
            raise ValueError(f"layer {layer} sidecar contains SQG payload lineage")
        if not expected_mcg.issubset(payload_hashes) or not expected_trellis.issubset(
            payload_hashes
        ):
            raise ValueError(f"layer {layer} sidecar omits selected MCG payloads")

        shard_path = model_path / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            actual_mcg = {key for key in keys if key.startswith(prefix) and key.endswith(".mcg")}
            actual_sqg = {key for key in keys if key.startswith(prefix) and key.endswith(".sqg")}
            actual_trellis = {
                key for key in keys if key.startswith(prefix) and key.endswith(".trellis")
            }
            if actual_sqg:
                raise ValueError(f"layer {layer} shard contains SQG markers")
            if actual_mcg != expected_mcg or actual_trellis != expected_trellis:
                raise ValueError(f"layer {layer} shard marker/trellis census differs")
            for key in sorted(expected_mcg):
                marker = handle.get_tensor(key)
                if (
                    marker.dtype != torch.int32
                    or tuple(marker.shape) != ()
                    or int(marker.item()) != MCG_MARKER_I32
                ):
                    raise ValueError(f"invalid exclusive MCG marker payload: {key}")

        layer_reports.append(
            {
                "layer": layer,
                "trellis_tensors": len(expected_trellis),
                "mcg_markers": len(expected_mcg),
                "sqg_markers": 0,
                "exclusive_mcg_marker_payloads_verified": True,
                "shard": shard_name,
                "shard_sha256": verified_files[shard_name]["sha256"],
                "sidecar": sidecar_name,
                "sidecar_sha256": verified_files[sidecar_name]["sha256"],
            }
        )

    report: dict[str, Any] = {
        "schema": "glm52-native-mcg-control-preflight-v1",
        "valid": True,
        "protected_model": str(model_path),
        "teacher_receipt": str(receipt_path),
        "teacher_receipt_sha256": _sha256_file(receipt_path),
        "teacher_seal_sha256": receipt.get("seal_sha256"),
        "manifest_verified_sha256": marker_sha256,
        "selected_layers": list(LAYERS),
        "global_codebook": "mcg",
        "selected_trellis_tensors": len(LAYERS) * EXPERTS * len(PROJECTIONS),
        "selected_mcg_markers": len(LAYERS) * EXPERTS * len(PROJECTIONS),
        "selected_sqg_markers": 0,
        "codebook_overrides": 0,
        "tensor_overrides": 0,
        "protected_model_read_only_mount_required": True,
        "verified_files": verified_files,
        "layers": layer_reports,
    }
    report["preflight_id"] = _canonical_sha256(report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--expected-model", type=Path, required=True)
    parser.add_argument("--teacher-receipt", type=Path, required=True)
    parser.add_argument("--manifest-verified-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = validate_native_mcg_control(
        args.model,
        expected_model=args.expected_model,
        teacher_receipt=args.teacher_receipt,
        manifest_verified_sha256=args.manifest_verified_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
