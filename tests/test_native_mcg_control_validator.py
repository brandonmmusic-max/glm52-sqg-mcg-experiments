from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from evaluation.validate_native_mcg_control import (
    LAYERS,
    MCG_MARKER_I32,
    validate_native_mcg_control,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(
    root: Path,
    *,
    sqg_layer: int | None = None,
    bad_marker_layer: int | None = None,
    codebook_override: bool = False,
) -> tuple[Path, Path, str]:
    model = root / "protected"
    model.mkdir()
    marker = model / ".manifest_verified"
    marker.write_text("verified\n", encoding="utf-8")
    quant = {"codebook": "mcg"}
    if codebook_override:
        quant["codebook_overrides"] = {"6": "sqg_xor_cheb_t12"}
    (model / "config.json").write_text(
        json.dumps({"quantization_config": quant}), encoding="utf-8"
    )
    (model / "quantization_config.json").write_text(
        json.dumps(quant), encoding="utf-8"
    )

    weight_map: dict[str, str] = {}
    for layer in LAYERS:
        shard_name = f"r7-experts-layer-{layer:03d}.safetensors"
        tensors: dict[str, torch.Tensor] = {}
        bit_map: dict[str, int] = {}
        payloads: dict[str, str] = {}
        for expert in range(256):
            for projection in ("down_proj", "gate_proj", "up_proj"):
                base = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                marker_suffix = "sqg" if layer == sqg_layer else "mcg"
                marker_value = (
                    0 if layer == bad_marker_layer else MCG_MARKER_I32
                )
                tensors[f"{base}.{marker_suffix}"] = torch.tensor(
                    marker_value, dtype=torch.int32
                )
                tensors[f"{base}.trellis"] = torch.zeros(1, dtype=torch.int32)
                bit_map[base] = 3
                payloads[f"{base}.{marker_suffix}"] = "0" * 64
                payloads[f"{base}.trellis"] = "0" * 64
                weight_map[f"{base}.{marker_suffix}"] = shard_name
                weight_map[f"{base}.trellis"] = shard_name
        save_file(tensors, model / shard_name)
        sidecar_name = f"r7-experts-layer-{layer:03d}.json"
        (model / sidecar_name).write_text(
            json.dumps(
                {
                    "layer": layer,
                    "shard": shard_name,
                    "marker": "CODEX_ROUND7",
                    "bit_map": bit_map,
                    "payload_sha256": payloads,
                }
            ),
            encoding="utf-8",
        )
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )

    entries = []
    for path in sorted(model.iterdir()):
        if path.name == ".manifest_verified":
            continue
        entries.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "role": "test",
            }
        )
    receipt = root / "teacher.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "glm52-r33-teacher-identity-receipt-v1",
                "seal": {
                    "schema": "glm52-r33-teacher-identity-seal-v1",
                    "files": entries,
                },
                "seal_sha256": "1" * 64,
            }
        ),
        encoding="utf-8",
    )
    return model, receipt, _sha256(marker)


def test_validates_exact_all_mcg_selected_layers(tmp_path: Path) -> None:
    model, receipt, marker_sha256 = _write_fixture(tmp_path)
    report = validate_native_mcg_control(
        model,
        expected_model=model,
        teacher_receipt=receipt,
        manifest_verified_sha256=marker_sha256,
    )
    assert report["valid"] is True
    assert report["selected_layers"] == [6, 28, 52, 77]
    assert report["selected_trellis_tensors"] == 3072
    assert report["selected_mcg_markers"] == 3072
    assert report["selected_sqg_markers"] == 0
    assert all(layer["mcg_markers"] == 768 for layer in report["layers"])


def test_rejects_sqg_marker_in_control_layer(tmp_path: Path) -> None:
    model, receipt, marker_sha256 = _write_fixture(tmp_path, sqg_layer=28)
    with pytest.raises(ValueError, match="SQG markers"):
        validate_native_mcg_control(
            model,
            expected_model=model,
            teacher_receipt=receipt,
            manifest_verified_sha256=marker_sha256,
        )


def test_rejects_bad_mcg_marker_payload(tmp_path: Path) -> None:
    model, receipt, marker_sha256 = _write_fixture(tmp_path, bad_marker_layer=52)
    with pytest.raises(ValueError, match="invalid exclusive MCG marker payload"):
        validate_native_mcg_control(
            model,
            expected_model=model,
            teacher_receipt=receipt,
            manifest_verified_sha256=marker_sha256,
        )


def test_rejects_codebook_override_and_alternate_model(tmp_path: Path) -> None:
    model, receipt, marker_sha256 = _write_fixture(
        tmp_path, codebook_override=True
    )
    with pytest.raises(ValueError, match="codebook overrides"):
        validate_native_mcg_control(
            model,
            expected_model=model,
            teacher_receipt=receipt,
            manifest_verified_sha256=marker_sha256,
        )
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="protected checkpoint"):
        validate_native_mcg_control(
            model,
            expected_model=other,
            teacher_receipt=receipt,
            manifest_verified_sha256=marker_sha256,
        )
