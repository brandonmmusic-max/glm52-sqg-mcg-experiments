from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

import src.glm52_bf16_source as source
from src.seal_bf16_source import build_source_seal


def _write_index(
    path: Path,
    weight_map: dict[str, str],
    *,
    total_size: int,
) -> None:
    payload = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    path.write_bytes(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )


def _patch_tiny_manifest(
    monkeypatch: pytest.MonkeyPatch,
    *,
    index_path: Path,
    config_path: Path,
    shard_path: Path,
    index_total_size: int,
) -> None:
    header = source.read_safetensors_header(shard_path)
    identity = source.ShardIdentity(
        bytes=shard_path.stat().st_size,
        sha256=source.sha256_file(shard_path),
        header_sha256=header.header_sha256,
        tensor_count=len(header.tensors),
    )
    monkeypatch.setattr(source, "EXPECTED_INDEX_BYTES", index_path.stat().st_size)
    monkeypatch.setattr(source, "EXPECTED_INDEX_SHA256", source.sha256_file(index_path))
    monkeypatch.setattr(source, "EXPECTED_INDEX_TOTAL_BYTES", index_total_size)
    index = json.loads(index_path.read_text())
    monkeypatch.setattr(source, "EXPECTED_INDEX_TENSOR_COUNT", len(index["weight_map"]))
    monkeypatch.setattr(
        source, "EXPECTED_CONFIG_SHA256", source.sha256_file(config_path)
    )
    monkeypatch.setattr(source, "EXPECTED_SHARDS", {shard_path.name: identity})
    monkeypatch.setattr(source, "EXPECTED_TOTAL_SHARD_BYTES", identity.bytes)


def _make_tiny_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    dtype: torch.dtype = torch.bfloat16,
    shape_override: dict[str, tuple[int, int]] | None = None,
    extra_tensor: bool = False,
) -> dict[str, Any]:
    monkeypatch.setattr(source, "SELECTED_LAYERS", (6,))
    monkeypatch.setattr(source, "NUM_EXPERTS", 1)
    monkeypatch.setattr(source, "HIDDEN_SIZE", 4)
    monkeypatch.setattr(source, "INTERMEDIATE_SIZE", 2)

    shape_override = shape_override or {}
    shard_path = tmp_path / "sample.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    weight_map: dict[str, str] = {}
    for position, projection in enumerate(source.PROJECTIONS, start=1):
        name = source.tensor_name(6, 0, projection)
        shape = shape_override.get(projection, source.expected_shape(projection))
        tensors[name] = torch.full(shape, position, dtype=dtype)
        weight_map[name] = shard_path.name
    if extra_tensor:
        tensors["model.unindexed.weight"] = torch.zeros((1,), dtype=dtype)
    save_file(tensors, shard_path)

    index_total_size = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    index_path = tmp_path / "model.safetensors.index.json"
    _write_index(index_path, weight_map, total_size=index_total_size)
    config_path = tmp_path / "config.json"
    config_path.write_bytes(
        (
            json.dumps(
                {
                    "hidden_size": 4,
                    "model_type": "glm_moe_dsa",
                    "moe_intermediate_size": 2,
                    "n_routed_experts": 1,
                    "num_hidden_layers": 78,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    )
    _patch_tiny_manifest(
        monkeypatch,
        index_path=index_path,
        config_path=config_path,
        shard_path=shard_path,
        index_total_size=index_total_size,
    )
    return {
        "config": config_path,
        "index": index_path,
        "index_total_size": index_total_size,
        "shard": shard_path,
        "weight_map": weight_map,
    }


def _provider(files: dict[str, Any]) -> source.BF16ExpertSource:
    return source.BF16ExpertSource(
        index_path=files["index"],
        config_path=files["config"],
        shard_root=files["shard"].parent,
        source_revision=source.SOURCE_REVISION,
    )


def _rebind_expected_index(
    monkeypatch: pytest.MonkeyPatch,
    index_path: Path,
    *,
    tensor_count: int,
) -> None:
    monkeypatch.setattr(source, "EXPECTED_INDEX_BYTES", index_path.stat().st_size)
    monkeypatch.setattr(source, "EXPECTED_INDEX_SHA256", source.sha256_file(index_path))
    monkeypatch.setattr(source, "EXPECTED_INDEX_TENSOR_COUNT", tensor_count)


def test_tensor_names_shapes_and_frozen_manifest() -> None:
    assert source.tensor_name(6, 0, "gate_proj") == (
        "model.layers.6.mlp.experts.0.gate_proj.weight"
    )
    assert source.expected_shape("gate_proj") == (2048, 6144)
    assert source.expected_shape("down_proj") == (6144, 2048)
    assert len(source.EXPECTED_SHARDS) == 18
    assert sum(item.bytes for item in source.EXPECTED_SHARDS.values()) == (
        96_536_706_248
    )
    assert source.EXPECTED_INDEX_SHA256 == (
        "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e"
    )
    assert source.EXPECTED_CONFIG_SHA256 == (
        "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a"
    )
    with pytest.raises(ValueError):
        source.tensor_name(7, 0, "gate_proj")


def test_valid_source_loads_fresh_fp32_and_seals(tmp_path, monkeypatch) -> None:
    files = _make_tiny_source(tmp_path, monkeypatch)
    provider = _provider(files)
    assert provider.required_shards() == ("sample.safetensors",)
    expert = provider.load_expert(6, 0)
    assert expert.gate_hf.dtype == torch.float32
    assert tuple(expert.gate_hf.shape) == (2, 4)
    assert tuple(expert.down_hf.shape) == (4, 2)
    assert expert.gate_hf.unique().tolist() == [1.0]
    assert expert.up_hf.unique().tolist() == [2.0]
    assert expert.down_hf.unique().tolist() == [3.0]
    assert set(expert.tensor_sha256) == set(source.PROJECTIONS)

    strict = provider.load_expert_bf16(6, 0)
    assert strict.gate_hf.dtype == torch.bfloat16
    assert strict.up_hf.dtype == torch.bfloat16
    assert strict.down_hf.dtype == torch.bfloat16
    assert strict.tensor_sha256 == expert.tensor_sha256

    seal = build_source_seal(
        index_path=files["index"],
        config_path=files["config"],
        shard_root=tmp_path,
    )
    assert seal["schema"] == "glm52-fresh-sqg-bf16-source-v2"
    assert seal["complete_index_header_binding_validated"] is True
    assert seal["shard_count"] == 1
    assert seal["target_tensor_count"] == 3
    assert seal["total_shard_bytes"] == files["shard"].stat().st_size


def test_revision_fails_before_any_file_trust(tmp_path) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {}}))
    with pytest.raises(ValueError, match="source revision drift"):
        source.BF16ExpertSource(
            index_path=index,
            shard_root=tmp_path,
            source_revision="main",
        )


def test_index_digest_drift_same_size_fails_closed(tmp_path, monkeypatch) -> None:
    files = _make_tiny_source(tmp_path, monkeypatch)
    raw = files["index"].read_bytes()
    needle = str(files["index_total_size"]).encode()
    replacement = str(files["index_total_size"] + 1).encode()
    assert len(needle) == len(replacement)
    files["index"].write_bytes(raw.replace(needle, replacement, 1))
    with pytest.raises(ValueError, match="index SHA256 mismatch"):
        _provider(files)


def test_config_digest_drift_same_size_fails_closed(tmp_path, monkeypatch) -> None:
    files = _make_tiny_source(tmp_path, monkeypatch)
    raw = files["config"].read_bytes()
    changed = raw.replace(b"glm_moe_dsa", b"glm_moe_dsx", 1)
    assert len(changed) == len(raw) and changed != raw
    files["config"].write_bytes(changed)
    with pytest.raises(ValueError, match="config SHA256 mismatch"):
        _provider(files)


def test_shard_size_drift_fails_closed(tmp_path, monkeypatch) -> None:
    files = _make_tiny_source(tmp_path, monkeypatch)
    with files["shard"].open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError, match="shard size mismatch"):
        _provider(files)


def test_shard_payload_drift_same_size_fails_closed(tmp_path, monkeypatch) -> None:
    files = _make_tiny_source(tmp_path, monkeypatch)
    with files["shard"].open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 1]))
    with pytest.raises(ValueError, match="shard SHA256 mismatch"):
        _provider(files)


def test_missing_sampled_tensor_in_index_fails_closed(tmp_path, monkeypatch) -> None:
    files = _make_tiny_source(tmp_path, monkeypatch)
    changed_map = dict(files["weight_map"])
    changed_map.pop(source.tensor_name(6, 0, "gate_proj"))
    _write_index(files["index"], changed_map, total_size=files["index_total_size"])
    _rebind_expected_index(monkeypatch, files["index"], tensor_count=2)
    with pytest.raises(ValueError, match="missing sampled tensors"):
        _provider(files)


def test_sampled_tensor_to_shard_binding_fails_closed(tmp_path, monkeypatch) -> None:
    files = _make_tiny_source(tmp_path, monkeypatch)
    changed_map = dict(files["weight_map"])
    changed_map[source.tensor_name(6, 0, "gate_proj")] = "wrong.safetensors"
    _write_index(files["index"], changed_map, total_size=files["index_total_size"])
    _rebind_expected_index(monkeypatch, files["index"], tensor_count=3)
    with pytest.raises(ValueError, match="shard binding mismatch"):
        _provider(files)


def test_complete_index_header_binding_is_required(tmp_path, monkeypatch) -> None:
    files = _make_tiny_source(tmp_path, monkeypatch, extra_tensor=True)
    with pytest.raises(ValueError, match="complete index/header binding mismatch"):
        _provider(files)


def test_wrong_dtype_is_rejected_during_validation(tmp_path, monkeypatch) -> None:
    files = _make_tiny_source(tmp_path, monkeypatch, dtype=torch.float16)
    with pytest.raises(TypeError, match="expected BF16"):
        _provider(files)


def test_wrong_shape_is_rejected_during_validation(tmp_path, monkeypatch) -> None:
    files = _make_tiny_source(
        tmp_path,
        monkeypatch,
        shape_override={"gate_proj": (4, 2)},
    )
    with pytest.raises(ValueError, match=r"expected \(2, 4\), got \(4, 2\)"):
        _provider(files)


def test_post_validation_shard_mutation_is_rejected_before_load(
    tmp_path, monkeypatch
) -> None:
    files = _make_tiny_source(tmp_path, monkeypatch)
    provider = _provider(files)
    with files["shard"].open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 1]))
    with pytest.raises(ValueError, match="changed before use"):
        provider.load_expert(6, 0)


def test_manifest_hash_strings_are_well_formed() -> None:
    for identity in source.EXPECTED_SHARDS.values():
        assert len(identity.sha256) == 64
        assert len(identity.header_sha256) == 64
        int(identity.sha256, 16)
        int(identity.header_sha256, 16)
    assert hashlib.sha256().digest_size == 32
