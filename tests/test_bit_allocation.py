from __future__ import annotations

import pytest

from src.export_bit_allocation import (
    EXPECTED_HISTOGRAM,
    build_contract,
    validate_layer_map,
)


def make_map(layer: int) -> dict[str, int]:
    result: dict[str, int] = {}
    ordinal = 0
    for expert in range(256):
        for projection in ("down_proj", "gate_proj", "up_proj"):
            name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
            result[name] = 3 if ordinal < EXPECTED_HISTOGRAM["3"] else 4
            ordinal += 1
    return result


def test_complete_k3_k4_map_passes() -> None:
    result = validate_layer_map(6, make_map(6))
    assert len(result) == 768
    assert sum(bits == 3 for bits in result.values()) == 384
    assert sum(bits == 4 for bits in result.values()) == 384


def test_k5_is_rejected() -> None:
    value = make_map(6)
    value[next(iter(value))] = 5
    with pytest.raises(ValueError, match="only K3/K4"):
        validate_layer_map(6, value)


def test_cross_layer_name_is_rejected() -> None:
    value = make_map(6)
    old_name = next(iter(value))
    bits = value.pop(old_name)
    value[old_name.replace("layers.6", "layers.7")] = bits
    with pytest.raises(ValueError, match="cross-layer"):
        validate_layer_map(6, value)


def test_exported_contract_retains_only_bit_assignments(
    tmp_path, monkeypatch
) -> None:
    import src.export_bit_allocation as allocation

    monkeypatch.setattr(allocation, "LAYERS", (6,))
    source = tmp_path / "source"
    source.mkdir()
    (source / "r7-experts-layer-006.json").write_text(
        __import__("json").dumps({"layer": 6, "bit_map": make_map(6)})
    )

    contract = build_contract(source)
    layer = contract["layers"]["6"]
    assert set(layer) == {"bit_map", "histogram"}
    serialized = __import__("json").dumps(contract).lower()
    for forbidden in (
        "sidecar",
        "mcg",
        "suh",
        "svh",
        "scale",
        "sign",
        "seed",
        "permutation",
        "trellis",
        "packed",
        "codebook",
    ):
        assert forbidden not in serialized
