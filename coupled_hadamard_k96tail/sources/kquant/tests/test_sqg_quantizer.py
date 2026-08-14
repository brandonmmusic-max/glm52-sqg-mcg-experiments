from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import kquant.sqg_quantizer as sqg_quantizer
from kquant.exl3_reference import reconstruct_trellis_states
from kquant.sqg_quantizer import install_sqg_quantizer
from kquant.sqg_e4m3 import sqg_xor_cheb_t12_bytes


def _sealed_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    filename: str = "kquant_sqg_quantize_ext_v22.so",
) -> tuple[Path, SimpleNamespace]:
    path = tmp_path / filename
    payload = b"sealed-sqg-extension-test-double"
    path.write_bytes(payload)
    module = SimpleNamespace(
        __file__=str(path),
        quantize_tiles_procedural=lambda *_args: None,
        quantize_tiles_sqg=lambda *_args: None,
    )
    monkeypatch.setenv("KQUANT_SQG_REQUIRE_PREBUILT", "1")
    monkeypatch.setenv("KQUANT_SQG_EXTENSION_PATH", str(path))
    monkeypatch.setenv(
        "KQUANT_SQG_EXTENSION_SHA256", hashlib.sha256(payload).hexdigest()
    )
    return path, module


def test_required_prebuilt_extension_uses_exact_path_and_never_jit_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, module = _sealed_extension(tmp_path, monkeypatch)
    imported: list[Path] = []

    def fake_import(actual: Path):
        imported.append(actual)
        return module

    monkeypatch.setattr(sqg_quantizer, "_import_prebuilt_extension", fake_import)
    monkeypatch.setattr(
        sqg_quantizer,
        "load",
        lambda **_kwargs: pytest.fail("JIT builder must not run in prebuilt mode"),
    )
    sqg_quantizer._extension.cache_clear()
    try:
        assert sqg_quantizer._extension() is module
        assert imported == [path.resolve()]
    finally:
        sqg_quantizer._extension.cache_clear()


def test_required_prebuilt_extension_rejects_wrong_filename_before_import_or_jit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sealed_extension(tmp_path, monkeypatch, filename="foreign_extension.so")
    monkeypatch.setattr(
        sqg_quantizer,
        "_import_prebuilt_extension",
        lambda _path: pytest.fail("foreign extension must not be imported"),
    )
    monkeypatch.setattr(
        sqg_quantizer,
        "load",
        lambda **_kwargs: pytest.fail("JIT builder must not run in prebuilt mode"),
    )
    sqg_quantizer._extension.cache_clear()
    try:
        with pytest.raises(ValueError, match="filename must identify"):
            sqg_quantizer._extension()
    finally:
        sqg_quantizer._extension.cache_clear()


def test_required_prebuilt_extension_rejects_hash_mismatch_before_import_or_jit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sealed_extension(tmp_path, monkeypatch)
    monkeypatch.setenv("KQUANT_SQG_EXTENSION_SHA256", "0" * 64)
    monkeypatch.setattr(
        sqg_quantizer,
        "_import_prebuilt_extension",
        lambda _path: pytest.fail("mismatched extension must not be imported"),
    )
    monkeypatch.setattr(
        sqg_quantizer,
        "load",
        lambda **_kwargs: pytest.fail("JIT builder must not run in prebuilt mode"),
    )
    sqg_quantizer._extension.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="SHA256 mismatch"):
            sqg_quantizer._extension()
    finally:
        sqg_quantizer._extension.cache_clear()


@pytest.mark.parametrize(
    ("missing", "message"),
    (
        ("KQUANT_SQG_EXTENSION_PATH", "are both required"),
        ("KQUANT_SQG_EXTENSION_SHA256", "are both required"),
    ),
)
def test_required_prebuilt_extension_rejects_partial_binding_without_jit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    message: str,
) -> None:
    _sealed_extension(tmp_path, monkeypatch)
    monkeypatch.delenv(missing)
    monkeypatch.setattr(
        sqg_quantizer,
        "load",
        lambda **_kwargs: pytest.fail("JIT builder must not run in prebuilt mode"),
    )
    sqg_quantizer._extension.cache_clear()
    try:
        with pytest.raises(RuntimeError, match=message):
            sqg_quantizer._extension()
    finally:
        sqg_quantizer._extension.cache_clear()


def test_rate_specific_none_dispatches_to_kquant_mcg(monkeypatch) -> None:
    calls: list[tuple[torch.Tensor, dict]] = []
    procedural_calls: list[tuple] = []

    def original(tiles: torch.Tensor, args: dict):
        calls.append((tiles, args))
        return "upstream"

    def procedural(*args):
        procedural_calls.append(args)

    monkeypatch.setattr(
        sqg_quantizer,
        "_sqg_temp_buffers",
        lambda _device, _bits: (torch.empty(0), torch.empty(0)),
    )
    monkeypatch.setattr(
        sqg_quantizer,
        "_extension",
        lambda: SimpleNamespace(quantize_tiles_procedural=procedural),
    )

    module = SimpleNamespace(quantize_tiles=original)
    install_sqg_quantizer(module)
    tiles = torch.zeros((1, 256), dtype=torch.float32)
    result = module.quantize_tiles(
        tiles,
        {
            "K": 2,
            "devices": ["cuda:0"],
            "sqg_e4m3_luts_by_bits": {2: None, 3: None, 4: None},
        },
    )

    assert all(isinstance(tensor, torch.Tensor) for tensor in result)
    assert not calls
    assert len(procedural_calls) == 1
    assert procedural_calls[0][5:] == (2, 1, 128)


def test_sqg_luts_are_transposed_by_predecessor_for_vector_loads(
    monkeypatch,
) -> None:
    calls: list[tuple] = []

    monkeypatch.setattr(
        sqg_quantizer,
        "_sqg_temp_buffers",
        lambda _device, _bits: (torch.empty(0), torch.empty(0)),
    )
    monkeypatch.setattr(
        sqg_quantizer,
        "_extension",
        lambda: SimpleNamespace(quantize_tiles_sqg=lambda *args: calls.append(args)),
    )

    module = SimpleNamespace(quantize_tiles=lambda *_args: "upstream")
    install_sqg_quantizer(module)
    tiles = torch.zeros((1, 256), dtype=torch.float32)
    for bits in (2, 3, 4):
        raw = torch.arange(65536, dtype=torch.int64).to(torch.uint8)
        module.quantize_tiles(
            tiles,
            {
                "K": bits,
                "devices": ["cuda:0"],
                "sqg_e4m3_lut": raw,
            },
        )
        actual = calls[-1][5]
        predecessors = 1 << bits
        out_edge_pairs = (65536 >> bits) // 2
        expected = (
            raw.reshape(predecessors, out_edge_pairs, 2)
            .permute(1, 0, 2)
            .contiguous()
            .reshape(-1)
        )
        assert torch.equal(actual, expected)
        assert calls[-1][6:] == (bits, 128)


@pytest.mark.parametrize("bits", [5, 6])
def test_k5_k6_sqg_luts_remain_in_direct_state_order(
    monkeypatch, bits: int
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(
        sqg_quantizer,
        "_sqg_temp_buffers",
        lambda _device, _bits: (torch.empty(0), torch.empty(0)),
    )
    monkeypatch.setattr(
        sqg_quantizer,
        "_extension",
        lambda: SimpleNamespace(quantize_tiles_sqg=lambda *args: calls.append(args)),
    )
    module = SimpleNamespace(quantize_tiles=lambda *_args: "upstream")
    install_sqg_quantizer(module)
    raw = torch.arange(65536, dtype=torch.int64).to(torch.uint8)
    module.quantize_tiles(
        torch.zeros((1, 256), dtype=torch.float32),
        {"K": bits, "devices": ["cuda:0"], "sqg_e4m3_lut": raw},
    )
    assert torch.equal(calls[-1][5], raw)
    assert calls[-1][6:] == (bits, 128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("bits", range(2, 7))
def test_sqg_quantizer_keeps_saturated_cost_paths_closed(bits: int) -> None:
    module = SimpleNamespace(quantize_tiles=lambda *_args: "upstream")
    install_sqg_quantizer(module)
    tiles = torch.full((2, 256), 1000.0, dtype=torch.float32, device="cuda")

    _output, states = module.quantize_tiles(
        tiles,
        {
            "K": bits,
            "devices": ["cuda:0"],
            "sqg_e4m3_lut": sqg_xor_cheb_t12_bytes(bits),
            "tailbite_context": 128,
        },
    )

    edges = states.to(torch.int64) & ((1 << bits) - 1)
    expected = reconstruct_trellis_states(edges, bits)
    assert torch.equal(states, expected)
