from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = (
    ROOT
    / "evaluation/runtime_overlay/_overrides/vllm/model_executor/layers"
    / "quantization/exl3.py"
)
EXPECTED_FUSED = frozenset(
    {
        6, 7, 8,
        *range(10, 55),
    }
)


def _load_allowlist_helpers() -> dict[str, Any]:
    tree = ast.parse(OVERLAY.read_text())
    wanted = {
        "_r7_fused_layer_budget",
        "_r7_fused_layer_allowlist",
        "_r7_native_mcg_control_layers",
        "_r7_expected_reserved_layers",
        "_r7_layer_index_for_allowlist",
        "_reserve_preserved_r7_fused_slot",
        "_validate_preserved_r7_accounting",
    }
    wanted_constants = {
        "_PRESERVED_R7_FUSED_LAYERS",
        "_FRESH_SQG_TREATMENT_LAYERS",
    }
    definitions = [
        node
        for node in tree.body
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in wanted
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in wanted_constants
                for target in node.targets
            )
        )
    ]
    assert {
        node.name for node in definitions if isinstance(node, ast.FunctionDef)
    } == wanted
    namespace: dict[str, Any] = {
        "Any": Any,
        "logger": SimpleNamespace(info=lambda *args, **kwargs: None),
        "os": os,
        "re": re,
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(OVERLAY), "exec"), namespace)
    return namespace


def test_exact_preserved_baseline_allowlist_is_sealed_in_runtime_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    line = next(
        line
        for line in (ROOT / "evaluation/r33_exact_runtime.args").read_text().splitlines()
        if line.startswith("--env=VLLM_EXL3_R7_FUSED_ALLOWLIST=")
    )
    value = line.split("=", 2)[2]
    monkeypatch.setenv("VLLM_EXL3_R7_FUSED_ALLOWLIST", value)
    helpers = _load_allowlist_helpers()
    assert helpers["_r7_fused_layer_allowlist"]() == EXPECTED_FUSED
    assert len(EXPECTED_FUSED) == 48
    assert {6, 28, 52}.issubset(EXPECTED_FUSED)
    assert 77 not in EXPECTED_FUSED
    assert EXPECTED_FUSED.isdisjoint({55, 56, 57})

    evidence = json.loads(
        (ROOT / "evaluation/preserved_fused_path.json").read_text()
    )
    assert frozenset(evidence["fused_layers_all_five_runs"]) == EXPECTED_FUSED
    assert evidence["selected_sqg_layers_that_reserve_baseline_slots"] == [6, 28, 52]
    assert evidence["selected_sqg_layers_nonfused_in_baseline"] == [77]
    assert evidence["substitution_layers_forbidden"] == [55, 56, 57]


@pytest.mark.parametrize(
    "value",
    ["6,6", "6,,7", "6, 7", "06,7", "-1,6", "six,7"],
)
def test_fused_allowlist_rejects_ambiguous_or_noncanonical_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("VLLM_EXL3_R7_FUSED_ALLOWLIST", value)
    helpers = _load_allowlist_helpers()
    with pytest.raises(ValueError):
        helpers["_r7_fused_layer_allowlist"]()


def test_fused_allowlist_layer_parser_fails_closed() -> None:
    helper = _load_allowlist_helpers()["_r7_layer_index_for_allowlist"]
    assert helper(SimpleNamespace(layer_name="model.layers.52.mlp.experts")) == 52
    with pytest.raises(ValueError):
        helper(SimpleNamespace(layer_name="model.mlp.experts"))


@pytest.mark.parametrize(
    ("value", "valid"),
    [("6,28,52", False), ("6,28,52,77,78", False), ("77,52,28,6", True)],
)
def test_native_mcg_control_accepts_only_canonical_preregistered_arm(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    valid: bool,
) -> None:
    monkeypatch.setenv("VLLM_EXL3_NATIVE_MCG_CONTROL_LAYERS", value)
    helper = _load_allowlist_helpers()["_r7_native_mcg_control_layers"]
    if not valid:
        with pytest.raises(ValueError):
            helper()
    else:
        assert helper() == frozenset({6, 28, 52, 77})


def test_preserved_slot_reservation_and_final_accounting_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "VLLM_EXL3_R7_FUSED_ALLOWLIST",
        ",".join(str(value) for value in sorted(EXPECTED_FUSED)),
    )
    monkeypatch.setenv("VLLM_EXL3_R7_FUSED_LAYERS", "48")
    monkeypatch.setenv("VLLM_EXL3_R7_EXPECT_RESERVED_LAYERS", "6,28,52")
    helpers = _load_allowlist_helpers()
    reserve = helpers["_reserve_preserved_r7_fused_slot"]
    validate = helpers["_validate_preserved_r7_accounting"]
    quant = SimpleNamespace(_r7_fused_layers=45)
    assert reserve(quant, SimpleNamespace(layer_name="model.layers.6.mlp.experts")) == 46
    assert reserve(quant, SimpleNamespace(layer_name="model.layers.28.mlp.experts")) == 47
    assert reserve(quant, SimpleNamespace(layer_name="model.layers.52.mlp.experts")) == 48
    validate(quant, SimpleNamespace(layer_name="model.layers.55.mlp.experts"))
    assert quant._r7_fused_accounting_logged is True
    with pytest.raises(RuntimeError, match="duplicate"):
        reserve(quant, SimpleNamespace(layer_name="model.layers.52.mlp.experts"))

    broken = SimpleNamespace(
        _r7_fused_layers=48,
        _r7_reserved_fused_layer_indices=(6, 28),
    )
    with pytest.raises(RuntimeError, match="accounting differs"):
        validate(broken, SimpleNamespace(layer_name="model.layers.55.mlp.experts"))


def test_three_runtime_arms_differ_only_in_preregistered_control_env() -> None:
    paths = {
        name: ROOT / f"evaluation/{name}.args"
        for name in (
            "r33_exact_runtime",
            "r33_native_mcg_control",
            "r33_overlay_historical",
        )
    }
    lines = {name: path.read_text().splitlines() for name, path in paths.items()}
    control_prefixes = (
        "--env=VLLM_EXL3_R7_EXPECT_RESERVED_LAYERS=",
        "--env=VLLM_EXL3_NATIVE_MCG_CONTROL_LAYERS=",
    )
    common = {
        name: [
            line for line in values if not line.startswith(control_prefixes)
        ]
        for name, values in lines.items()
    }
    assert common["r33_exact_runtime"] == common["r33_native_mcg_control"]
    assert common["r33_exact_runtime"] == common["r33_overlay_historical"]
    assert "--env=VLLM_EXL3_R7_EXPECT_RESERVED_LAYERS=6,28,52" in lines[
        "r33_exact_runtime"
    ]
    assert "--env=VLLM_EXL3_NATIVE_MCG_CONTROL_LAYERS=6,28,52,77" in lines[
        "r33_native_mcg_control"
    ]
    assert "--env=VLLM_EXL3_R7_EXPECT_RESERVED_LAYERS=none" in lines[
        "r33_overlay_historical"
    ]


def test_allowlist_guard_precedes_fused_packing_and_sqg_reserves_slot() -> None:
    source = OVERLAY.read_text()
    prepare_start = source.index("    def _prepare_r7_sparkinfer_weights")
    split_at = source.index("        split = self._r7_projection_tiers(layer)", prepare_start)
    guard_at = source.index("        fused_allowlist = _r7_fused_layer_allowlist()", prepare_start)
    assert guard_at < split_at

    sqg_start = source.index("        if sqg_payloads:")
    sqg_end = source.index("        if getattr(layer, \"exl3_r7_graph\", False):", sqg_start)
    sqg_block = source[sqg_start:sqg_end]
    assert "_reserve_preserved_r7_fused_slot(self.quant_config, layer)" in sqg_block
    assert "no later MCG layer can substitute" in sqg_block
    assert "native-MCG control requires one exclusive MCG marker" in sqg_block
