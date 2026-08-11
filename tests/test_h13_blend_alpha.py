from pathlib import Path

from scripts.encode_expert_local_h13_shard import _fixed_alpha_construction


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_alpha_construction_is_exact_and_stable() -> None:
    assert _fixed_alpha_construction(None).endswith("cap_0p75_v1")
    assert _fixed_alpha_construction(0.0).endswith(
        "fixed_alpha_0_layer_global_prior_v2"
    )
    assert _fixed_alpha_construction(0.25).endswith(
        "fixed_alpha_0p25_layer_global_prior_v2"
    )
    assert _fixed_alpha_construction(0.5).endswith(
        "fixed_alpha_0p5_layer_global_prior_v2"
    )
    assert _fixed_alpha_construction(0.75).endswith(
        "fixed_alpha_0p75_layer_global_prior_v2"
    )
    assert _fixed_alpha_construction(1.0).endswith(
        "fixed_alpha_1_layer_global_prior_v2"
    )


def test_late_panel_encodes_full_grid_and_holds_out_only_after_selection() -> None:
    source = (ROOT / "scripts/run_contiguous_late_blend_panel.sh").read_text()
    for label in ("sqg_a000", "sqg_a025", "sqg_a050", "sqg_a075", "sqg_a100"):
        assert label in source
    assert "FRESH_SQG_SELECTED_LAYERS=74,75,76,77" in source
    assert "selection \"$SELECTION_JSON\" mcg" in source
    assert "holdout \"$HOLDOUT_JSON\" mcg" in source
    assert "diagnostic_fallback_nonbaseline" in source
