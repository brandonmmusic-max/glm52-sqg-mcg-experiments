from __future__ import annotations

import numpy as np

from scripts.evaluate_layerwise_h13_mapping import evaluate
from scripts.select_layerwise_h13_blend import select_layerwise


def test_layerwise_selector_combines_existing_error_vectors(tmp_path):
    rows = 200
    source = {
        "aggregate": {
            "role": "selection",
            "metrics": {"alpha0": {}, "alpha1": {}},
            "selection_policy": {"hard_constraints": ["test constraint"]},
        },
        "layers": {},
    }
    for layer in (6, 7):
        path = tmp_path / f"layer-{layer}.npz"
        baseline = np.full(rows, 2.0, dtype=np.float64)
        candidate = np.full(rows, 1.0, dtype=np.float64)
        np.savez_compressed(
            path,
            doc_epochs=np.zeros(rows, dtype=np.uint32),
            token_positions=np.arange(rows, dtype=np.uint16),
            reference_energy=np.full(rows, 10.0, dtype=np.float64),
            error__alpha0=baseline,
            error__alpha1=candidate,
        )
        source["layers"][str(layer)] = {"row_output": str(path)}

    result = select_layerwise(source, baseline_label="alpha0")

    assert result["search_space"] == 4
    assert result["eligible_count"] == 3
    assert result["winner"]["layer_to_label"] == {
        "6": "alpha1",
        "7": "alpha1",
    }
    assert result["winner"]["tail_vs_baseline"]["improved_fraction"] == 1.0
    assert result["recommended_mapping"] == {"6": "alpha1", "7": "alpha1"}

    confirmation = evaluate(source, result, baseline_label="alpha0")
    assert confirmation["mapping"] == {"6": "alpha1", "7": "alpha1"}
    assert confirmation["passes_all_hard_constraints"] is True


def test_layerwise_selector_can_forbid_baseline_layers(tmp_path):
    rows = 200
    source = {
        "aggregate": {
            "role": "selection",
            "metrics": {
                "mcg": {"signed_top8_nmse": 0.2},
                "sqg_a": {"signed_top8_nmse": 0.1},
            },
            "selection_policy": {
                "winner": "mcg",
                "diagnostic_fallback_nonbaseline": "sqg_a",
                "hard_constraints": ["test constraint"],
            },
        },
        "layers": {},
    }
    for layer in (74, 75):
        path = tmp_path / f"layer-{layer}.npz"
        np.savez_compressed(
            path,
            doc_epochs=np.zeros(rows, dtype=np.uint32),
            token_positions=np.arange(rows, dtype=np.uint16),
            reference_energy=np.full(rows, 10.0, dtype=np.float64),
            error__mcg=np.full(rows, 2.0, dtype=np.float64),
            error__sqg_a=np.full(rows, 1.0, dtype=np.float64),
        )
        source["layers"][str(layer)] = {"row_output": str(path)}

    result = select_layerwise(
        source, baseline_label="mcg", require_all_nonbaseline=True
    )

    assert result["search_space"] == 1
    assert result["winner"]["layer_to_label"] == {
        "74": "sqg_a",
        "75": "sqg_a",
    }
    assert result["require_all_nonbaseline"] is True
