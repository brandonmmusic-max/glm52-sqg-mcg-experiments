from __future__ import annotations

import numpy as np

from scripts.analyze_h13_alpha_by_effective_support import analyze


def test_effective_support_analysis_detects_support_adaptive_alpha():
    layers = (74, 75)
    labels = ("sqg_a000", "sqg_a050", "sqg_a100")
    source = {"aggregate": {"role": "selection"}, "layers": {}}
    support = {}
    for layer in layers:
        mcg = np.full(256, 100.0)
        alpha0 = np.full(256, 95.0)
        alpha50 = np.empty(256)
        alpha100 = np.empty(256)
        for expert in range(256):
            n_eff = float(expert + 1)
            support[(layer, expert)] = n_eff
            alpha50[expert] = 100.0 - 0.03 * n_eff
            alpha100[expert] = 110.0 - 0.10 * n_eff
        source["layers"][str(layer)] = {
            "per_expert_individual_route_sse": {
                "mcg": mcg.tolist(),
                "sqg_a000": alpha0.tolist(),
                "sqg_a050": alpha50.tolist(),
                "sqg_a100": alpha100.tolist(),
            }
        }

    result = analyze(
        source,
        support,
        baseline_label="mcg",
        alpha_by_label={"sqg_a000": 0.0, "sqg_a050": 0.5, "sqg_a100": 1.0},
    )

    assert result["experts"] == 512
    assert result["best_alpha_spearman_with_effective_support"] > 0.7
    assert result["gain_correlations"]["sqg_a100"][
        "spearman_effective_support_vs_gain_over_alpha0"
    ] > 0.99
    assert result["effective_support_quartiles"][0]["winning_local_alpha"] == 0.0
    assert result["effective_support_quartiles"][-1]["winning_local_alpha"] == 1.0
