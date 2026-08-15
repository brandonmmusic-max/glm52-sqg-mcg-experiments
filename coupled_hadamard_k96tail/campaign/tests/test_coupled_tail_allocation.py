from __future__ import annotations

import argparse
import json

import numpy as np

from scripts.score_coupled_tail_triplet_candidates import (
    ALLOCATION_SCHEMA,
    EXPERTS,
    EXPERT_SCHEMA,
    RATE_TRIPLETS,
    _atomic_npz,
    _finalize,
    _paths,
    _sha256_file,
)


def test_finalize_closes_exact_budget_with_fixed_position_tail(tmp_path) -> None:
    layer = 5
    rows = np.asarray([0, 1], dtype=np.int64)
    documents = np.asarray([100, 200], dtype=np.int64)
    reference = np.ones(2, dtype=np.float64)
    profile_selection = {
        "selection_id": "a" * 64,
        "selected_cell_id": "draw-00__identity",
    }
    final_profile_binding = {
        "schema": "glm52-updated-qsrt-coupled-final-profile-binding-v1",
        "complete": True,
        "profile_selection_id": profile_selection["selection_id"],
        "selected_beta": 0.0625,
        "no_b300_owner_speed_rescue": True,
        "binding_id": "b" * 64,
    }
    for expert in range(EXPERTS):
        manifest_path, score_path = _paths(tmp_path, layer, expert)
        benefit = 1.0 if expert < 16 else 0.1
        scores = np.empty((len(RATE_TRIPLETS), 2), dtype=np.float64)
        candidates = []
        preliminary_h2_evidence = {}
        for gate_bits in (3, 4):
            for up_bits in (3, 4):
                pair = f"k{gate_bits}_k{up_bits}"
                preliminary_h2_evidence[pair] = {
                    "schema": "glm52-coupled-candidate-h2-v2",
                    "evidence_id": f"{expert * 4 + (gate_bits - 3) * 2 + up_bits - 3:064x}",
                    "layer": layer,
                    "expert": expert,
                    "role": "fit/calibration",
                    "fit_only": True,
                    "row_mask_applied": True,
                    "routed_selected_row_count": 3,
                    "routed_parent_row_count": 5,
                    "selection_used": False,
                    "holdout_used": False,
                    "routed": {"role": "fit", "expert": expert, "rows": 5},
                }
        for index, rates in enumerate(RATE_TRIPLETS):
            k4 = sum(rate == 4 for rate in rates)
            pair = f"k{rates[0]}_k{rates[1]}"
            scores[index] = (10.0 - benefit * k4, 1.0 - 0.01 * benefit * k4)
            candidates.append(
                {
                    "candidate_id": f"expert-{expert}-candidate-{index}",
                    "rates": dict(zip(("gate_proj", "up_proj", "down_proj"), rates)),
                    "preliminary_h2_evidence_id": preliminary_h2_evidence[pair][
                        "evidence_id"
                    ],
                }
            )
        _atomic_npz(
            score_path,
            row_indices=rows,
            document_epochs=documents,
            row_sse=scores,
            row_reference_energy=reference,
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": EXPERT_SCHEMA,
                    "complete": True,
                    "layer": layer,
                    "expert": expert,
                    "selection_used": False,
                    "holdout_used": False,
                    "beta": 0.0625,
                    "profile_selection": profile_selection,
                    "final_profile_binding": final_profile_binding,
                    "calibration_row_count": 3,
                    "row_count": rows.size,
                    "row_sse_sha256": _sha256_file(score_path),
                    "score_id": f"score-{expert}",
                    "preliminary_h2_evidence": preliminary_h2_evidence,
                    "candidates": candidates,
                }
            )
        )

    allocation_path = tmp_path / "allocation.json"
    _finalize(
        argparse.Namespace(
            output_root=tmp_path,
            allocation_output=allocation_path,
            layer=layer,
            target_k4=48,
            tail_fraction=0.02,
            diagnostic_tail_document_count=40,
            tail_weight=1.0,
            body_regression_limit=0.01,
        )
    )
    allocation = json.loads(allocation_path.read_text())
    assert allocation["schema"] == ALLOCATION_SCHEMA
    assert allocation["production_eligible"] is True
    assert allocation["histogram"] == {"3": 720, "4": 48}
    assert allocation["bit_units"] == 2352
    assert allocation["bpw"] == 3.0625
    assert allocation["tail_policy"]["fixed_tail_positions"] == 1
    assert all(
        set(allocation["expert_assignments"][str(expert)]["rates"].values()) == {4}
        for expert in range(16)
    )
