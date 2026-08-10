from __future__ import annotations

import pytest

from capture_calibration import _all_rank_fused_audits
from src.calibration_capture import (
    EXPECTED_FUSED_LAYERS,
    OWNER_TOKENS,
    REFERENCE_ROUTE_ATOL,
    REFERENCE_ROUTE_DESCRIPTION,
    REFERENCE_ROUTE_RTOL,
    REFERENCE_ROUTE_SELECTION_ATOL,
    REFERENCE_ROUTE_SCHEMA,
    ROUTED_LAYERS,
    ROUTED_SCALING_FACTOR,
    TOPK,
    validate_fused_layer_audit,
    validate_raw_route_weight_evidence,
    validate_reference_route_check,
)


def _reference(rows: int) -> dict:
    return {
        "schema": REFERENCE_ROUTE_SCHEMA,
        "checked_rows": rows,
        "selection_violation_rows": 0,
        "boundary_ambiguous_rows": 1,
        "max_selection_violation": 1.9073486328125e-6,
        "weight_mismatch_rows": 0,
        "max_abs_weight_error": 1e-7,
        "selection_atol": REFERENCE_ROUTE_SELECTION_ATOL,
        "weight_rtol": REFERENCE_ROUTE_RTOL,
        "weight_atol": REFERENCE_ROUTE_ATOL,
        "coverage": "all captured rows",
        "reference": REFERENCE_ROUTE_DESCRIPTION,
    }


def _raw_evidence() -> tuple[dict, dict]:
    raw_sum = float(OWNER_TOKENS)
    raw_sq_sum = float(OWNER_TOKENS) / TOPK
    value = {
        "source": (
            "exact float32 weights returned by original live "
            "router.select_experts"
        ),
        "dtype": "float32-le",
        "shape": [OWNER_TOKENS, TOPK],
        "bytes": OWNER_TOKENS * TOPK * 4,
        "sha256": "0" * 64,
        "payload_stored": False,
        "effective_payload": "topk_weights.f32le.bin",
        "router_return_scale": 1.0,
        "runner_output_scale": ROUTED_SCALING_FACTOR,
        "effective_scale_product": ROUTED_SCALING_FACTOR,
        "sum": raw_sum,
        "sq_sum": raw_sq_sum,
        "row_sum_min": 1.0,
        "row_sum_max": 1.0,
    }
    applied = {
        "gate_sum": raw_sum * ROUTED_SCALING_FACTOR,
        "gate_sq_sum": raw_sq_sum * ROUTED_SCALING_FACTOR**2,
    }
    return value, applied


def _fused_audit() -> dict:
    fused = list(EXPECTED_FUSED_LAYERS)
    return {
        "schema": "glm52-r33-all-mcg-teacher-fused-layer-audit-v1",
        "routed_layers": list(ROUTED_LAYERS),
        "fused_layers": fused,
        "nonfused_layers": sorted(set(ROUTED_LAYERS) - set(fused)),
        "fused_count": 48,
        "configured_budget": 48,
        "observed_budget_counters": [48],
        "reserved_layers": [],
        "selected_layer_modes": {
            "6": "fused",
            "28": "fused",
            "52": "fused",
            "77": "nonfused",
        },
        "teacher_codebook": "mcg",
    }


def test_exact_all_mcg_teacher_fused_population_is_required() -> None:
    audit = _fused_audit()
    assert validate_fused_layer_audit(audit) == audit

    drifted = {**audit, "reserved_layers": [6]}
    with pytest.raises(ValueError, match="differs"):
        validate_fused_layer_audit(drifted)


def test_collective_fused_evidence_must_cover_all_four_identical_tp_ranks() -> None:
    audit = _fused_audit()
    values = [
        {"rank": rank, "fused_layer_audit": audit} for rank in (3, 1, 0, 2)
    ]
    assert list(_all_rank_fused_audits(values)) == ["0", "1", "2", "3"]

    with pytest.raises(RuntimeError, match="does not cover TP ranks"):
        _all_rank_fused_audits(values[:-1])

    drifted = {**audit, "fused_layers": audit["fused_layers"][:-1]}
    with pytest.raises(ValueError, match="differs"):
        validate_fused_layer_audit(drifted)


def test_all_row_reference_route_evidence_is_required() -> None:
    assert validate_reference_route_check(_reference(17), rows=17)[
        "checked_rows"
    ] == 17
    bad = _reference(17)
    bad["checked_rows"] = 16
    with pytest.raises(ValueError, match="checked_rows"):
        validate_reference_route_check(bad, rows=17)
    bad = _reference(17)
    bad["weight_mismatch_rows"] = 1
    with pytest.raises(ValueError, match="weight_mismatch_rows"):
        validate_reference_route_check(bad, rows=17)
    bad = _reference(17)
    bad["max_selection_violation"] = 2.1e-6
    with pytest.raises(ValueError, match="maximum selection violation"):
        validate_reference_route_check(bad, rows=17)
    bad = _reference(17)
    bad["boundary_ambiguous_rows"] = 0
    with pytest.raises(ValueError, match="selection evidence is inconsistent"):
        validate_reference_route_check(bad, rows=17)


def test_raw_return_and_effective_gate_statistics_close_under_sole_scale() -> None:
    value, applied = _raw_evidence()
    canonical = validate_raw_route_weight_evidence(value, applied=applied)
    assert canonical["router_return_scale"] == 1.0
    assert canonical["runner_output_scale"] == 2.5

    value["runner_output_scale"] = 2.0
    with pytest.raises(ValueError, match="sole 2.5 product"):
        validate_raw_route_weight_evidence(value, applied=applied)


def test_per_gate_float32_scale_rounding_has_only_bounded_aggregate_slack() -> None:
    value, applied = _raw_evidence()
    # The completed four-layer capture accumulated up to 2.92e-4 of difference
    # after 8,403,744 independent float32 products.  This models that rounding
    # without relaxing the exact scale-placement checks above.
    applied["gate_sum"] += 2.6e-4
    applied["gate_sq_sum"] -= 3.0e-4
    validate_raw_route_weight_evidence(value, applied=applied)

    applied["gate_sum"] += 5e-3
    with pytest.raises(ValueError, match="do not close under runner scale"):
        validate_raw_route_weight_evidence(value, applied=applied)
