from __future__ import annotations

import torch

from scripts.run_corouting_retained_profiles import optimize_corouted_profiles


def test_corouting_uses_cancellation_when_unary_cost_is_equal() -> None:
    positions = {
        0: torch.tensor([0]),
        1: torch.tensor([0]),
    }
    errors = {
        0: torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        1: torch.tensor([[[1.0, 0.0], [0.0, -1.0]]]),
    }
    result = optimize_corouted_profiles(
        positions,
        errors,
        role_rows=1,
        hidden=2,
        unary_relative_slack=0.0,
    )

    assert result["objective"] == 0.0
    assert result["selected_unary"] == 2.0
    assert result["cross_term"] == -2.0


def test_corouting_unary_bound_rejects_damaging_cancellation() -> None:
    positions = {
        0: torch.tensor([0]),
        1: torch.tensor([0]),
    }
    errors = {
        0: torch.tensor([[[1.0], [2.0]]]),
        1: torch.tensor([[[1.0], [-2.0]]]),
    }
    strict = optimize_corouted_profiles(
        positions,
        errors,
        role_rows=1,
        hidden=1,
        unary_relative_slack=0.0,
    )
    permissive = optimize_corouted_profiles(
        positions,
        errors,
        role_rows=1,
        hidden=1,
        unary_relative_slack=3.0,
    )

    assert strict["objective"] == 4.0
    assert strict["selection"] == {0: 0, 1: 0}
    assert permissive["objective"] == 0.0
    assert permissive["selected_unary"] == 8.0
