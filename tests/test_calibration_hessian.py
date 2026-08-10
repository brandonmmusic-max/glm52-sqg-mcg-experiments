from __future__ import annotations

import pytest
import torch

from kquant.candidate_hessian import adaptive_identity_shrinkage
from src.calibration_hessian import (
    H2_LOCAL_ALPHA_CAP,
    apply_frozen_h2_shrinkage,
    expert_route_selection,
    glm_candidate_intermediate,
    validate_h2_support,
    weighted_dense_covariance,
)


def test_weighted_dense_covariance_matches_direct_definition() -> None:
    rows = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
    weights = torch.tensor([1.0, 2.0, 0.5])
    actual, denominator = weighted_dense_covariance(rows, weights)
    expected = rows.T @ (rows * weights[:, None]) / weights.sum()
    assert denominator == pytest.approx(3.5)
    torch.testing.assert_close(actual, expected)


def test_frozen_h2_shrinkage_is_exact_kquant_policy() -> None:
    rows = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [2.0, 1.0], [-1.0, 0.5]]
    )
    weights = torch.tensor([0.5, 0.25, 1.5, 0.75]).square()
    local, _ = weighted_dense_covariance(rows, weights)

    actual, evidence = apply_frozen_h2_shrinkage(local, weights)
    expected, expected_evidence = adaptive_identity_shrinkage(
        local, weights, max_local_alpha=H2_LOCAL_ALPHA_CAP
    )
    torch.testing.assert_close(actual, expected)
    assert evidence == expected_evidence
    assert evidence["effective_sample_size"] == pytest.approx(
        float(weights.double().sum().square() / weights.double().square().sum())
    )
    assert evidence["local_alpha"] <= 0.75
    torch.testing.assert_close(torch.trace(actual), torch.trace(local))


def test_expert_route_selection_preserves_the_aligned_gate() -> None:
    ids = torch.tensor([[4, 2, 9], [1, 4, 8], [7, 3, 0]])
    gates = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    selected, selected_gates = expert_route_selection(ids, gates, 4)
    torch.testing.assert_close(selected, torch.tensor([0, 1]))
    torch.testing.assert_close(selected_gates, torch.tensor([0.1, 0.5]))

    ids[0, 1] = 4
    with pytest.raises(ValueError, match="more than once"):
        expert_route_selection(ids, gates, 4)


def test_glm_candidate_intermediate_is_silu_gate_times_up() -> None:
    hidden = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    gate = torch.tensor([[1.0, 0.25], [-0.5, 2.0]])
    up = torch.tensor([[0.5, -1.0], [2.0, 0.0]])
    actual = glm_candidate_intermediate(hidden, gate, up)
    expected = torch.nn.functional.silu(hidden @ gate.T) * (hidden @ up.T)
    torch.testing.assert_close(actual, expected)


def test_h2_support_fails_instead_of_borrowing_a_prior() -> None:
    validate_h2_support(routed_rows=6, fit_documents=6)
    with pytest.raises(ValueError, match="fit routes"):
        validate_h2_support(routed_rows=5, fit_documents=100)
    with pytest.raises(ValueError, match="fit documents"):
        validate_h2_support(routed_rows=100, fit_documents=5)

