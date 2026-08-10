from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from glm52_fresh_sqg.permutation import (
    FreshExpertPermutation,
    TEST_PERMUTATION_POLICY,
    apply_glm_expert_permutation,
    derive_glm_h2_reverse_permutation,
    draw_fresh_expert_permutation,
    glm_expert_forward,
)


def test_fresh_glm_permutation_maps_all_three_coupled_axes_and_closes() -> None:
    generator = torch.Generator().manual_seed(71)
    hidden = 8
    intermediate = 16
    gate = torch.randn((intermediate, hidden), generator=generator)
    up = torch.randn((intermediate, hidden), generator=generator)
    down = torch.randn((hidden, intermediate), generator=generator)
    inputs = torch.randn((5, hidden), generator=generator)
    originals = (gate.clone(), up.clone(), down.clone())
    permutation = draw_fresh_expert_permutation(
        intermediate,
        seed=9921,
        scope="layer-006/expert-000",
    )

    gate_p, up_p, down_p = apply_glm_expert_permutation(
        gate,
        up,
        down,
        permutation,
    )

    torch.testing.assert_close(
        glm_expert_forward(inputs, gate_p, up_p, down_p),
        glm_expert_forward(inputs, gate, up, down),
        rtol=2e-5,
        atol=2e-5,
    )
    order = permutation.new_to_old
    assert torch.equal(gate_p, gate.index_select(0, order))
    assert torch.equal(up_p, up.index_select(0, order))
    assert torch.equal(down_p, down.index_select(1, order))
    for actual, expected in zip((gate, up, down), originals, strict=True):
        assert torch.equal(actual, expected)
    assert permutation.manifest()["legacy_permutation_input"] is False


def test_fresh_permutation_is_scope_deterministic() -> None:
    first = draw_fresh_expert_permutation(128, seed=17, scope="l6/e0")
    repeat = draw_fresh_expert_permutation(128, seed=17, scope="l6/e0")
    other = draw_fresh_expert_permutation(128, seed=17, scope="l6/e1")
    assert torch.equal(first.new_to_old, repeat.new_to_old)
    assert not torch.equal(first.new_to_old, other.new_to_old)


def test_production_h2_reverse_is_calibration_bound_and_2048_wide() -> None:
    generator = torch.Generator().manual_seed(808)
    middle = torch.randn((9, 2048), generator=generator)
    gates = torch.rand((9,), generator=generator)
    permutation = derive_glm_h2_reverse_permutation(
        middle,
        gates,
        scope="layer-028/expert-017",
        evidence_id="fresh-middle-fit-v1",
        split_id="fit",
    )

    manifest = permutation.manifest()
    assert permutation.production_qualified is True
    assert permutation.intermediate_features == 2048
    assert manifest["policy"] == "fresh_glm_h2_reverse_16x128_v1"
    assert manifest["calibration_evidence_sha256"] is not None
    assert manifest["calibration_split_id"] == "fit"
    assert manifest["legacy_permutation_input"] is False


def test_production_h2_reverse_rejects_selection_rows() -> None:
    generator = torch.Generator().manual_seed(809)
    with pytest.raises(ValueError, match="only the fit split"):
        derive_glm_h2_reverse_permutation(
            torch.randn((9, 2048), generator=generator),
            torch.rand((9,), generator=generator),
            scope="layer-028/expert-017",
            evidence_id="selection-middle",
            split_id="selection",
        )


def test_non_self_inverse_new_to_old_aligns_candidate_h2_and_down_basis() -> None:
    generator = torch.Generator().manual_seed(991)
    hidden = 6
    intermediate = 4
    order = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    inverse = torch.argsort(order)
    assert not torch.equal(order, inverse)
    permutation = FreshExpertPermutation(
        seed=19,
        scope="non-self-inverse-alignment",
        new_to_old=order,
        policy=TEST_PERMUTATION_POLICY,
    )
    x = torch.randn((9, hidden), generator=generator)
    gate = torch.randn((intermediate, hidden), generator=generator)
    up = torch.randn((intermediate, hidden), generator=generator)
    down = torch.randn((hidden, intermediate), generator=generator)
    route_gates = torch.rand((9,), generator=generator)
    p_gate, p_up, p_down = apply_glm_expert_permutation(
        gate, up, down, permutation
    )

    old_middle = F.silu(F.linear(x, gate)) * F.linear(x, up)
    # Exact decoded-candidate EXL orientations after fresh physical mapping.
    gate_exl = p_gate.T.contiguous()
    up_exl = p_up.T.contiguous()
    down_exl = p_down.T.contiguous()
    candidate_middle = F.silu(x @ gate_exl) * (x @ up_exl)
    assert torch.allclose(candidate_middle, old_middle[:, order])
    assert torch.equal(down_exl, down.T[order])

    weights = route_gates.square()
    old_h2 = (old_middle.T @ (old_middle * weights[:, None])) / weights.sum()
    candidate_h2 = (
        candidate_middle.T @ (candidate_middle * weights[:, None])
    ) / weights.sum()
    expected_h2 = old_h2.index_select(0, order).index_select(1, order)
    assert torch.allclose(candidate_h2, expected_h2)

    reference = glm_expert_forward(x, gate, up, down)
    candidate = candidate_middle @ down_exl
    assert torch.allclose(candidate, reference, rtol=1e-5, atol=1e-5)

    wrong_inverse_middle = old_middle[:, inverse]
    wrong_double_middle = candidate_middle[:, order]
    assert not torch.allclose(wrong_inverse_middle, candidate_middle)
    assert not torch.allclose(wrong_double_middle, candidate_middle)
    assert not torch.allclose(wrong_inverse_middle @ down_exl, reference)
