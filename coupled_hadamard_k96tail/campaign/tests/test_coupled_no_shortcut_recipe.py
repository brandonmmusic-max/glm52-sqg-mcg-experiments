from __future__ import annotations

import pytest

from scripts.coupled_recipe_core import (
    ALL_TRIPLETS,
    _is_sha256,
    validate_triplets,
)
from scripts.run_coupled_no_shortcut_recipe import select_layer_beta
from scripts.select_full_w4a8_beta_panel import BETAS


def _record(scores: dict[float, float]) -> dict[str, object]:
    return {
        "candidates": [
            {"beta": float(beta), "fit_allocation_sse": float(scores[beta])}
            for beta in BETAS
        ]
    }


def test_triplet_contract_accepts_unique_k3_k4_cube() -> None:
    assert validate_triplets(ALL_TRIPLETS) == ALL_TRIPLETS


@pytest.mark.parametrize(
    "triplets",
    [(), ((3, 3, 3), (3, 3, 3)), ((2, 3, 4),), ((3, 3),)],
)
def test_triplet_contract_rejects_invalid_panels(triplets) -> None:
    with pytest.raises(ValueError):
        validate_triplets(triplets)


def test_layer_beta_sums_fit_allocation_and_uses_smaller_tie_break() -> None:
    first = {float(beta): 10.0 for beta in BETAS}
    second = {float(beta): 20.0 for beta in BETAS}
    first[0.0625] = 1.0
    second[0.0625] = 2.0
    first[0.125] = 2.0
    second[0.125] = 1.0

    totals, selected = select_layer_beta([_record(first), _record(second)])

    assert totals[0.0625] == pytest.approx(3.0)
    assert totals[0.125] == pytest.approx(3.0)
    assert selected == 0.0625


def test_recipe_digest_contract_requires_lowercase_sha256() -> None:
    assert _is_sha256("a" * 64)
    assert not _is_sha256("A" * 64)
    assert not _is_sha256("a" * 63)
