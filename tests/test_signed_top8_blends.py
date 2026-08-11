import argparse

import numpy as np
import pytest

from scripts.score_signed_top8_blends import _parse_layers, tail_comparison


def test_tail_comparison_preserves_win_rate_and_large_positive_tail() -> None:
    baseline = np.array([1.0, 1.0, 1.0, 1.0])
    candidate = np.array([0.5, 0.5, 0.5, 10.0])
    energy = np.ones(4)

    result = tail_comparison(candidate, baseline, energy)

    assert result["improved_positions"] == 3
    assert result["improved_fraction"] == 0.75
    assert result["worsened_positions"] == 1
    assert result["mean_relative_delta"] > 0.0
    assert result["relative_delta"]["positive_max"] == 9.0


def test_tail_comparison_normalizes_by_reference_energy() -> None:
    baseline = np.array([2.0, 8.0])
    candidate = np.array([1.0, 4.0])
    energy = np.array([1.0, 4.0])

    result = tail_comparison(candidate, baseline, energy)

    assert result["improved_fraction"] == 1.0
    assert result["candidate_relative_error"]["mean"] == 1.0
    assert result["baseline_relative_error"]["mean"] == 2.0


def test_parse_layers_accepts_contiguous_late_block() -> None:
    assert _parse_layers("74,75,76,77") == (74, 75, 76, 77)


@pytest.mark.parametrize(
    "raw",
    ("74,75,76", "74,75,75,77", "77,76,75,74", "2,3,4,5"),
)
def test_parse_layers_rejects_invalid_blocks(raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_layers(raw)
