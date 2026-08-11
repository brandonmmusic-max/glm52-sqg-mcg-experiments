from __future__ import annotations

import torch

from scripts.compare_capture_representativeness import matrix_comparison


def test_matrix_comparison_identical_closes() -> None:
    matrix = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    result = matrix_comparison(matrix, matrix.clone(), probes=8, seed=7)
    assert result["relative"]["frobenius_difference_over_full"] == 0.0
    assert result["relative"]["frobenius_cosine"] == 1.0
    assert result["relative"]["trace_change_percent"] == 0.0
    probes = result["random_probe_rayleigh_ratio_reduced_over_full"]
    assert probes["minimum"] == 1.0
    assert probes["maximum"] == 1.0


def test_matrix_comparison_detects_uniform_scale() -> None:
    matrix = torch.tensor([[2.0, 0.25], [0.25, 1.0]])
    result = matrix_comparison(matrix, matrix * 1.1, probes=8, seed=9)
    assert abs(result["relative"]["trace_change_percent"] - 10.0) < 1.0e-5
    assert result["relative"]["frobenius_cosine"] > 0.999999
    probes = result["random_probe_rayleigh_ratio_reduced_over_full"]
    assert abs(probes["median"] - 1.1) < 1.0e-6
