from __future__ import annotations

import torch

from scripts.compare_e4m3_endpoint_distortion import (
    error_decomposition,
    round_finite_e4m3,
)


def test_round_finite_e4m3_returns_exact_widened_bytes() -> None:
    values = torch.tensor([-3.95, -0.3, 0.0, 0.3, 3.95], dtype=torch.float32)
    widened, raw = round_finite_e4m3(values)
    assert widened.dtype == torch.float16
    assert raw.dtype == torch.uint8
    assert torch.equal(raw, widened.to(torch.float8_e4m3fn).view(torch.uint8))
    assert not torch.equal(values.to(torch.float16), widened)


def test_error_decomposition_closes_with_nonzero_cross_term() -> None:
    reference = torch.tensor([1.0, -2.0, 3.0])
    baseline = torch.tensor([1.2, -1.5, 2.0])
    endpoint = torch.tensor([1.0, -1.75, 2.25])
    result = error_decomposition(reference, baseline, endpoint)
    assert abs(result["closure_residual"]) < 1.0e-12
    assert result["conversion_sse"] > 0.0
    assert result["cross_term"] != 0.0
    assert abs(
        result["endpoint_minus_baseline_sse"]
        - result["conversion_sse"]
        - result["cross_term"]
    ) < 1.0e-12


def test_exact_endpoint_has_zero_increment() -> None:
    reference = torch.tensor([0.0, 1.0, 2.0])
    candidate = torch.tensor([0.1, 0.9, 2.2])
    result = error_decomposition(reference, candidate, candidate.clone())
    assert result["conversion_sse"] == 0.0
    assert result["cross_term"] == 0.0
    assert result["endpoint_minus_baseline_sse"] == 0.0
