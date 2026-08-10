from __future__ import annotations

import pytest
import torch

from r7_encoder.hessian import (
    FullCovarianceAccumulator,
    down_inputs_from_roundtrip,
    least_squares_output_scale,
)
from r7_encoder.permutation import (
    energy_balanced_permutation,
    functional_oracle,
    inverse_permutation,
)
from r7_encoder.rotations import (
    coordinate_search_block_scales,
    fold_block_g_scale,
    rademacher_vector,
)


def test_permutation_swiglu_function_oracle():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(5, 8, generator=generator, dtype=torch.float64)
    gate = torch.randn(16, 8, generator=generator, dtype=torch.float64)
    up = torch.randn(16, 8, generator=generator, dtype=torch.float64)
    down = torch.randn(8, 16, generator=generator, dtype=torch.float64)
    diagonal = torch.linspace(1, 10, 16).tolist()
    permutation = energy_balanced_permutation(diagonal, block=4)
    audit = functional_oracle(x, gate, up, down, permutation)
    assert audit.exact_inverse
    assert audit.max_abs_function_error < 1e-10
    inverse = inverse_permutation(permutation)
    assert [permutation[inverse[index]] for index in range(16)] == list(range(16))


def test_down_inputs_use_supplied_roundtrip():
    x = torch.tensor([[1.0, -2.0]])
    gate = torch.tensor([[0.5, 0.25], [1.0, -1.0]])
    up = torch.tensor([[1.5, -0.5], [0.25, 2.0]])
    actual = down_inputs_from_roundtrip(x, gate, up)
    expected = torch.nn.functional.silu(x @ gate) * (x @ up)
    assert torch.equal(actual, expected)


def test_full_covariance_is_not_sliced():
    accumulator = FullCovarianceAccumulator(16, device="cpu")
    x = torch.arange(64, dtype=torch.float32).reshape(4, 16) / 10
    accumulator.add(x)
    result = accumulator.finalize(0.025)
    expected = (x.double().T @ x.double()) / 4
    assert result.matrix.shape == (16, 16)
    assert torch.allclose(result.matrix.double(), expected, rtol=1e-6, atol=1e-6)
    assert result.rows == 4


def test_singular_nonzero_covariance_is_preserved_but_zero_is_rejected():
    singular = FullCovarianceAccumulator(16, device="cpu")
    singular.add(torch.ones((3, 16)))
    result = singular.finalize(0.025)
    assert torch.linalg.matrix_rank(result.matrix.double()) == 1

    degenerate = FullCovarianceAccumulator(16, device="cpu")
    degenerate.add(torch.zeros((2, 16)))
    with pytest.raises(ValueError, match="identity fallback is forbidden"):
        degenerate.finalize(0.025)


def test_least_squares_output_scale():
    reference = torch.tensor([[2.0, -6.0], [4.0, 3.0]])
    reconstructed = torch.tensor([[1.0, -2.0], [2.0, 1.0]])
    scale = least_squares_output_scale(reference, reconstructed)
    assert torch.allclose(scale, torch.tensor([2.0, 3.0], dtype=torch.float64))


def test_block_scale_folding_and_deterministic_draws():
    su = [1.0] * 256
    sv = [-1.0] * 128
    folded = fold_block_g_scale(su, sv, [2.0, 4.0], [0.5])
    assert folded.suh[:128] == (0.5,) * 128
    assert folded.suh[128:] == (0.25,) * 128
    assert folded.svh == (-2.0,) * 128
    assert rademacher_vector(64, "L3", "shared") == rademacher_vector(
        64, "L3", "shared"
    )


def test_coordinate_block_search_has_stable_ties():
    def score(ks, ns):
        return sum((value - 1.1) ** 2 for value in ks + ns)

    ks, ns, value = coordinate_search_block_scales(
        k_blocks=2,
        n_blocks=1,
        score=score,
        grid=(0.9, 1.0, 1.1),
        sweeps=1,
    )
    assert ks == (1.1, 1.1)
    assert ns == (1.1,)
    assert value < 1e-20
