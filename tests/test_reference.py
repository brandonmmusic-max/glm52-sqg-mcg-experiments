from __future__ import annotations

import pytest
import torch

from glm52_fresh_sqg.reference import (
    pack_trellis_states,
    reconstruct_trellis_states,
    unpack_trellis_states,
)


@pytest.mark.parametrize(
    ("bits", "expected_prefix"),
    [
        (3, [30469, 1337, 1337, 14711, 14711, 30469]),
        (4, [17767, 291, -12817, -30293, 17767, 291]),
    ],
)
def test_native_word_order_and_cyclic_state_closure(
    bits: int,
    expected_prefix: list[int],
) -> None:
    edges = (torch.arange(256, dtype=torch.int16) % (1 << bits)).reshape(1, 1, 256)
    states = reconstruct_trellis_states(edges, bits)
    packed = pack_trellis_states(states, bits)

    assert packed.shape == (1, 1, 16 * bits)
    assert packed[0, 0, : len(expected_prefix)].tolist() == expected_prefix
    assert torch.equal(unpack_trellis_states(packed, bits), states)


def test_uniform_reference_rejects_non_treatment_rates() -> None:
    states = torch.zeros((1, 1, 256), dtype=torch.int16)
    with pytest.raises(ValueError, match="K3 or K4"):
        pack_trellis_states(states, 2)
