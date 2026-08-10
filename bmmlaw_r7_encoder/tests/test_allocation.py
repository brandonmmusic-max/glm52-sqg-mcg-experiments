from __future__ import annotations

from r7_encoder.allocation import (
    allocation_to_json,
    audit_allocation,
    build_curves,
    solve_exact_allocation,
)
from r7_encoder.constants import ALLOWED_BITS, NUM_EXPERTS, PROJECTIONS, TensorId
from r7_encoder.determinism import canonical_json_bytes
from r7_encoder.types import CandidateLoss


def _candidates(layer: int = 3):
    records = []
    for expert in range(NUM_EXPERTS):
        mass = format(float(expert + 1), ".17g")
        for projection_index, projection in enumerate(PROJECTIONS):
            tensor_id = TensorId(layer, expert, projection)
            # A few tensors have a strong second marginal; the rest still have
            # a valid but smaller benefit. This exercises 5-bit choices.
            base = 10.0 + projection_index
            improvement4 = 0.25 + expert / 1000
            improvement5 = improvement4 + (2.0 if expert >= 250 else 0.1)
            losses = {3: base, 4: base - improvement4, 5: base - improvement5}
            for bits in ALLOWED_BITS:
                records.append(
                    CandidateLoss(
                        tensor_id=tensor_id,
                        bits=bits,
                        loss=format(losses[bits], ".17g"),
                        mass=mass,
                        fit_rows=128,
                        holdout_rows=32,
                        roundtrip_sha256=f"rt-{tensor_id.key}-{bits}",
                        gate_up_roundtrip_sha256=(
                            f"gu-{expert}" if projection == "down_proj" else None
                        ),
                    )
                )
    return records


def test_exact_layer_allocation_and_floor():
    masses = [float(expert + 1) for expert in range(NUM_EXPERTS)]
    curves = build_curves(3, _candidates(), masses)
    allocation = solve_exact_allocation(
        3,
        curves,
        fixed_point_iteration=1,
        probe_sha256="probe",
    )
    audit_allocation(allocation.bits)
    assert len(allocation.bits) == 768
    assert sum(allocation.bits.values()) == 2688
    assert min(allocation.bits.values()) == 3
    assert max(allocation.bits.values()) == 5
    assert allocation.upgrade_units == 384


def test_allocation_rejects_fractional_bits():
    values = {
        TensorId(3, expert, projection).key: 3
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    }
    first = next(iter(values))
    values[first] = 3.5  # type: ignore[assignment]
    try:
        audit_allocation(values)
    except ValueError as exc:
        assert "integers" in str(exc)
    else:
        raise AssertionError("fractional bits were accepted")


def test_fully_tied_allocation_is_byte_deterministic():
    candidates = []
    for expert in range(NUM_EXPERTS):
        for projection in PROJECTIONS:
            tensor_id = TensorId(3, expert, projection)
            for bits in ALLOWED_BITS:
                candidates.append(
                    CandidateLoss(
                        tensor_id=tensor_id,
                        bits=bits,
                        loss="1",
                        mass="1",
                        fit_rows=1,
                        holdout_rows=1,
                        roundtrip_sha256=f"rt-{tensor_id.key}-{bits}",
                        gate_up_roundtrip_sha256=(
                            "gu" if projection == "down_proj" else None
                        ),
                    )
                )
    curves = build_curves(3, candidates, ["1"] * NUM_EXPERTS)
    first = solve_exact_allocation(
        3, curves, fixed_point_iteration=0, probe_sha256="probe"
    )
    second = solve_exact_allocation(
        3, curves, fixed_point_iteration=0, probe_sha256="probe"
    )
    assert canonical_json_bytes(allocation_to_json(first)) == canonical_json_bytes(
        allocation_to_json(second)
    )
