from __future__ import annotations

import hashlib

import pytest
import torch

from scripts.score_glm52_w4a8_activation_quality import (
    ARMS,
    EXECUTION_ORDER,
    MagnitudeStats,
    NativeProjection,
    apply_output_transform,
    _block_hadamard_right,
    cross_harness_delta_percent,
    evaluate_sqg_arms,
    gate_up_transform_points,
    prepare_gate_up_operand,
    quant_dequant_mxfp8,
    sealed_mcg_layer_scalar,
    validate_result_contract,
    validate_tensor_contract,
    verify_sha256_sidecar,
)
from src.glm52_fresh_sqg.reference import normalized_hadamard


def test_mxfp8_qdq_uses_one_ue8m0_scale_per_k32_block() -> None:
    values = torch.linspace(-500.0, 500.0, 64, dtype=torch.float32).reshape(1, 64)
    output, observation = quant_dequant_mxfp8(values)

    assert observation.scale_bytes.shape == (1, 2)
    assert observation.scale_bytes.tolist() == [[128, 128]]
    manual = (
        (values.reshape(1, 2, 32) / 2.0)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .float()
        .mul(2.0)
        .reshape_as(values)
    )
    assert torch.equal(output, manual)
    assert not bool(observation.preclamp_overflow.any())


def test_mxfp8_qdq_zero_block_has_zero_scale_and_roundtrips() -> None:
    values = torch.zeros((3, 32), dtype=torch.float32)
    output, observation = quant_dequant_mxfp8(values)

    assert torch.equal(output, values)
    assert torch.equal(observation.scale_bytes, torch.zeros((3, 1), dtype=torch.uint8))
    assert not bool(observation.preclamp_overflow.any())


def test_gate_up_order_is_suh_then_hadamard_then_qdq() -> None:
    hidden = torch.linspace(-4.0, 5.0, 32).reshape(1, 32)
    suh = torch.linspace(0.25, 2.0, 32).to(torch.float16)
    hadamard = normalized_hadamard(device="cpu", dtype=torch.float32, size=32)

    actual, observation, source = prepare_gate_up_operand(
        hidden, suh, hadamard, quantize_a8=True
    )
    scaled = (hidden.half().float() * suh.float()).half().float()
    expected_source = _block_hadamard_right(scaled, hadamard)
    expected, _ = quant_dequant_mxfp8(expected_source)
    wrong_first, _ = quant_dequant_mxfp8(hidden.half().float())
    wrong_order = _block_hadamard_right(
        (wrong_first * suh.float()).half().float(), hadamard
    ).half().float()

    assert observation is not None
    assert torch.equal(source, expected_source)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, wrong_order)


def test_activation_side_hadamard_and_scales_match_decoded_weight_algebra() -> None:
    generator = torch.Generator().manual_seed(19)
    rows = 3
    width = 32
    hidden = torch.randn((rows, width), generator=generator)
    labels = (
        torch.randn((width, width), generator=generator)
        .to(torch.float8_e4m3fn)
        .float()
    )
    suh = torch.linspace(0.3, 1.4, width)
    svh = torch.linspace(1.7, 0.4, width)
    hadamard = normalized_hadamard(device="cpu", dtype=torch.float32, size=width)

    direct_input = _block_hadamard_right(hidden * suh, hadamard)
    direct = apply_output_transform(direct_input @ labels, svh, hadamard)
    decoded = hadamard @ labels
    decoded = decoded * suh[:, None]
    decoded = decoded @ hadamard
    decoded = decoded * svh[None, :]

    assert torch.allclose(direct, hidden @ decoded, rtol=2.0e-5, atol=2.0e-5)


def _native_projection(
    weight: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
) -> NativeProjection:
    native = weight.to(torch.float8_e4m3fn).float()
    return NativeProjection(weight=native, suh=suh.half(), svh=svh.half(), bits=3)


def test_four_arms_isolate_h_and_act_activation_sites() -> None:
    generator = torch.Generator().manual_seed(73)
    hidden = torch.randn((3, 32), generator=generator) * 3.0
    shared_suh = torch.linspace(0.4, 1.7, 32)
    down_suh = torch.linspace(0.2, 2.2, 32)
    projections = {
        "gate_proj": _native_projection(
            torch.randn((32, 32), generator=generator) * 0.7,
            shared_suh,
            torch.linspace(0.3, 1.9, 32),
        ),
        "up_proj": _native_projection(
            torch.randn((32, 32), generator=generator) * 0.9,
            shared_suh,
            torch.linspace(1.8, 0.25, 32),
        ),
        "down_proj": _native_projection(
            torch.randn((32, 32), generator=generator) * 0.6,
            down_suh,
            torch.linspace(0.5, 1.5, 32),
        ),
    }
    hadamard = normalized_hadamard(device="cpu", dtype=torch.float32, size=32)

    stages, qdq, magnitudes = evaluate_sqg_arms(hidden, projections, hadamard)

    assert set(stages) == set(ARMS)
    assert set(qdq) == {"h", "act_from_a16", "act_from_h_a8"}
    assert set(magnitudes) == {"h", "act_from_a16", "act_from_h_a8"}
    assert all(
        set(points) == {"raw", "after_suh_pre_h", "post_h_qdq_input"}
        for points in magnitudes.values()
    )
    assert torch.equal(stages["sqg_a16"]["gate"], stages["sqg_act_a8"]["gate"])
    assert torch.equal(stages["sqg_h_a8"]["gate"], stages["sqg_w4a8"]["gate"])
    assert torch.equal(stages["sqg_a16"]["act"], stages["sqg_act_a8"]["act"])
    assert torch.equal(stages["sqg_h_a8"]["act"], stages["sqg_w4a8"]["act"])
    assert not torch.equal(stages["sqg_a16"]["gate"], stages["sqg_h_a8"]["gate"])
    assert not torch.equal(
        stages["sqg_a16"]["expert_output"],
        stages["sqg_act_a8"]["expert_output"],
    )


def test_magnitude_path_separates_raw_pre_h_and_post_h_statistics() -> None:
    hidden = torch.arange(1, 33, dtype=torch.float32).reshape(1, 32)
    suh = torch.linspace(0.5, 1.5, 32).half()
    hadamard = normalized_hadamard(device="cpu", dtype=torch.float32, size=32)
    points = gate_up_transform_points(hidden, suh, hadamard)

    assert set(points) == {"raw", "after_suh_pre_h", "post_h_qdq_input"}
    assert not torch.equal(points["raw"], points["after_suh_pre_h"])
    assert not torch.equal(points["after_suh_pre_h"], points["post_h_qdq_input"])
    summaries = {}
    for name, values in points.items():
        stats = MagnitudeStats()
        stats.update(values)
        summaries[name] = stats.finish()
    assert all(summary["exact"]["elements"] == 32 for summary in summaries.values())
    assert all(
        summary["strided_abs_distribution"]["samples"] == 32
        for summary in summaries.values()
    )
    assert summaries["raw"]["exact"]["max_abs"] == 32.0


def _tensor_manifest() -> dict[str, object]:
    return {
        "bits": 3,
        "codebook": "sqg_xor_cheb_t12",
        "forbidden_input_reads": [
            "mcg.suh",
            "mcg.svh",
            "mcg.scales",
            "mcg.trellis",
            "mcg.packed_bytes",
        ],
        "source": {
            "kind": "official_bf16",
            "mcg_source": False,
            "tensor_name": "source.weight",
            "tensor_payload_sha256": "a" * 64,
            "shard_name": "source.safetensors",
        },
        "transform": {"hadamard_block": 128, "legacy_transform_input": False},
    }


def _source_binding() -> dict[str, str]:
    return {
        "source_name": "source.weight",
        "bf16_sha256": "a" * 64,
        "source_shard": "source.safetensors",
    }


def test_tensor_contract_fails_closed_on_rate_or_source_drift() -> None:
    validate_tensor_contract(
        prefix="tensor", bits=3, tensor_manifest=_tensor_manifest(), source_binding=_source_binding()
    )
    with pytest.raises(RuntimeError, match="assignment differs"):
        validate_tensor_contract(
            prefix="tensor",
            bits=4,
            tensor_manifest=_tensor_manifest(),
            source_binding=_source_binding(),
        )
    wrong_source = _source_binding()
    wrong_source["bf16_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="payload binding differs"):
        validate_tensor_contract(
            prefix="tensor",
            bits=3,
            tensor_manifest=_tensor_manifest(),
            source_binding=wrong_source,
        )


def test_sha256_sidecar_fails_closed_after_payload_mutation(tmp_path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_bytes(b"sealed\n")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    sidecar = tmp_path / "payload.json.sha256"
    sidecar.write_text(f"{digest}  payload.json\n")

    assert verify_sha256_sidecar(payload, sidecar) == digest
    payload.write_bytes(b"mutated\n")
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_sha256_sidecar(payload, sidecar)


def test_result_receipt_requires_order_arms_roles_and_method_invariants() -> None:
    method = {
        "arms": list(ARMS),
        "execution_order": list(EXECUTION_ORDER),
        "native_e4m3_weight_labels": True,
        "per_k32_ue8m0_e4m3_activation_qdq": True,
        "fp32_gemm_accumulation": True,
        "fp16_inter_gemm_boundaries": True,
        "hadamard_kept_on_activation_side": True,
        "per_tensor_rate_map_frozen": True,
        "no_mcg_payload_bytes": True,
        "signed_router_weighted_top8_sum_before_square": True,
        "activation_magnitude_exact_moments_and_strided_quantiles": True,
        "sealed_mcg_a16_scalar_comparator_bound": True,
        "mcg_a16_is_not_numeric_oracle_arm": True,
        "mcg_per_position_tail_comparison": False,
    }
    role = {
        "positions": 2,
        "routes": 16,
        "metrics": {arm: {} for arm in ARMS},
        "external_mcg_a16_scalar_comparator": {
            "comparison_kind": "cross_harness_scalar_only",
            "scope": "layer-077",
            "numeric_oracle_arm": False,
            "tail_comparison_permitted": False,
        },
        "activation_magnitude_path": {
            operand: {
                point: {}
                for point in ("raw", "after_suh_pre_h", "post_h_qdq_input")
            }
            for operand in ("h", "act_from_a16", "act_from_h_a8")
        },
    }
    result = {
        "schema": "glm52-sqg-w4a8-activation-quality-v1",
        "method": method,
        "go_no_go_limit": {
            "can_kill_w4a8_for_activation_quality": True,
            "can_greenlight_full_quant_by_itself": False,
        },
        "roles": {"selection": role, "holdout": role},
    }
    validate_result_contract(result)
    method["hadamard_kept_on_activation_side"] = False
    with pytest.raises(RuntimeError, match="required invariant"):
        validate_result_contract(result)


def test_mcg_scalar_delta_is_explicitly_scalar_math() -> None:
    assert cross_harness_delta_percent(0.11, 0.10) == pytest.approx(10.0)
    assert cross_harness_delta_percent(0.09, 0.10) == pytest.approx(-10.0)
    with pytest.raises(ValueError, match="NMSE scalars"):
        cross_harness_delta_percent(0.1, 0.0)


def test_mcg_comparator_uses_layer_scalar_not_four_layer_aggregate() -> None:
    receipt = {
        "aggregate": {"metrics": {"mcg": {"signed_top8_nmse": 9.0}}},
        "layers": {"77": {"metrics": {"mcg": {"signed_top8_nmse": 0.125}}}},
    }
    assert sealed_mcg_layer_scalar(receipt, 77) == 0.125
    with pytest.raises(RuntimeError, match="layer-matched"):
        sealed_mcg_layer_scalar(receipt, 76)
