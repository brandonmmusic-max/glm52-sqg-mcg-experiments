from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from benchmarks.benchmark_glm_sqg_w4a8_core import (
    _buffers_a16,
    _buffers_a8,
    _build_route_plan,
    _end_to_end_speedup,
    _load_rate_map,
)


def _rates() -> dict[int, dict[str, int]]:
    return {
        expert: {
            "gate_proj": 3 if expert % 2 == 0 else 4,
            "up_proj": 4 if expert % 2 == 0 else 3,
            "down_proj": 3 if expert < 128 else 4,
        }
        for expert in range(256)
    }


def test_route_plan_uses_exact_prefix_histogram_and_capture_rows() -> None:
    topk = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 2, 8, 9, 10, 11, 12, 13],
            [0, 1, 14, 15, 16, 17, 18, 19],
        ],
        dtype=np.uint8,
    )
    source_rows = np.array([10, 20, 30], dtype=np.int64)

    cases, histograms = _build_route_plan(
        topk,
        source_rows,
        _rates(),
        global_ms=(1, 3),
        quantiles=(0.5, 1.0),
    )

    assert histograms["1"]["all_experts"]["total_routes"] == 8
    assert histograms["3"]["all_experts"]["total_routes"] == 24
    assert histograms["3"]["all_experts"]["counts"][0] == 3
    max_gate_k3 = next(
        case
        for case in cases
        if case.global_m == 3
        and case.projection == "gate_proj"
        and case.bits == 3
        and case.route_quantile == 1.0
    )
    assert max_gate_k3.expert == 0
    assert max_gate_k3.expert_rows == 3
    assert max_gate_k3.row_indices == (10, 20, 30)


def test_route_plan_rejects_duplicate_expert_within_top8() -> None:
    topk = np.array([[0, 0, 1, 2, 3, 4, 5, 6]], dtype=np.uint8)

    with pytest.raises(ValueError, match="duplicate expert"):
        _build_route_plan(
            topk,
            np.array([0]),
            _rates(),
            global_ms=(1,),
            quantiles=(1.0,),
        )


def test_load_rate_map_requires_exact_topology_neutral_census(tmp_path) -> None:
    bit_map = {}
    for expert, projections in _rates().items():
        for projection, bits in projections.items():
            bit_map[f"model.layers.77.mlp.experts.{expert}.{projection}"] = bits
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"bit_map": bit_map}))

    loaded = _load_rate_map(path, layer=77)

    assert loaded[0]["gate_proj"] == 3
    assert loaded[1]["gate_proj"] == 4
    assert sum(
        loaded[expert][projection] == 3
        for expert in range(256)
        for projection in ("gate_proj", "up_proj", "down_proj")
    ) == 384


def test_amdahl_projection_labels_moe_and_end_to_end_speedup() -> None:
    assert _end_to_end_speedup(2.0, 0.31) == pytest.approx(1.0 / 0.845)
    assert _end_to_end_speedup(3.91, 0.31) == pytest.approx(1.2997, rel=2e-3)


def test_a16_capture_buffers_cover_m48_and_m64_rounding() -> None:
    source = torch.empty((49, 128), dtype=torch.float16)

    buffers = _buffers_a16(source, 256)

    assert buffers["gemm_output"].dtype == torch.float16
    assert buffers["rotated_f16"].dtype == torch.float16
    assert buffers["c_tmp"].numel() >= 96 * 256


def test_a8_rotation_buffer_is_fp16_for_bf16_capture_rows() -> None:
    source = torch.empty((3, 128), dtype=torch.float16)

    buffers = _buffers_a8(source, 256)

    assert buffers["output"].dtype == torch.float16
    assert buffers["rotated_f16"].dtype == torch.float16
    assert buffers["gemm_output_f16"].dtype == torch.float16
