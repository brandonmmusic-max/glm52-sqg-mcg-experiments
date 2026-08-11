from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.torch import save_file
import torch

from scripts.analyze_tail_trace_pair import (
    _correlation,
    _load_trace_root,
    _routing_metrics,
    _row_relative_delta,
)


ROOT = Path(__file__).resolve().parents[1]


def test_routing_metrics_align_weights_by_expert_id_not_slot() -> None:
    baseline_ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    candidate_ids = np.array([[3, 1, 2], [4, 5, 7]], dtype=np.int16)
    baseline_weights = np.array(
        [[0.5, 0.3, 0.2], [0.6, 0.3, 0.1]], dtype=np.float32
    )
    candidate_weights = np.array(
        [[0.2, 0.5, 0.3], [0.5, 0.3, 0.2]], dtype=np.float32
    )

    metrics = _routing_metrics(
        baseline_ids, candidate_ids, baseline_weights, candidate_weights
    )

    assert metrics["overlap"].tolist() == [3, 2]
    assert metrics["exact_order"].tolist() == [False, False]
    assert metrics["exact_set"].tolist() == [True, False]
    assert np.allclose(metrics["weight_l1"], [0.0, 0.4], atol=1e-7)


def test_row_relative_delta_uses_baseline_vector_energy() -> None:
    baseline = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    candidate = np.array([[4.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    assert np.allclose(_row_relative_delta(candidate, baseline), [1.0 / 25.0, 1.0])


def test_correlation_reports_tail_monotonicity() -> None:
    result = _correlation(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([10.0, 20.0, 30.0, 40.0]),
    )
    assert np.isclose(result["pearson"], 1.0)
    assert np.isclose(result["spearman"], 1.0)


def test_trace_loader_selects_complete_invocation_and_proves_dcp_replicas(
    tmp_path,
) -> None:
    def payload(positions: torch.Tensor) -> dict[str, torch.Tensor]:
        rows = positions.numel()
        return {
            "positions": positions.to(torch.int32),
            "hidden": torch.arange(rows * 2, dtype=torch.float32)
            .reshape(rows, 2)
            .to(torch.bfloat16),
            "router_logits": torch.zeros((rows, 3), dtype=torch.float32),
            "topk_weights": torch.ones((rows, 2), dtype=torch.float32) * 0.5,
            "topk_ids": torch.tensor([[0, 1]], dtype=torch.int32).repeat(rows, 1),
            "moe_output": torch.ones((rows, 2), dtype=torch.bfloat16),
            "residual_before_moe": torch.ones((rows, 2), dtype=torch.bfloat16),
            "residual_stream_after_moe": torch.ones(
                (rows, 2), dtype=torch.bfloat16
            ),
        }

    for rank in range(4):
        for invocation, positions in (
            (0, torch.zeros(4, dtype=torch.int32)),
            (3, torch.arange(4, dtype=torch.int32)),
        ):
            save_file(
                payload(positions),
                tmp_path
                / f"layer-006-rank-{rank:03d}-call-{invocation:03d}.safetensors",
                metadata={
                    "schema": "glm52-sqg-tail-trace-v1",
                    "layer": "6",
                    "rank": str(rank),
                    "invocation": str(invocation),
                },
            )

    layers, evidence = _load_trace_root(tmp_path)

    assert list(layers) == [6]
    assert layers[6]["positions"].tolist() == [0, 1, 2, 3]
    assert layers[6]["hidden"].dtype == np.float32
    selected = [item for item in evidence if item["analysis_selected"]]
    assert len(selected) == 4
    assert {item["invocation"] for item in selected} == {3}
    assert all(item["dcp_rank_replica_exact"] for item in selected)


def test_late_trace_runner_keeps_r33_diagnostic_distinct_from_control() -> None:
    source = (ROOT / "evaluation/run_r33_trace_once.sh").read_text()
    assert "One diagnostic trace" in source
    assert "SQG_TAIL_TRACE_LAYERS:-74,75,76,77" in source
    assert "r33_overlay_historical.args" in source
    assert "VLLM_EXL3_R7_EXPECT_RESERVED_LAYERS=none" in source
    assert "run_native_mcg_control.sh" not in source


def test_late_trace_pair_uses_exact_four_layer_candidate_contract() -> None:
    source = (ROOT / "scripts/run_contiguous_late_trace_pair.sh").read_text()
    assert "SQG_EVAL_LAYERS=74,75,76,77" in source
    assert "SQG_TAIL_TRACE_LAYERS=74,75,76,77" in source
    assert "RUNS=1" in source
    assert "analyze_tail_trace_pair.py" in source
