from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

import src.glm52_capture_worker as worker_module
from src.calibration_capture import (
    ATTENTION_BACKEND,
    EFFECTIVE_KV_CACHE_DTYPE,
    EXPECTED_FUSED_LAYERS,
    NUM_EXPERTS,
    ROUTED_LAYERS,
    ROUTED_SCALING_FACTOR,
    TOPK,
    validate_router_audit,
)
from src.capture_runtime import (
    IMAGE_DECLARATION_ENV,
    RUNTIME_CLASS_SOURCES,
    RUNTIME_ENV,
    RUNTIME_IMAGE_ID,
)
from src.glm52_capture_worker import (
    FreshSQGCaptureWorkerExtension,
    compare_admissible_live_route,
    glm_reference_applied_route,
    glm_reference_route_scores,
    install_post_select_capture,
    restore_select_capture,
)


class DummyRouter:
    def __init__(self, events: list[str], weights: torch.Tensor, ids: torch.Tensor):
        self.events = events
        self.weights = weights
        self.ids = ids

    def select_experts(self, hidden_states, router_logits):
        self.events.append("original")
        return self.weights, self.ids


class Exl3MoEMethod:
    is_monolithic = False


class MoERunner(SimpleNamespace):
    pass


class DeepseekV2MoE(SimpleNamespace):
    pass


class GroupedTopKRouter(SimpleNamespace):
    pass


class _FusedAuditModel:
    def __init__(self, *, fused_layers=EXPECTED_FUSED_LAYERS, codebook="mcg"):
        quant_config = SimpleNamespace(
            r7_routed_experts={"codebook": codebook},
            _r7_fused_layers=48,
        )
        self.modules = []
        fused = set(fused_layers)
        for layer in ROUTED_LAYERS:
            routed = SimpleNamespace(
                quant_method=SimpleNamespace(quant_config=quant_config),
                exl3_r7_fused=layer in fused,
            )
            if layer in fused:
                routed.exl3_mixed_trellis = {}
            self.modules.append(
                (f"model.layers.{layer}.mlp.experts", SimpleNamespace(routed_experts=routed))
            )

    def named_modules(self):
        return list(self.modules)


def _fused_audit_worker(**kwargs) -> FreshSQGCaptureWorkerExtension:
    worker = FreshSQGCaptureWorkerExtension()
    worker.model_runner = SimpleNamespace(model=_FusedAuditModel(**kwargs))
    return worker


def _runtime_config(*, dcp_interleave: int = 64, cp_interleave: int = 1):
    parallel = SimpleNamespace(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=4,
        dcp_comm_backend="a2a",
        dcp_kv_cache_interleave_size=dcp_interleave,
        cp_kv_cache_interleave_size=cp_interleave,
        enable_expert_parallel=False,
        enable_eplb=False,
        use_sequence_parallel_moe=False,
        enable_dbo=False,
        disable_custom_all_reduce=True,
    )
    return SimpleNamespace(
        parallel_config=parallel,
        scheduler_config=SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=2_048,
            async_scheduling=False,
            enable_chunked_prefill=True,
        ),
        cache_config=SimpleNamespace(
            gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=268_435_456,
            enable_prefix_caching=False,
            cache_dtype=EFFECTIVE_KV_CACHE_DTYPE,
        ),
        model_config=SimpleNamespace(
            max_model_len=4_352,
            enforce_eager=True,
            quantization="exl3",
        ),
        load_config=SimpleNamespace(load_format="safetensors"),
        attention_config=SimpleNamespace(backend=ATTENTION_BACKEND),
        kernel_config=SimpleNamespace(moe_backend="b12x"),
        speculative_config=None,
    )


def _runtime_worker(config) -> FreshSQGCaptureWorkerExtension:
    worker = FreshSQGCaptureWorkerExtension()
    worker._fresh_vllm_config = lambda: config
    worker._fresh_world = lambda: 4
    return worker


def test_worker_runtime_accepts_raw_cp1_dcp64_and_proves_effective_64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_module, "RUNTIME_PYTHON", sys.executable)
    monkeypatch.setenv(IMAGE_DECLARATION_ENV, RUNTIME_IMAGE_ID)
    for key, value in RUNTIME_ENV.items():
        monkeypatch.setenv(key, value)

    audit = _runtime_worker(_runtime_config())._fresh_runtime_contract()

    assert audit["dcp_kv_cache_interleave_size_raw"] == 64
    assert audit["cp_kv_cache_interleave_size_raw"] == 1
    assert audit["effective_kv_cache_interleave_size"] == 64


@pytest.mark.parametrize(
    ("dcp_interleave", "cp_interleave"),
    ((32, 1), (64, 64), (64, 2)),
)
def test_worker_runtime_rejects_other_raw_interleave_combinations(
    monkeypatch: pytest.MonkeyPatch,
    dcp_interleave: int,
    cp_interleave: int,
) -> None:
    monkeypatch.setattr(worker_module, "RUNTIME_PYTHON", sys.executable)
    monkeypatch.setenv(IMAGE_DECLARATION_ENV, RUNTIME_IMAGE_ID)
    for key, value in RUNTIME_ENV.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match="runtime contract differs"):
        _runtime_worker(
            _runtime_config(
                dcp_interleave=dcp_interleave,
                cp_interleave=cp_interleave,
            )
        )._fresh_runtime_contract()


def test_worker_observes_exact_all_mcg_teacher_fused_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_EXL3_R7_FUSED_LAYERS", "48")
    audit = _fused_audit_worker()._fresh_fused_layer_audit()
    assert audit["fused_layers"] == list(EXPECTED_FUSED_LAYERS)
    assert audit["reserved_layers"] == []
    assert audit["selected_layer_modes"]["77"] == "nonfused"


def test_worker_rejects_fused_population_or_teacher_codebook_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_EXL3_R7_FUSED_LAYERS", "48")
    with pytest.raises(RuntimeError, match="fused-layer audit differs"):
        _fused_audit_worker(
            fused_layers=EXPECTED_FUSED_LAYERS[:-1]
        )._fresh_fused_layer_audit()
    with pytest.raises(RuntimeError, match="not the all-MCG"):
        _fused_audit_worker(codebook="sqg")._fresh_fused_layer_audit()


def test_wrapper_calls_original_first_and_returns_exact_objects() -> None:
    events: list[str] = []
    weights = torch.ones((2, TOPK), dtype=torch.float32)
    ids = torch.arange(TOPK, dtype=torch.int32).repeat(2, 1)
    hidden = object()
    logits = object()
    router = DummyRouter(events, weights, ids)

    def capture(observed_hidden, observed_logits, observed_weights, observed_ids):
        assert events == ["original"]
        assert observed_hidden is hidden
        assert observed_logits is logits
        assert observed_weights is weights
        assert observed_ids is ids
        events.append("capture")

    state = install_post_select_capture(router, capture)
    result = router.select_experts(hidden_states=hidden, router_logits=logits)
    assert result[0] is weights and result[1] is ids
    assert events == ["original", "capture"]
    restore_select_capture(state)
    assert "select_experts" not in router.__dict__


def test_reference_routing_uses_bias_only_for_selection_and_live_ids() -> None:
    torch.manual_seed(7)
    logits = torch.randn((5, NUM_EXPERTS), dtype=torch.float32)
    bias = torch.linspace(-0.2, 0.2, NUM_EXPERTS)
    weights, ids = glm_reference_applied_route(logits, bias, torch)

    assert torch.allclose(
        weights.sum(dim=1),
        torch.full((5,), ROUTED_SCALING_FACTOR),
    )
    unbiased = torch.sigmoid(logits)
    expected = unbiased.gather(1, ids)
    expected = expected / expected.sum(dim=1, keepdim=True) * 2.5
    torch.testing.assert_close(weights, expected)

    reverse = torch.arange(TOPK - 1, -1, -1).repeat(5, 1)
    permuted_ids = ids.gather(1, reverse)
    permuted_weights = weights.gather(1, reverse)
    reference_weights, _ = glm_reference_applied_route(
        logits, bias, torch, selected_ids=permuted_ids
    )
    unbiased, biased = glm_reference_route_scores(logits, bias, torch)
    audit = compare_admissible_live_route(
        permuted_weights,
        permuted_ids,
        reference_weights,
        unbiased,
        biased,
        torch,
    )
    assert audit["checked_rows"] == 5
    assert audit["selection_violation_rows"] == 0
    assert audit["weight_mismatch_rows"] == 0

    permuted_weights[0, 0] += 1e-2
    audit = compare_admissible_live_route(
        permuted_weights,
        permuted_ids,
        reference_weights,
        unbiased,
        biased,
        torch,
    )
    assert audit["weight_mismatch_rows"] == 1


def test_route_admissibility_accepts_only_measured_cutoff_boundary() -> None:
    ids = torch.arange(TOPK, dtype=torch.int32).reshape(1, TOPK)
    unbiased = torch.full((1, NUM_EXPERTS), 0.5, dtype=torch.float32)
    biased = torch.zeros_like(unbiased)
    biased[0, :TOPK] = 1.0
    measured_fused_boundary = 1.9073486328125e-6
    biased[0, TOPK] = 1.0 + measured_fused_boundary
    reference_weights = torch.full(
        (1, TOPK), ROUTED_SCALING_FACTOR / TOPK, dtype=torch.float32
    )

    audit = compare_admissible_live_route(
        reference_weights,
        ids,
        reference_weights,
        unbiased,
        biased,
        torch,
    )
    assert audit["selection_violation_rows"] == 0
    assert audit["boundary_ambiguous_rows"] == 1
    assert audit["max_selection_violation"] == measured_fused_boundary

    biased[0, TOPK] = 1.0 + 2.1457672119140625e-6
    audit = compare_admissible_live_route(
        reference_weights,
        ids,
        reference_weights,
        unbiased,
        biased,
        torch,
    )
    assert audit["selection_violation_rows"] == 1
    assert audit["max_selection_violation"] > 2e-6


def _audit_objects(
    *,
    router_scale: float,
    runner_scale: float,
    owner_scale: float | None = None,
):
    quant_method = Exl3MoEMethod()
    routed_experts = SimpleNamespace(
        quant_method=quant_method,
        routed_scaling_factor=router_scale,
        apply_router_weight_on_input=False,
    )
    runner = MoERunner(
        enable_dbo=False,
        routed_input_transform=None,
        routed_scaling_factor=runner_scale,
    )
    module = MoERunner(
        routed_experts=routed_experts,
        runner=runner,
        routed_scaling_factor=runner_scale,
        _routed_input_transform=None,
        use_ep=False,
        is_sequence_parallel=False,
    )
    owner = DeepseekV2MoE(
        routed_scaling_factor=(runner_scale if owner_scale is None else owner_scale),
        is_rocm_aiter_moe_enabled=False,
    )
    router = GroupedTopKRouter(
        top_k=TOPK,
        global_num_experts=NUM_EXPERTS,
        renormalize=True,
        scoring_func="sigmoid",
        routed_scaling_factor=router_scale,
        num_expert_group=1,
        topk_group=1,
        num_fused_shared_experts=0,
        enable_eplb=False,
    )
    return owner, module, router


@pytest.mark.parametrize("router_scale,runner_scale", [(1.0, 2.5), (2.5, 1.0)])
def test_scale_placement_accepts_only_one_effective_2_5_product(
    router_scale: float, runner_scale: float
) -> None:
    owner, module, router = _audit_objects(
        router_scale=router_scale, runner_scale=runner_scale
    )
    audit = FreshSQGCaptureWorkerExtension._fresh_router_audit(
        6,
        "model.layers.6.mlp.experts",
        owner,
        module,
        router,
        class_sources=RUNTIME_CLASS_SOURCES,
    )
    assert audit["effective_routed_scaling_factor"] == 2.5
    assert validate_router_audit(audit, layer=6)["runner_output_scale"] == (
        runner_scale
    )


def test_unknown_scale_placement_fails_closed() -> None:
    owner, module, router = _audit_objects(router_scale=1.0, runner_scale=1.0)
    with pytest.raises(RuntimeError, match="unknown scaling placement"):
        FreshSQGCaptureWorkerExtension._fresh_router_audit(
            6,
            "model.layers.6.mlp.experts",
            owner,
            module,
            router,
            class_sources=RUNTIME_CLASS_SOURCES,
        )


def test_owner_config_scale_cannot_substitute_for_actual_runner_scale() -> None:
    owner, module, router = _audit_objects(
        router_scale=1.0,
        runner_scale=1.0,
        owner_scale=2.5,
    )
    with pytest.raises(RuntimeError, match="owner/module/runner output scales disagree"):
        FreshSQGCaptureWorkerExtension._fresh_router_audit(
            6,
            "model.layers.6.mlp.experts",
            owner,
            module,
            router,
            class_sources=RUNTIME_CLASS_SOURCES,
        )


def test_live_class_source_provenance_fails_closed() -> None:
    owner, module, router = _audit_objects(router_scale=1.0, runner_scale=2.5)
    drifted = {key: dict(value) for key, value in RUNTIME_CLASS_SOURCES.items()}
    drifted["runner"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="class sources differ"):
        FreshSQGCaptureWorkerExtension._fresh_router_audit(
            6,
            "model.layers.6.mlp.experts",
            owner,
            module,
            router,
            class_sources=drifted,
        )
