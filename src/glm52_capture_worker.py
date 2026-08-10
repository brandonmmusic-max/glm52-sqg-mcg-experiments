"""vLLM worker extension for exact live-router GLM-5.2 calibration capture.

Only TP rank zero writes.  Each selected router instance is wrapped so its
original ``select_experts`` method executes first.  The wrapper preserves the
exact returned IDs and weights, then applies only the separately audited GLM
runner output scale so the stored gates are the effective expert multipliers.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import os
from pathlib import Path
import re
import sys
import types

import numpy as np

from .calibration_capture import (
    ATTENTION_BACKEND,
    EFFECTIVE_KV_CACHE_DTYPE,
    EXPECTED_FUSED_LAYERS,
    FILE_ABI,
    HIDDEN,
    INDEX_TOPK_PATTERN,
    KV_CACHE_INTERLEAVE_RESOLUTION,
    NUM_EXPERTS,
    OWNER_DOCUMENTS,
    OWNER_TOKENS,
    REFERENCE_ROUTE_ATOL,
    REFERENCE_ROUTE_DESCRIPTION,
    REFERENCE_ROUTE_RTOL,
    REFERENCE_ROUTE_SELECTION_ATOL,
    REFERENCE_ROUTE_SCHEMA,
    ROUTED_SCALING_FACTOR,
    ROUTED_LAYERS,
    SELECTED_LAYERS,
    TOPK,
    document_audit_bytes,
    effective_kv_cache_interleave_size,
    validate_fused_layer_audit,
)
from .calibration_plan import ROLE_TO_ID, document_role, sha256_file
from .capture_runtime import (
    IMAGE_DECLARATION_ENV,
    RUNTIME_ENV,
    RUNTIME_CLASS_SOURCES,
    RUNTIME_IMAGE_ID,
    RUNTIME_PYTHON,
)


def glm_reference_route_scores(router_logits, correction_bias, torch):
    """Return independent unbiased and selection scores for GLM routing."""

    if (
        not isinstance(router_logits, torch.Tensor)
        or router_logits.ndim != 2
        or int(router_logits.shape[1]) != NUM_EXPERTS
    ):
        raise ValueError("reference router logits must have shape [rows, 256]")
    if (
        not isinstance(correction_bias, torch.Tensor)
        or tuple(correction_bias.shape) != (NUM_EXPERTS,)
    ):
        raise ValueError("reference correction bias must have shape [256]")
    logits = router_logits.detach().to(torch.float32)
    bias = correction_bias.detach().to(device=logits.device, dtype=torch.float32)
    if not bool(torch.isfinite(logits).all().item()) or not bool(
        torch.isfinite(bias).all().item()
    ):
        raise ValueError("reference routing inputs are non-finite")
    unbiased_scores = torch.sigmoid(logits)
    return unbiased_scores, unbiased_scores + bias


def glm_reference_applied_route(
    router_logits, correction_bias, torch, *, selected_ids=None
):
    """Reconstruct effective weights on explicit IDs (or a diagnostic top-k).

    Runtime capture always supplies the exact IDs returned by the live fused
    router.  A Python ``torch.topk`` is deliberately not treated as an exact
    oracle for the fused CUDA kernel: the kernel evaluates sigmoid as
    ``0.5*tanhf(0.5*x)+0.5`` and resolves exact ties by lower expert ID.
    """

    unbiased_scores, biased_scores = glm_reference_route_scores(
        router_logits, correction_bias, torch
    )
    if selected_ids is None:
        selected_ids = torch.topk(
            biased_scores, TOPK, dim=-1, largest=True, sorted=False
        ).indices
    if (
        not isinstance(selected_ids, torch.Tensor)
        or tuple(selected_ids.shape) != (int(router_logits.shape[0]), TOPK)
        or selected_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("selected route IDs must have shape [rows, 8]")
    selected_ids = selected_ids.detach().to(
        device=unbiased_scores.device, dtype=torch.int64
    )
    reference_weights = unbiased_scores.gather(1, selected_ids)
    denominator = reference_weights.sum(dim=1, keepdim=True)
    if not bool(torch.isfinite(denominator).all().item()) or bool(
        (denominator <= 0).any().item()
    ):
        raise ValueError("reference routing normalization is invalid")
    reference_weights = (
        reference_weights / denominator * ROUTED_SCALING_FACTOR
    ).to(torch.float32)
    return reference_weights, selected_ids


def compare_admissible_live_route(
    returned_weights,
    returned_ids,
    reference_weights,
    unbiased_scores,
    biased_scores,
    torch,
) -> dict[str, float | int]:
    """Certify live IDs against a tight independent numerical boundary.

    The returned set is admissible iff no unselected biased score exceeds the
    weakest selected score by more than the measured 2e-6 fused-vs-Torch
    sigmoid boundary.  Weights remain independently reconstructed and checked
    on the exact live IDs.  Thus a numerical cutoff ambiguity is preserved,
    while an arbitrary or materially sub-top-k expert still fails closed.
    """

    if (
        returned_weights.shape != reference_weights.shape
        or returned_weights.shape != returned_ids.shape
        or unbiased_scores.shape != biased_scores.shape
        or int(unbiased_scores.shape[0]) != int(returned_ids.shape[0])
        or int(unbiased_scores.shape[1]) != NUM_EXPERTS
    ):
        raise ValueError("returned/reference route shapes differ")
    ids = returned_ids.to(device=biased_scores.device, dtype=torch.int64)
    selected_mask = torch.zeros_like(biased_scores, dtype=torch.bool)
    selected_mask.scatter_(1, ids, True)
    selected_min = biased_scores.gather(1, ids).min(dim=1).values
    unselected_max = biased_scores.masked_fill(
        selected_mask, float("-inf")
    ).max(dim=1).values
    selection_excess = unselected_max - selected_min
    # Zero means an exact cutoff tie.  A small positive value means the live
    # fused sigmoid and the independent Torch sigmoid straddled the boundary.
    boundary_ambiguous = selection_excess >= 0.0
    selection_bad = selection_excess > REFERENCE_ROUTE_SELECTION_ATOL

    absolute = (
        returned_weights.to(torch.float32) - reference_weights.to(torch.float32)
    ).abs()
    tolerance = REFERENCE_ROUTE_ATOL + (
        REFERENCE_ROUTE_RTOL * reference_weights.to(torch.float32).abs()
    )
    weight_bad = (absolute > tolerance).any(dim=1)
    return {
        "checked_rows": int(returned_ids.shape[0]),
        "selection_violation_rows": int(selection_bad.sum().item()),
        "boundary_ambiguous_rows": int(boundary_ambiguous.sum().item()),
        "max_selection_violation": (
            max(0.0, float(selection_excess.max().item()))
            if selection_excess.numel()
            else 0.0
        ),
        "weight_mismatch_rows": int(weight_bad.sum().item()),
        "max_abs_weight_error": (
            float(absolute.max().item()) if absolute.numel() else 0.0
        ),
    }


def install_post_select_capture(router, capture_fn):
    """Wrap one router instance and capture only after its original call.

    Returns opaque restoration state consumed by
    :func:`restore_select_capture`.  Kept independent of vLLM so the ordering
    and exact-return contract can be unit tested without loading a model.
    """

    if not callable(getattr(router, "select_experts", None)):
        raise ValueError("router has no callable select_experts")
    had_instance_override = "select_experts" in getattr(router, "__dict__", {})
    previous_override = getattr(router, "__dict__", {}).get("select_experts")
    original = router.select_experts

    def wrapped(router_self, *args, **kwargs):
        result = original(*args, **kwargs)
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("router.select_experts return schema changed")
        router_logits = kwargs.get("router_logits")
        if router_logits is None and len(args) > 1:
            router_logits = args[1]
        capture_fn(hidden, router_logits, result[0], result[1])
        return result

    router.select_experts = types.MethodType(wrapped, router)
    return router, had_instance_override, previous_override


def restore_select_capture(state) -> None:
    router, had_instance_override, previous_override = state
    if had_instance_override:
        router.select_experts = previous_override
    else:
        delattr(router, "select_experts")


class FreshSQGCaptureWorkerExtension:
    """Collective-RPC extension mixed into each vLLM worker."""

    def _fresh_rank(self) -> int:
        try:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

            return int(get_tensor_model_parallel_rank())
        except Exception:
            return int(getattr(self, "rank", 0))

    def _fresh_world(self) -> int:
        try:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_world_size,
            )

            return int(get_tensor_model_parallel_world_size())
        except Exception:
            return -1

    def _fresh_model_runner(self):
        runner = getattr(self, "model_runner", None)
        if runner is None:
            raise RuntimeError("vLLM worker has no model_runner")
        return runner

    def _fresh_model(self):
        runner = self._fresh_model_runner()
        return runner.get_model() if hasattr(runner, "get_model") else runner.model

    def _fresh_vllm_config(self):
        runner = self._fresh_model_runner()
        config = getattr(runner, "vllm_config", None)
        if config is not None:
            return config
        from vllm.config import get_current_vllm_config

        return get_current_vllm_config()

    def _fresh_runtime_contract(self) -> dict:
        config = self._fresh_vllm_config()
        parallel = config.parallel_config
        scheduler = config.scheduler_config
        cache = config.cache_config
        model_config = config.model_config
        load_format = config.load_config.load_format
        load_format = getattr(load_format, "value", load_format)
        attention_backend = config.attention_config.backend
        attention_backend = getattr(
            attention_backend, "name", getattr(attention_backend, "value", attention_backend)
        )
        dcp_interleave_raw = int(parallel.dcp_kv_cache_interleave_size)
        cp_interleave_raw = int(parallel.cp_kv_cache_interleave_size)
        effective_interleave = effective_kv_cache_interleave_size(
            decode_context_parallel_size=int(
                parallel.decode_context_parallel_size
            ),
            dcp_kv_cache_interleave_size_raw=dcp_interleave_raw,
            cp_kv_cache_interleave_size_raw=cp_interleave_raw,
        )
        observed = {
            "tensor_parallel_size": int(parallel.tensor_parallel_size),
            "pipeline_parallel_size": int(parallel.pipeline_parallel_size),
            "data_parallel_size": int(parallel.data_parallel_size),
            "decode_context_parallel_size": int(
                parallel.decode_context_parallel_size
            ),
            "dcp_comm_backend": str(parallel.dcp_comm_backend),
            "dcp_kv_cache_interleave_size_raw": dcp_interleave_raw,
            "cp_kv_cache_interleave_size_raw": cp_interleave_raw,
            "effective_kv_cache_interleave_size": effective_interleave,
            "kv_cache_interleave_resolution": KV_CACHE_INTERLEAVE_RESOLUTION,
            "enable_expert_parallel": bool(parallel.enable_expert_parallel),
            "enable_eplb": bool(parallel.enable_eplb),
            "moe_backend": str(config.kernel_config.moe_backend),
            "use_sequence_parallel_moe": bool(parallel.use_sequence_parallel_moe),
            "enable_dbo": bool(parallel.enable_dbo),
            "disable_custom_all_reduce": bool(parallel.disable_custom_all_reduce),
            "max_num_seqs": int(scheduler.max_num_seqs),
            "max_model_len": int(model_config.max_model_len),
            "max_num_batched_tokens": int(scheduler.max_num_batched_tokens),
            "gpu_memory_utilization": float(cache.gpu_memory_utilization),
            "kv_cache_memory_bytes": int(cache.kv_cache_memory_bytes),
            "async_scheduling": bool(scheduler.async_scheduling),
            "enable_chunked_prefill": bool(scheduler.enable_chunked_prefill),
            "enable_prefix_caching": bool(cache.enable_prefix_caching),
            "speculative_config": config.speculative_config is not None,
            "enforce_eager": bool(model_config.enforce_eager),
            "quantization": str(model_config.quantization),
            "load_format": str(load_format),
            "attention_backend": str(attention_backend),
            "kv_cache_dtype": str(cache.cache_dtype),
            "python_executable": sys.executable,
            "runtime_image_id_declared": os.environ.get(IMAGE_DECLARATION_ENV),
            "exl3_environment": {
                key: os.environ.get(key) for key in RUNTIME_ENV
            },
        }
        expected = {
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "decode_context_parallel_size": 4,
            "dcp_comm_backend": "a2a",
            "dcp_kv_cache_interleave_size_raw": 64,
            "cp_kv_cache_interleave_size_raw": 1,
            "effective_kv_cache_interleave_size": 64,
            "kv_cache_interleave_resolution": KV_CACHE_INTERLEAVE_RESOLUTION,
            "enable_expert_parallel": False,
            "enable_eplb": False,
            "moe_backend": "b12x",
            "use_sequence_parallel_moe": False,
            "enable_dbo": False,
            "disable_custom_all_reduce": True,
            "max_num_seqs": 1,
            "max_model_len": 4_352,
            "max_num_batched_tokens": 2_048,
            "gpu_memory_utilization": 0.90,
            "kv_cache_memory_bytes": 268_435_456,
            "async_scheduling": False,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "speculative_config": False,
            "enforce_eager": True,
            "quantization": "exl3",
            "load_format": "safetensors",
            "attention_backend": ATTENTION_BACKEND,
            "kv_cache_dtype": EFFECTIVE_KV_CACHE_DTYPE,
            "python_executable": RUNTIME_PYTHON,
            "runtime_image_id_declared": RUNTIME_IMAGE_ID,
            "exl3_environment": RUNTIME_ENV,
        }
        if observed != expected:
            raise RuntimeError(
                f"fresh-SQG capture runtime contract differs: {observed} != {expected}"
            )
        if self._fresh_world() != 4:
            raise RuntimeError(f"fresh-SQG capture requires TP4, got {self._fresh_world()}")
        return observed

    @staticmethod
    def _fresh_hf_contract(config) -> dict:
        hf = config.model_config.hf_config
        observed = {
            "hidden_size": int(hf.hidden_size),
            "moe_intermediate_size": int(hf.moe_intermediate_size),
            "n_routed_experts": int(hf.n_routed_experts),
            "num_experts_per_tok": int(hf.num_experts_per_tok),
            "routed_scaling_factor": float(hf.routed_scaling_factor),
            "norm_topk_prob": bool(hf.norm_topk_prob),
            "n_group": int(hf.n_group),
            "topk_group": int(hf.topk_group),
            "hidden_act": str(hf.hidden_act),
            "num_hidden_layers": int(hf.num_hidden_layers),
            "use_index_cache": bool(hf.use_index_cache),
            "index_topk_pattern": str(hf.index_topk_pattern),
        }
        expected = {
            "hidden_size": HIDDEN,
            "moe_intermediate_size": 2_048,
            "n_routed_experts": NUM_EXPERTS,
            "num_experts_per_tok": TOPK,
            "routed_scaling_factor": ROUTED_SCALING_FACTOR,
            "norm_topk_prob": True,
            "n_group": 1,
            "topk_group": 1,
            "hidden_act": "silu",
            "num_hidden_layers": 78,
            "use_index_cache": True,
            "index_topk_pattern": INDEX_TOPK_PATTERN,
        }
        if observed != expected:
            raise RuntimeError(f"GLM-5.2 MoE contract differs: {observed} != {expected}")
        return observed

    def _fresh_discover_routers(
        self,
    ) -> dict[int, tuple[str, object, object, object]]:
        model = self._fresh_model()
        modules = dict(model.named_modules())
        found: dict[int, tuple[str, object, object, object]] = {}
        pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")
        for name, module in modules.items():
            match = pattern.search(name)
            if match is None:
                continue
            layer = int(match.group(1))
            if layer not in SELECTED_LAYERS:
                continue
            router = getattr(module, "router", None)
            if router is None or not callable(getattr(router, "select_experts", None)):
                raise RuntimeError(f"layer {layer}: live MoE router.select_experts absent")
            if layer in found:
                raise RuntimeError(f"layer {layer}: multiple live MoE runners discovered")
            owner_name = name[: -len(".experts")]
            owner = modules.get(owner_name)
            if owner is None:
                raise RuntimeError(f"layer {layer}: owning GLM MoE module is absent")
            found[layer] = (name, owner, module, router)
        if sorted(found) != list(SELECTED_LAYERS):
            raise RuntimeError(
                f"selected live routers {sorted(found)} != {list(SELECTED_LAYERS)}"
            )
        return found

    def _fresh_fused_layer_audit(self) -> dict:
        """Observe, do not infer, the exact post-load all-MCG teacher modes."""

        modules = dict(self._fresh_model().named_modules())
        pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")
        discovered: dict[int, object] = {}
        budget_counters: set[int] = set()
        fused: list[int] = []
        for name, module in modules.items():
            match = pattern.search(name)
            if match is None:
                continue
            layer = int(match.group(1))
            if layer not in ROUTED_LAYERS:
                continue
            if layer in discovered:
                raise RuntimeError(
                    f"layer {layer}: multiple routed expert modules in fused audit"
                )
            routed_experts = getattr(module, "routed_experts", module)
            quant_method = getattr(routed_experts, "quant_method", None)
            quant_config = getattr(quant_method, "quant_config", None)
            r7 = getattr(quant_config, "r7_routed_experts", None)
            if not isinstance(r7, dict) or r7.get("codebook") != "mcg":
                raise RuntimeError(
                    f"layer {layer}: capture teacher is not the all-MCG R7 source"
                )
            counter = getattr(quant_config, "_r7_fused_layers", None)
            if type(counter) is not int:
                raise RuntimeError(
                    f"layer {layer}: post-load fused budget counter is absent"
                )
            budget_counters.add(counter)
            is_fused = bool(getattr(routed_experts, "exl3_r7_fused", False))
            if is_fused:
                if not isinstance(
                    getattr(routed_experts, "exl3_mixed_trellis", None), dict
                ):
                    raise RuntimeError(
                        f"layer {layer}: fused flag lacks mixed-trellis payload"
                    )
                fused.append(layer)
            discovered[layer] = routed_experts
        if sorted(discovered) != list(ROUTED_LAYERS):
            raise RuntimeError(
                "post-load routed-layer population differs: "
                f"{sorted(discovered)} != {list(ROUTED_LAYERS)}"
            )
        fused = sorted(fused)
        audit = {
            "schema": "glm52-r33-all-mcg-teacher-fused-layer-audit-v1",
            "routed_layers": list(ROUTED_LAYERS),
            "fused_layers": fused,
            "nonfused_layers": sorted(set(ROUTED_LAYERS) - set(fused)),
            "fused_count": len(fused),
            "configured_budget": int(os.environ["VLLM_EXL3_R7_FUSED_LAYERS"]),
            "observed_budget_counters": sorted(budget_counters),
            "reserved_layers": sorted(set(EXPECTED_FUSED_LAYERS) - set(fused)),
            "selected_layer_modes": {
                str(layer): "fused" if layer in fused else "nonfused"
                for layer in SELECTED_LAYERS
            },
            "teacher_codebook": "mcg",
        }
        try:
            return validate_fused_layer_audit(audit)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    @staticmethod
    def _fresh_router_audit(
        layer: int,
        name: str,
        owner,
        module,
        router,
        *,
        class_sources: dict | None = None,
    ) -> dict:
        routed_experts = getattr(module, "routed_experts", module)
        quant_method = routed_experts.quant_method
        runner = getattr(module, "runner", module)
        router_scale = float(getattr(router, "routed_scaling_factor", math.nan))
        # FusedMoE(...) returns a MoERunner in the exact serving vLLM.  The
        # scale on that runner is the post-expert output scale.  The scale on
        # RoutedExperts is instead the scale already applied by the router.
        # Do not infer either value from the owning DeepseekV2MoE config.
        container_scale = float(
            getattr(routed_experts, "routed_scaling_factor", math.nan)
        )
        runner_output_scale = float(
            getattr(runner, "routed_scaling_factor", math.nan)
        )
        owner_configured_scale = float(
            getattr(owner, "routed_scaling_factor", math.nan)
        )
        module_output_scale = float(
            getattr(module, "routed_scaling_factor", math.nan)
        )
        if class_sources is None:
            class_sources = {}
            for role, value in (
                ("owner", owner),
                ("runner", runner),
                ("router", router),
                ("quant_method", quant_method),
            ):
                source = inspect.getsourcefile(type(value))
                if source is None:
                    raise RuntimeError(
                        f"layer {layer}: {role} class source is unavailable"
                    )
                path = str(Path(source).absolute())
                class_sources[role] = {
                    "class": type(value).__name__,
                    "path": path,
                    "sha256": sha256_file(path),
                }
        if class_sources != RUNTIME_CLASS_SOURCES:
            raise RuntimeError(
                f"layer {layer}: live runner/router class sources differ: "
                f"{class_sources} != {RUNTIME_CLASS_SOURCES}"
            )
        audit = {
            "layer": layer,
            "module_name": name,
            "module_class": type(module).__name__,
            "router_class": type(router).__name__,
            "quant_method_class": type(quant_method).__name__,
            "quant_method_is_monolithic": bool(quant_method.is_monolithic),
            "router_top_k": int(getattr(router, "top_k", -1)),
            "router_global_num_experts": int(
                getattr(router, "global_num_experts", -1)
            ),
            "router_renormalize": bool(getattr(router, "renormalize", False)),
            "router_scoring_func": str(getattr(router, "scoring_func", None)),
            "router_routed_scaling_factor": router_scale,
            "router_num_expert_group": int(
                getattr(router, "num_expert_group", 1)
            ),
            "router_topk_group": int(getattr(router, "topk_group", 1)),
            "router_num_fused_shared_experts": int(
                getattr(router, "num_fused_shared_experts", 0)
            ),
            "router_eplb_enabled": bool(getattr(router, "enable_eplb", False)),
            "expert_container_routed_scaling_factor": container_scale,
            "runner_output_scale": runner_output_scale,
            "owner_configured_routed_scaling_factor": owner_configured_scale,
            "module_output_scale": module_output_scale,
            "class_sources": class_sources,
            "effective_routed_scaling_factor": router_scale * runner_output_scale,
            "runner_enable_dbo": bool(getattr(runner, "enable_dbo", False)),
            "runner_has_input_transform": (
                getattr(runner, "routed_input_transform", None) is not None
                or getattr(module, "_routed_input_transform", None) is not None
            ),
            "runner_has_output_transform": (
                getattr(runner, "routed_output_transform", None) is not None
                or getattr(module, "routed_output_transform", None) is not None
                or getattr(module, "_routed_output_transform", None) is not None
            ),
            "owner_class": type(owner).__name__,
            "owner_rocm_aiter_moe_enabled": bool(
                getattr(owner, "is_rocm_aiter_moe_enabled", False)
            ),
            "moe_use_ep": bool(getattr(module, "use_ep", False)),
            "moe_sequence_parallel": bool(
                getattr(module, "is_sequence_parallel", False)
            ),
            "apply_router_weight_on_input": bool(
                routed_experts.apply_router_weight_on_input
            ),
        }
        required = {
            "module_class": "MoERunner",
            "router_class": "GroupedTopKRouter",
            "quant_method_class": "Exl3MoEMethod",
            "owner_class": "DeepseekV2MoE",
            "quant_method_is_monolithic": False,
            "router_top_k": TOPK,
            "router_global_num_experts": NUM_EXPERTS,
            "router_renormalize": True,
            "router_scoring_func": "sigmoid",
            "router_num_expert_group": 1,
            "router_topk_group": 1,
            "router_num_fused_shared_experts": 0,
            "router_eplb_enabled": False,
            "runner_enable_dbo": False,
            "runner_has_input_transform": False,
            "runner_has_output_transform": False,
            "moe_use_ep": False,
            "moe_sequence_parallel": False,
            "apply_router_weight_on_input": False,
            "owner_rocm_aiter_moe_enabled": False,
        }
        for key, expected in required.items():
            if audit[key] != expected:
                raise RuntimeError(
                    f"layer {layer}: router contract {key}={audit[key]!r}, "
                    f"expected {expected!r}"
                )
        if not all(
            math.isfinite(value)
            for value in (
                router_scale,
                container_scale,
                runner_output_scale,
                owner_configured_scale,
                module_output_scale,
            )
        ):
            raise RuntimeError(f"layer {layer}: routing scale placement is unavailable")
        if not math.isclose(
            container_scale, router_scale, rel_tol=0.0, abs_tol=0.0
        ):
            raise RuntimeError(
                f"layer {layer}: expert-container/router scales disagree: "
                f"{container_scale} != {router_scale}"
            )
        if not math.isclose(
            owner_configured_scale,
            runner_output_scale,
            rel_tol=0.0,
            abs_tol=0.0,
        ) or not math.isclose(
            module_output_scale,
            runner_output_scale,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError(
                f"layer {layer}: owner/module/runner output scales disagree: "
                f"owner={owner_configured_scale}, module={module_output_scale}, "
                f"runner={runner_output_scale}"
            )
        if router_scale <= 0.0 or runner_output_scale <= 0.0 or not math.isclose(
            router_scale * runner_output_scale,
            ROUTED_SCALING_FACTOR,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"layer {layer}: unknown scaling placement; router={router_scale}, "
                f"runner={runner_output_scale}, expected sole product 2.5"
            )
        return audit

    def fresh_sqg_capture_init(self, out_dir: str, plan_fingerprint: str) -> dict:
        """Validate runtime and install exact post-original router wrappers."""

        import torch

        if hasattr(self, "_fresh_initialized"):
            raise RuntimeError("fresh-SQG capture is already initialized")
        if sys.byteorder != "little":
            raise RuntimeError("fresh-SQG raw capture ABI requires a little-endian host")
        runtime = self._fresh_runtime_contract()
        config = self._fresh_vllm_config()
        hf_contract = self._fresh_hf_contract(config)
        fused_layer_audit = self._fresh_fused_layer_audit()
        rank = self._fresh_rank()
        self._fresh_initialized = True
        self._fresh_plan_fingerprint = str(plan_fingerprint)
        if rank != 0:
            return {
                "rank": rank,
                "capturing": False,
                "runtime": runtime,
                "hf_contract": hf_contract,
                "fused_layer_audit": fused_layer_audit,
            }

        routers = self._fresh_discover_routers()
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        self._fresh_handles = {}
        self._fresh_hashes = {}
        self._fresh_bytes = {}
        self._fresh_counts = {layer: 0 for layer in SELECTED_LAYERS}
        self._fresh_routed = {
            layer: np.zeros(NUM_EXPERTS, dtype=np.int64) for layer in SELECTED_LAYERS
        }
        self._fresh_gate_sum = {layer: 0.0 for layer in SELECTED_LAYERS}
        self._fresh_gate_sq_sum = {layer: 0.0 for layer in SELECTED_LAYERS}
        self._fresh_gate_row_min = {layer: math.inf for layer in SELECTED_LAYERS}
        self._fresh_gate_row_max = {layer: -math.inf for layer in SELECTED_LAYERS}
        self._fresh_raw_weight_hashes = {
            layer: hashlib.sha256() for layer in SELECTED_LAYERS
        }
        self._fresh_raw_weight_bytes = {layer: 0 for layer in SELECTED_LAYERS}
        self._fresh_raw_gate_sum = {layer: 0.0 for layer in SELECTED_LAYERS}
        self._fresh_raw_gate_sq_sum = {layer: 0.0 for layer in SELECTED_LAYERS}
        self._fresh_raw_gate_row_min = {
            layer: math.inf for layer in SELECTED_LAYERS
        }
        self._fresh_raw_gate_row_max = {
            layer: -math.inf for layer in SELECTED_LAYERS
        }
        self._fresh_reference_route = {
            layer: {
                "schema": REFERENCE_ROUTE_SCHEMA,
                "checked_rows": 0,
                "selection_violation_rows": 0,
                "boundary_ambiguous_rows": 0,
                "max_selection_violation": 0.0,
                "weight_mismatch_rows": 0,
                "max_abs_weight_error": 0.0,
                "selection_atol": REFERENCE_ROUTE_SELECTION_ATOL,
                "weight_rtol": REFERENCE_ROUTE_RTOL,
                "weight_atol": REFERENCE_ROUTE_ATOL,
                "coverage": "all captured rows",
                "reference": REFERENCE_ROUTE_DESCRIPTION,
            }
            for layer in SELECTED_LAYERS
        }
        self._fresh_role_rows = {
            layer: {role_id: 0 for role_id in ROLE_TO_ID.values()}
            for layer in SELECTED_LAYERS
        }
        self._fresh_doc_audit = {
            layer: hashlib.sha256() for layer in SELECTED_LAYERS
        }
        self._fresh_doc_count = 0
        self._fresh_next_epoch = 0
        self._fresh_active = None
        self._fresh_originals = {}
        self._fresh_router_audits = {}

        self._fresh_runner_output_scales = {}
        for layer, (name, owner, module, router) in routers.items():
            self._fresh_router_audits[layer] = self._fresh_router_audit(
                layer, name, owner, module, router
            )
            output_scale = float(
                self._fresh_router_audits[layer]["runner_output_scale"]
            )
            self._fresh_runner_output_scales[layer] = output_scale
            directory = root / f"layer_{layer:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            for filename in FILE_ABI:
                final = directory / filename
                partial = directory / (filename + ".partial")
                if final.exists() or partial.exists():
                    raise RuntimeError(
                        f"layer {layer}: refusing to overwrite existing {filename} payload"
                    )
                key = (layer, filename)
                self._fresh_handles[key] = partial.open("xb", buffering=16 << 20)
                self._fresh_hashes[key] = hashlib.sha256()
                self._fresh_bytes[key] = 0

            def capture(
                hidden,
                router_logits,
                topk_weights,
                topk_ids,
                __layer=layer,
                __router=router,
                __output_scale=output_scale,
            ):
                self._fresh_capture_router_result(
                    __layer,
                    hidden,
                    router_logits,
                    topk_weights,
                    topk_ids,
                    __router,
                    __output_scale,
                    torch,
                )
            self._fresh_originals[layer] = install_post_select_capture(
                router, capture
            )

        return {
            "rank": rank,
            "capturing": True,
            "layers": list(SELECTED_LAYERS),
            "runtime": runtime,
            "hf_contract": hf_contract,
            "fused_layer_audit": fused_layer_audit,
            "router_audits": {
                str(layer): value for layer, value in self._fresh_router_audits.items()
            },
            "capture_semantics": (
                "original router.select_experts runs first; wrapper records its exact return"
            ),
        }

    def _fresh_write(self, layer: int, filename: str, value: bytes) -> None:
        key = (layer, filename)
        self._fresh_handles[key].write(value)
        self._fresh_hashes[key].update(value)
        self._fresh_bytes[key] += len(value)

    def _fresh_capture_router_result(
        self,
        layer: int,
        hidden,
        router_logits,
        topk_weights,
        topk_ids,
        router,
        runner_output_scale: float,
        torch,
    ) -> None:
        active = self._fresh_active
        if active is None:
            raise RuntimeError(
                f"layer {layer}: router ran outside a declared one-document request"
            )
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.dtype != torch.bfloat16
            or hidden.ndim != 2
            or tuple(hidden.shape[1:]) != (HIDDEN,)
        ):
            raise RuntimeError(
                f"layer {layer}: hidden-state contract differs: "
                f"{getattr(hidden, 'dtype', None)} {getattr(hidden, 'shape', None)}"
            )
        rows = int(hidden.shape[0])
        if (
            topk_weights.dtype != torch.float32
            or tuple(topk_weights.shape) != (rows, TOPK)
            or tuple(topk_ids.shape) != (rows, TOPK)
            or topk_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise RuntimeError(f"layer {layer}: live router return dtype/shape differs")
        if rows <= 0 or active["positions"][layer] + rows > active["tokens"]:
            raise RuntimeError(f"layer {layer}: document row accounting overflow")
        if not bool(torch.isfinite(topk_weights).all().item()) or bool(
            (topk_weights < 0).any().item()
        ):
            raise RuntimeError(f"layer {layer}: live applied gates are invalid")
        if int(topk_ids.min().item()) < 0 or int(topk_ids.max().item()) >= NUM_EXPERTS:
            raise RuntimeError(f"layer {layer}: live routed ID is outside [0,255]")
        sorted_ids = torch.sort(topk_ids.to(torch.int64), dim=1).values
        if bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any().item()):
            raise RuntimeError(f"layer {layer}: live router returned duplicate experts")
        raw_row_sums = topk_weights.sum(dim=1)
        router_scale = float(
            self._fresh_router_audits[layer]["router_routed_scaling_factor"]
        )
        if not bool(
            torch.allclose(
                raw_row_sums,
                torch.full_like(raw_row_sums, router_scale),
                rtol=2e-6,
                atol=2e-6,
            )
        ):
            raise RuntimeError(
                f"layer {layer}: returned gates do not close to audited router scale"
            )
        effective_weights = topk_weights * float(runner_output_scale)
        effective_row_sums = effective_weights.sum(dim=1)
        if not bool(
            torch.allclose(
                effective_row_sums,
                torch.full_like(effective_row_sums, ROUTED_SCALING_FACTOR),
                rtol=2e-6,
                atol=2e-6,
            )
        ):
            raise RuntimeError(f"layer {layer}: effective applied gates do not sum to 2.5")
        self._fresh_check_reference_route(
            layer, router_logits, effective_weights, topk_ids, router, torch
        )

        hidden_cpu = hidden.detach().contiguous().cpu()
        ids_cpu = topk_ids.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        raw_weights_cpu = topk_weights.detach().to(device="cpu").contiguous()
        weights_cpu = effective_weights.detach().to(device="cpu").contiguous()
        hidden_bytes = hidden_cpu.view(torch.int16).numpy().astype("<i2", copy=False).tobytes()
        ids_array = ids_cpu.numpy()
        ids_bytes = ids_array.tobytes()
        weights_array = weights_cpu.numpy().astype("<f4", copy=False)
        weights_bytes = weights_array.tobytes()
        raw_weights_array = raw_weights_cpu.numpy().astype("<f4", copy=False)
        raw_weights_bytes = raw_weights_array.tobytes()
        start = int(active["positions"][layer])
        epochs = np.full(rows, int(active["epoch"]), dtype="<u4").tobytes()
        positions = np.arange(start, start + rows, dtype="<u2").tobytes()
        roles = np.full(rows, int(active["role_id"]), dtype="u1").tobytes()

        self._fresh_write(layer, "hidden.bf16.bin", hidden_bytes)
        self._fresh_write(layer, "topk_ids.u8.bin", ids_bytes)
        self._fresh_write(layer, "topk_weights.f32le.bin", weights_bytes)
        self._fresh_write(layer, "doc_epochs.u32le.bin", epochs)
        self._fresh_write(layer, "token_positions.u16le.bin", positions)
        self._fresh_write(layer, "role_ids.u8.bin", roles)

        self._fresh_raw_weight_hashes[layer].update(raw_weights_bytes)
        self._fresh_raw_weight_bytes[layer] += len(raw_weights_bytes)
        raw64 = raw_weights_array.astype(np.float64)
        self._fresh_raw_gate_sum[layer] += float(raw64.sum())
        self._fresh_raw_gate_sq_sum[layer] += float(np.square(raw64).sum())
        raw_row_sums64 = raw64.sum(axis=1)
        self._fresh_raw_gate_row_min[layer] = min(
            self._fresh_raw_gate_row_min[layer], float(raw_row_sums64.min())
        )
        self._fresh_raw_gate_row_max[layer] = max(
            self._fresh_raw_gate_row_max[layer], float(raw_row_sums64.max())
        )

        self._fresh_counts[layer] += rows
        active["positions"][layer] += rows
        self._fresh_role_rows[layer][int(active["role_id"])] += rows
        self._fresh_routed[layer] += np.bincount(
            ids_array.reshape(-1), minlength=NUM_EXPERTS
        )
        weights64 = weights_array.astype(np.float64)
        self._fresh_gate_sum[layer] += float(weights64.sum())
        self._fresh_gate_sq_sum[layer] += float(np.square(weights64).sum())
        row_sums64 = weights64.sum(axis=1)
        self._fresh_gate_row_min[layer] = min(
            self._fresh_gate_row_min[layer], float(row_sums64.min())
        )
        self._fresh_gate_row_max[layer] = max(
            self._fresh_gate_row_max[layer], float(row_sums64.max())
        )

    def _fresh_check_reference_route(
        self, layer: int, router_logits, returned_weights, returned_ids, router, torch
    ) -> None:
        """Independently verify GLM routing while preserving the live return."""

        rows = int(returned_ids.shape[0])
        if (
            not isinstance(router_logits, torch.Tensor)
            or tuple(router_logits.shape) != (rows, NUM_EXPERTS)
        ):
            raise RuntimeError(
                f"layer {layer}: actual router_logits unavailable for reference gate"
            )
        bias = getattr(router, "e_score_correction_bias", None)
        if bias is None or tuple(bias.shape) != (NUM_EXPERTS,):
            raise RuntimeError(f"layer {layer}: router correction bias unavailable")
        unbiased_scores, biased_scores = glm_reference_route_scores(
            router_logits, bias, torch
        )
        reference_weights, _ = glm_reference_applied_route(
            router_logits, bias, torch, selected_ids=returned_ids
        )
        observed = compare_admissible_live_route(
            returned_weights,
            returned_ids,
            reference_weights,
            unbiased_scores,
            biased_scores,
            torch,
        )
        selection_bad_count = int(observed["selection_violation_rows"])
        ambiguous_count = int(observed["boundary_ambiguous_rows"])
        max_selection_violation = float(observed["max_selection_violation"])
        weight_bad_count = int(observed["weight_mismatch_rows"])
        max_abs = float(observed["max_abs_weight_error"])
        audit = self._fresh_reference_route[layer]
        audit["checked_rows"] += rows
        audit["selection_violation_rows"] += selection_bad_count
        audit["boundary_ambiguous_rows"] += ambiguous_count
        audit["max_selection_violation"] = max(
            float(audit["max_selection_violation"]), max_selection_violation
        )
        audit["weight_mismatch_rows"] += weight_bad_count
        audit["max_abs_weight_error"] = max(
            float(audit["max_abs_weight_error"]), max_abs
        )
        if selection_bad_count or weight_bad_count:
            raise RuntimeError(
                f"layer {layer}: independent route check failed for "
                f"{selection_bad_count} selection rows / {weight_bad_count} "
                f"weight rows; max_selection_violation="
                f"{max_selection_violation:.9g}; max_abs={max_abs:.9g}"
            )

    def fresh_sqg_begin_document(
        self,
        epoch: int,
        role_id: int,
        document_sha256: str,
        tokens: int,
    ) -> dict:
        rank = self._fresh_rank()
        if rank != 0:
            return {"rank": rank, "capturing": False}
        if self._fresh_active is not None:
            raise RuntimeError("cannot begin a document while another is active")
        epoch = int(epoch)
        role_id = int(role_id)
        tokens = int(tokens)
        if epoch != self._fresh_next_epoch:
            raise RuntimeError(f"document epoch {epoch} != next {self._fresh_next_epoch}")
        if not 0 <= role_id < len(ROLE_TO_ID) or not 8 <= tokens <= 4_096:
            raise RuntimeError("document role/token count is outside the sealed contract")
        role, _ = document_role(str(document_sha256))
        if ROLE_TO_ID[role] != role_id:
            raise RuntimeError("document hash and declared split role differ")
        self._fresh_active = {
            "epoch": epoch,
            "role_id": role_id,
            "document_sha256": str(document_sha256),
            "tokens": tokens,
            "positions": {layer: 0 for layer in SELECTED_LAYERS},
        }
        return {"rank": rank, "capturing": True, "epoch": epoch}

    def fresh_sqg_end_document(self) -> dict:
        rank = self._fresh_rank()
        if rank != 0:
            return {"rank": rank, "capturing": False}
        active = self._fresh_active
        if active is None:
            raise RuntimeError("no active document to end")
        bad = {
            layer: rows
            for layer, rows in active["positions"].items()
            if int(rows) != int(active["tokens"])
        }
        if bad:
            raise RuntimeError(
                f"document epoch {active['epoch']} did not traverse every selected "
                f"layer exactly once: {bad}"
            )
        document = {
            "epoch": active["epoch"],
            "role_id": active["role_id"],
            "document_sha256": active["document_sha256"],
            "tokens": active["tokens"],
        }
        for layer in SELECTED_LAYERS:
            self._fresh_doc_audit[layer].update(
                document_audit_bytes(document, int(active["positions"][layer]))
            )
        self._fresh_doc_count += 1
        self._fresh_next_epoch += 1
        self._fresh_active = None
        return {
            "rank": rank,
            "capturing": True,
            "epoch": document["epoch"],
            "tokens": document["tokens"],
            "cumulative_tokens": min(self._fresh_counts.values()),
        }

    def fresh_sqg_capture_status(self) -> dict:
        rank = self._fresh_rank()
        if rank != 0:
            return {"rank": rank, "capturing": False}
        return {
            "rank": rank,
            "capturing": True,
            "active_epoch": (
                None if self._fresh_active is None else self._fresh_active["epoch"]
            ),
            "documents_verified": self._fresh_doc_count,
            "counts": {str(layer): self._fresh_counts[layer] for layer in SELECTED_LAYERS},
        }

    def _fresh_restore_routers(self) -> None:
        for state in self._fresh_originals.values():
            restore_select_capture(state)
        self._fresh_originals.clear()

    def fresh_sqg_capture_abort(self) -> dict:
        """Close partial payloads after a failed run; never promote or delete them."""

        rank = self._fresh_rank()
        if rank != 0 or not hasattr(self, "_fresh_handles"):
            return {"rank": rank, "capturing": False}
        self._fresh_restore_routers()
        for handle in self._fresh_handles.values():
            if not handle.closed:
                handle.flush()
                handle.close()
        return {
            "rank": rank,
            "capturing": True,
            "partial_payloads_retained": True,
            "documents_verified": self._fresh_doc_count,
        }

    def fresh_sqg_capture_finalize(self) -> dict:
        rank = self._fresh_rank()
        if rank != 0:
            return {"rank": rank, "capturing": False, "layers": {}}
        if self._fresh_active is not None:
            raise RuntimeError("cannot finalize with an active document")
        if self._fresh_doc_count != OWNER_DOCUMENTS:
            raise RuntimeError(
                f"captured {self._fresh_doc_count} documents != {OWNER_DOCUMENTS}"
            )
        if set(self._fresh_counts.values()) != {OWNER_TOKENS}:
            raise RuntimeError(f"selected-layer token totals differ: {self._fresh_counts}")
        self._fresh_restore_routers()

        layers = {}
        for layer in SELECTED_LAYERS:
            files = {}
            for filename in FILE_ABI:
                key = (layer, filename)
                handle = self._fresh_handles[key]
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
                partial = Path(handle.name)
                final = partial.with_name(filename)
                os.replace(partial, final)
                files[filename] = {
                    "bytes": self._fresh_bytes[key],
                    "sha256": self._fresh_hashes[key].hexdigest(),
                }
            layers[str(layer)] = {
                "tokens": self._fresh_counts[layer],
                "documents_verified": self._fresh_doc_count,
                "document_audit_sha256": self._fresh_doc_audit[layer].hexdigest(),
                "role_rows": {
                    str(key): int(value)
                    for key, value in self._fresh_role_rows[layer].items()
                },
                "routed_counts": self._fresh_routed[layer].tolist(),
                "gate_sum": self._fresh_gate_sum[layer],
                "gate_sq_sum": self._fresh_gate_sq_sum[layer],
                "gate_row_sum_min": self._fresh_gate_row_min[layer],
                "gate_row_sum_max": self._fresh_gate_row_max[layer],
                "router_audit": self._fresh_router_audits[layer],
                "raw_router_return_weights": {
                    "source": (
                        "exact float32 weights returned by original live "
                        "router.select_experts"
                    ),
                    "dtype": "float32-le",
                    "shape": [OWNER_TOKENS, TOPK],
                    "bytes": self._fresh_raw_weight_bytes[layer],
                    "sha256": self._fresh_raw_weight_hashes[layer].hexdigest(),
                    "payload_stored": False,
                    "effective_payload": "topk_weights.f32le.bin",
                    "router_return_scale": self._fresh_router_audits[layer][
                        "router_routed_scaling_factor"
                    ],
                    "runner_output_scale": self._fresh_runner_output_scales[layer],
                    "effective_scale_product": (
                        self._fresh_router_audits[layer][
                            "router_routed_scaling_factor"
                        ]
                        * self._fresh_runner_output_scales[layer]
                    ),
                    "sum": self._fresh_raw_gate_sum[layer],
                    "sq_sum": self._fresh_raw_gate_sq_sum[layer],
                    "row_sum_min": self._fresh_raw_gate_row_min[layer],
                    "row_sum_max": self._fresh_raw_gate_row_max[layer],
                },
                "reference_route_check": self._fresh_reference_route[layer],
                "files": files,
            }
        return {
            "rank": rank,
            "capturing": True,
            "plan_fingerprint": self._fresh_plan_fingerprint,
            "layers": layers,
        }
