"""Topology-neutral GLM SQG atoms-v2 W4A8 MoE execution.

GLM assigns K3/K4 independently to gate, up, and down for every expert.  The
prepared layer therefore owns six compact trellis pools (three projections by
two rates) and three independent expert-to-pool partitions.  No TP rank is
encoded in the checkpoint contract: vLLM slices the intermediate axis before
calling :func:`prepare_weights`, and this module records that local extent.

The runtime owns every mutable route, MXFP8, transform, and output buffer.  A
prepared runtime can consequently be reused by eager execution and CUDA graph
capture without allocating in :func:`run`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import os
import weakref

import torch

from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x.gemm._shared.wo_mxfp8 import (
    MXFP8Rows,
    empty_mxfp8_rows_for_dense_gemm,
)
from b12x.moe._shared.kernels.glm_trellis_transform import (
    run_glm_coupled_gate_up_output_transform_silu,
    run_glm_coupled_residual_h512,
    run_glm_down_input_transform,
    run_glm_down_output_transform_sum,
    run_glm_gate_up_output_transform_silu,
)
from b12x.moe._shared.kernels.glm_trellis_w4a8 import (
    GLMRoutePackedW4A8Projection,
    prepare_glm_route_packed_w4a8_projection,
    run_glm_route_packed_w4a8_projection,
)
from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
from b12x.moe._shared.kernels.w4a16.kernel import (
    _run_trellis_dense_hadamard128,
    pack_topk_routes_by_expert,
)


_ROUTE_BLOCK_ROWS = 128
_CODEBOOK = "sqg_xor_cheb_t12"
_WEIGHT_REGISTRY: weakref.WeakValueDictionary[int, GLMSQGW4A8Weights]
_RUNTIME_REGISTRY: weakref.WeakValueDictionary[int, GLMSQGW4A8Runtime]


def glm_route_packed_w4a8_kernel_contract() -> dict[str, int | str]:
    """Return the resolved route-packed schedule for serving provenance."""

    return {
        "kernel": os.environ.get("B12X_GLM_W4A8_KERNEL", "m128n64")
        .strip()
        .lower(),
        "blocks_per_cta": int(
            os.environ.get("B12X_GLM_W4A8_V2_BLOCKS", "8")
        ),
        "stages": int(os.environ.get("B12X_GLM_W4A8_V2_STAGES", "2")),
        "route_block_rows": _ROUTE_BLOCK_ROWS,
        "tile_n": 64,
    }


# Acceptance schedules, keyed by device architecture.
#
# sm_120/sm_121: the PR11 M128xN64 winner measured on a 188-SM consumer
#   Blackwell part. Frozen -- do not edit without a new accepted measurement.
# sm_100/sm_103: seeded with the same values so behaviour is currently
#   identical, but flagged PROVISIONAL: B300 has a different SM count and
#   per-SM resource budget, so the occupancy-optimal blocks_per_cta/stages
#   must be re-measured by ABBA benchmark before this entry is treated as
#   authoritative.
_PR11_M128N64_SCHEDULE = {
    "kernel": "m128n64",
    "blocks_per_cta": 8,
    "stages": 2,
    "route_block_rows": 128,
    "tile_n": 64,
}

_ACCEPTANCE_SCHEDULE_BY_ARCH = {
    (12, 0): dict(_PR11_M128N64_SCHEDULE),
    (12, 1): dict(_PR11_M128N64_SCHEDULE),
    (10, 0): dict(_PR11_M128N64_SCHEDULE),
    (10, 3): dict(_PR11_M128N64_SCHEDULE),
}

_ACCEPTANCE_SCHEDULE_PROVISIONAL = frozenset({(10, 0), (10, 3)})


def _acceptance_arch() -> tuple[int, int]:
    """Resolve the architecture key for the acceptance schedule."""
    forced = os.environ.get("B12X_GLM_W4A8_ACCEPT_ARCH", "").strip()
    if forced:
        text = forced.lower().removeprefix("sm_").removeprefix("sm")
        if not text.isdigit() or len(text) < 2:
            raise ValueError(
                "B12X_GLM_W4A8_ACCEPT_ARCH must look like 'sm_103' or '103', "
                f"got {forced!r}"
            )
        return (int(text[:-1]), int(text[-1]))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GLM SQG full-W4A8 acceptance needs a CUDA device to resolve the "
            "architecture; set B12X_GLM_W4A8_ACCEPT_ARCH for offline checks"
        )
    return tuple(torch.cuda.get_device_capability(torch.cuda.current_device()))


def glm_route_packed_w4a8_acceptance_schedule_is_provisional() -> bool:
    """True when this architecture's schedule has not yet been re-measured."""
    return _acceptance_arch() in _ACCEPTANCE_SCHEDULE_PROVISIONAL


def validate_glm_route_packed_w4a8_acceptance_kernel() -> None:
    """Reject schedules other than the accepted winner for this architecture.

    The SM120 requirement is unchanged. Other architectures must have an
    explicitly recorded entry; an unknown architecture is an error, not a pass.
    """

    arch = _acceptance_arch()
    expected = _ACCEPTANCE_SCHEDULE_BY_ARCH.get(arch)
    if expected is None:
        raise RuntimeError(
            "GLM SQG full-W4A8 acceptance has no recorded schedule for "
            f"sm_{arch[0]}{arch[1]}; record a measured schedule in "
            "_ACCEPTANCE_SCHEDULE_BY_ARCH rather than bypassing this check"
        )
    resolved = glm_route_packed_w4a8_kernel_contract()
    if resolved != expected:
        raise RuntimeError(
            f"GLM SQG full-W4A8 acceptance requires the accepted "
            f"sm_{arch[0]}{arch[1]} schedule {expected}, got {resolved}"
        )


@dataclass(frozen=True)
class GLMSQGW4A8Weights:
    """One TP-local view of a topology-neutral GLM atoms-v2 layer."""

    gate: GLMRoutePackedW4A8Projection
    up: GLMRoutePackedW4A8Projection
    down: GLMRoutePackedW4A8Projection
    gate_up_suh: torch.Tensor
    gate_svh: torch.Tensor
    up_svh: torch.Tensor
    down_suh: torch.Tensor
    down_svh: torch.Tensor
    hidden_size: int
    intermediate_size: int
    global_intermediate_size: int
    num_experts: int
    tp_rank: int
    tp_size: int
    derived_down_target_id: str | None
    down_target_beta: float | None
    coupled_rotation_draws: torch.Tensor | None = None
    coupled_pre_signs: torch.Tensor | None = None
    coupled_post_signs: torch.Tensor | None = None
    coupled_hadamard: bool = False
    codebook: str = _CODEBOOK
    direct_e4m3_weights: bool = True
    allow_a16_fallback: bool = False


@dataclass(frozen=True)
class GLMSQGW4A8Runtime:
    """Fixed-capacity caller-owned storage for one GLM W4A8 execution scope."""

    max_tokens: int
    topk: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    input_f16: torch.Tensor
    hidden_boundary_h128: torch.Tensor
    hidden_coupled: torch.Tensor
    hidden_rotated: torch.Tensor
    hidden_quantized: MXFP8Rows
    gate_transformed: torch.Tensor
    up_transformed: torch.Tensor
    gate_hadamard: torch.Tensor
    up_hadamard: torch.Tensor
    pre_scaled: torch.Tensor
    pre_hadamard: torch.Tensor
    activation_signed: torch.Tensor
    activated: torch.Tensor
    down_scaled: torch.Tensor
    down_rotated: torch.Tensor
    down_quantized: MXFP8Rows
    down_transformed: torch.Tensor
    down_canonical: torch.Tensor
    output_coupled: torch.Tensor
    output_boundary_h128: torch.Tensor
    output: torch.Tensor
    packed_route_indices: torch.Tensor
    block_expert_ids: torch.Tensor
    packed_route_count: torch.Tensor
    expert_offsets: torch.Tensor
    expert_counts: torch.Tensor
    ones_intermediate: torch.Tensor
    ones_hidden: torch.Tensor
    ones_preactivation: torch.Tensor


def _coupled_rotation_signs(
    length: int,
    draws: torch.Tensor,
    *,
    axis: int,
    device: torch.device,
) -> torch.Tensor:
    """Materialize updated-QSRT's exact expert-static Rademacher family."""

    if length <= 0 or axis not in (1, 2):
        raise ValueError("coupled rotation sign geometry differs")
    result: list[torch.Tensor] = []
    for raw_draw in draws.detach().cpu().tolist():
        draw = int(raw_draw)
        if not 0 <= draw < 8:
            raise ValueError("coupled rotation draw must lie in 0..7")
        if draw == 0:
            signs = torch.ones(length, dtype=torch.float32)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                (
                    0x6A09E667F3BCC909 * draw
                    + 0xBB67AE8584CAA73B * axis
                )
                & ((1 << 63) - 1)
            )
            signs = (
                torch.randint(0, 2, (length,), generator=generator)
                .mul_(2)
                .sub_(1)
                .float()
            )
        result.append(signs)
    return torch.stack(result).half().to(device=device).contiguous()


def _validate_scale(
    name: str,
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if (
        value.dtype != torch.float16
        or value.device != device
        or tuple(value.shape) != shape
        or not value.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous FP16 {shape} on {device}, got "
            f"{tuple(value.shape)}/{value.dtype}/{value.device}"
        )
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} contains non-finite values")


def _validate_projection_partition(
    name: str,
    projection: GLMRoutePackedW4A8Projection,
) -> None:
    slots3 = projection.expert_slots_k3.detach().cpu()
    slots4 = projection.expert_slots_k4.detach().cpu()
    owns3 = slots3 >= 0
    owns4 = slots4 >= 0
    if not bool(torch.all(owns3 ^ owns4)):
        raise ValueError(f"{name} K3/K4 slot maps do not partition all experts")
    for bits, slots, owns, pool in (
        (3, slots3, owns3, projection.trellis_k3),
        (4, slots4, owns4, projection.trellis_k4),
    ):
        live = slots[owns]
        expected = torch.arange(live.numel(), dtype=torch.int32)
        if not torch.equal(torch.sort(live).values, expected):
            raise ValueError(f"{name} K{bits} slots are not dense and unique")
        expected_numel = (
            live.numel()
            * int(projection.size_k)
            * int(projection.size_n)
            * bits
            // 16
        )
        if int(pool.numel()) != expected_numel:
            raise ValueError(
                f"{name} K{bits} pool has {pool.numel()} int16 values; "
                f"expected {expected_numel}"
            )


def prepare_weights(
    *,
    gate_trellis: Sequence[torch.Tensor],
    gate_bits: Sequence[int],
    up_trellis: Sequence[torch.Tensor],
    up_bits: Sequence[int],
    down_trellis: Sequence[torch.Tensor],
    down_bits: Sequence[int],
    gate_up_suh: torch.Tensor,
    gate_svh: torch.Tensor,
    up_svh: torch.Tensor,
    down_suh: torch.Tensor,
    down_svh: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    global_intermediate_size: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    derived_down_target_id: str | None = None,
    down_target_beta: float | None = None,
    coupled_rotation_draws: torch.Tensor | None = None,
) -> GLMSQGW4A8Weights:
    """Prepare six independent K3/K4 pools from one TP-local atoms-v2 extent.

    ``intermediate_size`` is the local TP width.  Rate choices remain per
    logical tensor and are never coupled across gate/up/down or rewritten by
    topology.
    """

    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    tp_rank = int(tp_rank)
    tp_size = int(tp_size)
    if global_intermediate_size is None:
        global_intermediate_size = intermediate_size * tp_size
    global_intermediate_size = int(global_intermediate_size)
    if hidden_size <= 0 or hidden_size % 128:
        raise ValueError("GLM SQG W4A8 hidden_size must be a multiple of 128")
    if intermediate_size <= 0 or intermediate_size % 128:
        raise ValueError(
            "GLM SQG W4A8 local intermediate_size must be a multiple of 128"
        )
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError("tp_rank must identify one rank in tp_size")
    if global_intermediate_size != intermediate_size * tp_size:
        raise ValueError(
            "topology-neutral intermediate extent does not close exactly: "
            f"global={global_intermediate_size}, local={intermediate_size}, "
            f"tp={tp_size}"
        )
    if (derived_down_target_id is None) != (down_target_beta is None):
        raise ValueError(
            "derived_down_target_id and down_target_beta must be supplied together"
        )
    if down_target_beta is not None and (
        isinstance(down_target_beta, bool)
        or not math.isfinite(float(down_target_beta))
    ):
        raise ValueError("down_target_beta must be a finite real value")
    num_experts = len(gate_trellis)
    if num_experts <= 0 or any(
        len(values) != num_experts
        for values in (
            gate_bits,
            up_trellis,
            up_bits,
            down_trellis,
            down_bits,
        )
    ):
        raise ValueError("every GLM projection must describe every expert once")
    device = gate_up_suh.device
    coupled_pre_signs: torch.Tensor | None = None
    coupled_post_signs: torch.Tensor | None = None
    stored_draws: torch.Tensor | None = None
    if coupled_rotation_draws is not None:
        if (
            coupled_rotation_draws.dtype != torch.uint8
            or tuple(coupled_rotation_draws.shape) != (num_experts,)
            or not coupled_rotation_draws.is_contiguous()
        ):
            raise TypeError(
                "coupled_rotation_draws must be contiguous uint8 [experts]"
            )
        stored_draws = coupled_rotation_draws.detach().to(
            device=device, copy=True
        ).contiguous()
        global_pre_signs = _coupled_rotation_signs(
            2 * global_intermediate_size,
            coupled_rotation_draws,
            axis=1,
            device=device,
        )
        pre_start = 2 * tp_rank * intermediate_size
        coupled_pre_signs = global_pre_signs[
            :, pre_start : pre_start + 2 * intermediate_size
        ].contiguous()
        global_post_signs = _coupled_rotation_signs(
            global_intermediate_size,
            coupled_rotation_draws,
            axis=2,
            device=device,
        )
        post_start = tp_rank * intermediate_size
        coupled_post_signs = global_post_signs[
            :, post_start : post_start + intermediate_size
        ].contiguous()

    gate = prepare_glm_route_packed_w4a8_projection(
        gate_trellis,
        gate_bits,
        size_k=hidden_size,
        size_n=intermediate_size,
    )
    up = prepare_glm_route_packed_w4a8_projection(
        up_trellis,
        up_bits,
        size_k=hidden_size,
        size_n=intermediate_size,
    )
    down = prepare_glm_route_packed_w4a8_projection(
        down_trellis,
        down_bits,
        size_k=intermediate_size,
        size_n=hidden_size,
    )
    for name, projection in (("gate", gate), ("up", up), ("down", down)):
        if projection.trellis_k3.device != device or projection.trellis_k4.device != device:
            raise ValueError(f"{name} trellis pools must be on {device}")
        _validate_projection_partition(name, projection)

    _validate_scale(
        "gate_up_suh", gate_up_suh, shape=(hidden_size,), device=device
    )
    _validate_scale(
        "gate_svh",
        gate_svh,
        shape=(num_experts, intermediate_size),
        device=device,
    )
    _validate_scale(
        "up_svh",
        up_svh,
        shape=(num_experts, intermediate_size),
        device=device,
    )
    _validate_scale(
        "down_suh",
        down_suh,
        shape=(num_experts, intermediate_size),
        device=device,
    )
    _validate_scale("down_svh", down_svh, shape=(hidden_size,), device=device)

    prepared = GLMSQGW4A8Weights(
        gate=gate,
        up=up,
        down=down,
        gate_up_suh=gate_up_suh,
        gate_svh=gate_svh,
        up_svh=up_svh,
        down_suh=down_suh,
        down_svh=down_svh,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        global_intermediate_size=global_intermediate_size,
        num_experts=num_experts,
        tp_rank=tp_rank,
        tp_size=tp_size,
        derived_down_target_id=derived_down_target_id,
        down_target_beta=(
            None if down_target_beta is None else float(down_target_beta)
        ),
        coupled_rotation_draws=stored_draws,
        coupled_pre_signs=coupled_pre_signs,
        coupled_post_signs=coupled_post_signs,
        coupled_hadamard=stored_draws is not None,
    )
    _WEIGHT_REGISTRY[id(prepared)] = prepared
    return prepared


def prepare_runtime(
    weights: GLMSQGW4A8Weights,
    *,
    max_tokens: int,
    topk: int,
    output_dtype: torch.dtype,
) -> GLMSQGW4A8Runtime:
    """Allocate fixed graph-stable storage for one serving execution scope."""

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GLM SQG W4A8 runtime must be prepared before capture")
    validate_glm_route_packed_w4a8_acceptance_kernel()
    max_tokens = int(max_tokens)
    topk = int(topk)
    if max_tokens <= 0 or topk <= 0:
        raise ValueError("max_tokens and topk must be positive")
    if output_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("GLM SQG W4A8 output must be FP16 or BF16")
    device = weights.gate_up_suh.device
    if device.type != "cuda":
        raise ValueError("GLM SQG W4A8 execution requires CUDA weights")
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((12, 0), (12, 1)):
        raise ValueError(
            "GLM SQG direct-E4M3 W4A8 requires SM120/SM121, got "
            f"SM{capability[0]}{capability[1]}"
        )
    hidden = int(weights.hidden_size)
    intermediate = int(weights.intermediate_size)
    max_routes = max_tokens * topk
    _, packed_capacity, block_capacity = route_pack_capacity(
        max_routes,
        _ROUTE_BLOCK_ROWS,
        weights.num_experts,
        topk=topk,
    )

    def f16(rows: int, cols: int) -> torch.Tensor:
        return torch.empty((rows, cols), dtype=torch.float16, device=device)

    coupled_rows = max_routes if weights.coupled_hadamard else 1
    coupled_tokens = max_tokens if weights.coupled_hadamard else 1
    coupled_hidden = hidden if weights.coupled_hadamard else 1
    coupled_intermediate = intermediate if weights.coupled_hadamard else 1

    runtime = GLMSQGW4A8Runtime(
        max_tokens=max_tokens,
        topk=topk,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=weights.num_experts,
        input_f16=f16(max_tokens, hidden),
        hidden_boundary_h128=f16(coupled_tokens, coupled_hidden),
        hidden_coupled=f16(coupled_tokens, coupled_hidden),
        hidden_rotated=f16(max_tokens, hidden),
        hidden_quantized=empty_mxfp8_rows_for_dense_gemm(
            max_tokens, hidden, device=device
        ),
        gate_transformed=f16(max_routes, intermediate),
        up_transformed=f16(max_routes, intermediate),
        gate_hadamard=f16(max_routes, intermediate),
        up_hadamard=f16(max_routes, intermediate),
        pre_scaled=f16(coupled_rows, 2 * coupled_intermediate),
        pre_hadamard=f16(coupled_rows, 2 * coupled_intermediate),
        activation_signed=f16(coupled_rows, coupled_intermediate),
        activated=f16(max_routes, intermediate),
        down_scaled=f16(max_routes, intermediate),
        down_rotated=f16(max_routes, intermediate),
        down_quantized=empty_mxfp8_rows_for_dense_gemm(
            max_routes, intermediate, device=device
        ),
        down_transformed=f16(max_routes, hidden),
        down_canonical=f16(max_routes, hidden),
        output_coupled=f16(coupled_tokens, coupled_hidden),
        output_boundary_h128=f16(coupled_tokens, coupled_hidden),
        output=torch.empty(
            (max_tokens, hidden), dtype=output_dtype, device=device
        ),
        packed_route_indices=torch.empty(
            packed_capacity, dtype=torch.int32, device=device
        ),
        block_expert_ids=torch.empty(
            block_capacity, dtype=torch.int32, device=device
        ),
        packed_route_count=torch.empty(1, dtype=torch.int32, device=device),
        expert_offsets=torch.empty(
            weights.num_experts + 1, dtype=torch.int32, device=device
        ),
        expert_counts=torch.empty(
            weights.num_experts, dtype=torch.int32, device=device
        ),
        ones_intermediate=torch.ones(
            intermediate, dtype=torch.float16, device=device
        ),
        ones_hidden=torch.ones(hidden, dtype=torch.float16, device=device),
        ones_preactivation=torch.ones(
            2 * intermediate, dtype=torch.float16, device=device
        ),
    )
    _RUNTIME_REGISTRY[id(runtime)] = runtime
    return runtime


def _active_quantized(value: MXFP8Rows, rows: int) -> MXFP8Rows:
    scale_rows = (
        value.scale_rows[:, :rows]
        if value.scale_rows.ndim == 3
        else value.scale_rows[:rows]
    )
    return MXFP8Rows(
        values=value.values[:rows],
        scale_rows=scale_rows,
        scale_mma=value.scale_mma,
    )


def _run_impl(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weights: GLMSQGW4A8Weights,
    runtime: GLMSQGW4A8Runtime,
) -> torch.Tensor:
    """Run the complete route-packed GLM SQG W4A8 expert layer."""

    if hidden_states.ndim != 2 or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous [tokens, hidden]")
    tokens, hidden = (int(value) for value in hidden_states.shape)
    if tokens <= 0 or tokens > runtime.max_tokens:
        raise ValueError(
            f"token count {tokens} exceeds planned capacity {runtime.max_tokens}"
        )
    if (
        runtime.hidden_size != weights.hidden_size
        or runtime.intermediate_size != weights.intermediate_size
        or runtime.num_experts != weights.num_experts
    ):
        raise ValueError("prepared weights do not match the runtime geometry")
    if hidden != weights.hidden_size:
        raise ValueError(f"hidden width {hidden} != {weights.hidden_size}")
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("hidden_states must be FP16 or BF16")
    if hidden_states.device != weights.gate_up_suh.device:
        raise ValueError("hidden_states and prepared weights must share a device")
    expected_route_shape = (tokens, runtime.topk)
    if (
        tuple(topk_ids.shape) != expected_route_shape
        or topk_ids.dtype != torch.int32
        or topk_ids.device != hidden_states.device
        or not topk_ids.is_contiguous()
    ):
        raise TypeError(
            f"topk_ids must be contiguous int32 {expected_route_shape}"
        )
    if (
        tuple(topk_weights.shape) != expected_route_shape
        or topk_weights.dtype != torch.float32
        or topk_weights.device != hidden_states.device
        or not topk_weights.is_contiguous()
    ):
        raise TypeError(
            f"topk_weights must be contiguous FP32 {expected_route_shape}"
        )

    routes = tokens * runtime.topk
    packed, block_experts, _ = pack_topk_routes_by_expert(
        topk_ids,
        _ROUTE_BLOCK_ROWS,
        weights.num_experts,
        packed_route_indices=runtime.packed_route_indices,
        block_expert_ids=runtime.block_expert_ids,
        packed_route_count=runtime.packed_route_count,
        expert_offsets=runtime.expert_offsets,
        expert_counts=runtime.expert_counts,
    )
    input_f16 = runtime.input_f16[:tokens]
    input_f16.copy_(hidden_states)
    projection_input = input_f16
    if weights.coupled_hadamard:
        projection_input = run_glm_coupled_residual_h512(
            input_f16,
            runtime.hidden_boundary_h128[:tokens],
            runtime.hidden_coupled[:tokens],
            ones=runtime.ones_hidden,
        )
    hidden_rotated = runtime.hidden_rotated[:tokens]
    _run_trellis_dense_hadamard128(
        projection_input,
        hidden_rotated,
        weights.gate_up_suh,
        scale_before=True,
    )
    hidden_quantized = _active_quantized(runtime.hidden_quantized, tokens)
    quantize_mxfp8_rows_cute(
        hidden_rotated,
        hidden_quantized.values,
        hidden_quantized.scale_rows,
        hidden_quantized.scale_mma,
        value_order="trellis_native_mma",
    )

    gate = runtime.gate_transformed[:routes]
    up = runtime.up_transformed[:routes]
    run_glm_route_packed_w4a8_projection(
        hidden_quantized,
        weights.gate,
        packed,
        block_experts,
        gate,
        topk=runtime.topk,
        shared_input=True,
    )
    run_glm_route_packed_w4a8_projection(
        hidden_quantized,
        weights.up,
        packed,
        block_experts,
        up,
        topk=runtime.topk,
        shared_input=True,
    )
    activated = runtime.activated[:routes]
    route_experts = topk_ids.reshape(-1)
    if weights.coupled_hadamard:
        if weights.coupled_pre_signs is None or weights.coupled_post_signs is None:
            raise RuntimeError("coupled GLM weights lack rotation signs")
        run_glm_coupled_gate_up_output_transform_silu(
            gate,
            up,
            route_experts,
            weights.gate_svh,
            weights.up_svh,
            weights.coupled_pre_signs,
            weights.coupled_post_signs,
            runtime.gate_hadamard[:routes],
            runtime.up_hadamard[:routes],
            runtime.pre_scaled[:routes],
            runtime.pre_hadamard[:routes],
            runtime.activation_signed[:routes],
            activated,
            ones_intermediate=runtime.ones_intermediate,
            ones_preactivation=runtime.ones_preactivation,
            tp_rank=weights.tp_rank,
            tp_size=weights.tp_size,
        )
    else:
        run_glm_gate_up_output_transform_silu(
            gate,
            up,
            route_experts,
            weights.gate_svh,
            weights.up_svh,
            runtime.gate_hadamard[:routes],
            runtime.up_hadamard[:routes],
            activated,
            ones=runtime.ones_intermediate,
        )

    down_rotated = runtime.down_rotated[:routes]
    run_glm_down_input_transform(
        activated,
        route_experts,
        weights.down_suh,
        runtime.down_scaled[:routes],
        down_rotated,
        ones=runtime.ones_intermediate,
    )
    down_quantized = _active_quantized(runtime.down_quantized, routes)
    quantize_mxfp8_rows_cute(
        down_rotated,
        down_quantized.values,
        down_quantized.scale_rows,
        down_quantized.scale_mma,
        value_order="trellis_native_mma",
    )
    down_transformed = runtime.down_transformed[:routes]
    run_glm_route_packed_w4a8_projection(
        down_quantized,
        weights.down,
        packed,
        block_experts,
        down_transformed,
        topk=runtime.topk,
        shared_input=False,
    )
    if weights.coupled_hadamard:
        coupled_output = run_glm_down_output_transform_sum(
            down_transformed,
            topk_weights,
            weights.down_svh,
            runtime.down_canonical[:routes],
            runtime.output_coupled[:tokens],
            topk=runtime.topk,
        )
        decoded_output = run_glm_coupled_residual_h512(
            coupled_output,
            runtime.output_boundary_h128[:tokens],
            # The H512 primitive has an exact FP16 boundary.  The model output
            # may be BF16, so reuse the no-longer-live coupled input buffer and
            # perform the one explicit endpoint cast afterward.
            runtime.hidden_coupled[:tokens],
            ones=runtime.ones_hidden,
        )
        runtime.output[:tokens].copy_(decoded_output)
        return runtime.output[:tokens]
    return run_glm_down_output_transform_sum(
        down_transformed,
        topk_weights,
        weights.down_svh,
        runtime.down_canonical[:routes],
        runtime.output[:tokens],
        topk=runtime.topk,
    )


def _weight_tensors(weights: GLMSQGW4A8Weights) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for projection in (weights.gate, weights.up, weights.down):
        tensors.extend(
            (
                projection.trellis_k3,
                projection.trellis_k4,
                projection.expert_slots_k3,
                projection.expert_slots_k4,
            )
        )
    tensors.extend(
        (
            weights.gate_up_suh,
            weights.gate_svh,
            weights.up_svh,
            weights.down_suh,
            weights.down_svh,
        )
    )
    if weights.coupled_hadamard:
        if (
            weights.coupled_rotation_draws is None
            or weights.coupled_pre_signs is None
            or weights.coupled_post_signs is None
        ):
            raise RuntimeError("coupled GLM weight inventory is incomplete")
        tensors.extend(
            (
                weights.coupled_rotation_draws,
                weights.coupled_pre_signs,
                weights.coupled_post_signs,
            )
        )
    return tensors


def _runtime_tensors(runtime: GLMSQGW4A8Runtime) -> list[torch.Tensor]:
    return [
        runtime.input_f16,
        runtime.hidden_boundary_h128,
        runtime.hidden_coupled,
        runtime.hidden_rotated,
        runtime.hidden_quantized.values,
        runtime.hidden_quantized.scale_rows,
        runtime.hidden_quantized.scale_mma,
        runtime.gate_transformed,
        runtime.up_transformed,
        runtime.gate_hadamard,
        runtime.up_hadamard,
        runtime.pre_scaled,
        runtime.pre_hadamard,
        runtime.activation_signed,
        runtime.activated,
        runtime.down_scaled,
        runtime.down_rotated,
        runtime.down_quantized.values,
        runtime.down_quantized.scale_rows,
        runtime.down_quantized.scale_mma,
        runtime.down_transformed,
        runtime.down_canonical,
        runtime.output_coupled,
        runtime.output_boundary_h128,
        runtime.output,
        runtime.packed_route_indices,
        runtime.block_expert_ids,
        runtime.packed_route_count,
        runtime.expert_offsets,
        runtime.expert_counts,
        runtime.ones_intermediate,
        runtime.ones_hidden,
        runtime.ones_preactivation,
    ]


@torch.library.custom_op(
    "b12x::glm_sqg_w4a8_moe",
    mutates_args="unknown",
    device_types="cuda",
)
def _run_op(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weight_tensors: list[torch.Tensor],
    runtime_tensors: list[torch.Tensor],
    weights_id: int,
    runtime_id: int,
) -> None:
    weights = _WEIGHT_REGISTRY.get(int(weights_id))
    runtime = _RUNTIME_REGISTRY.get(int(runtime_id))
    if weights is None or runtime is None:
        raise RuntimeError("GLM SQG W4A8 prepared owner is no longer live")
    expected_weights = _weight_tensors(weights)
    expected_runtime = _runtime_tensors(runtime)
    if len(weight_tensors) != len(expected_weights):
        raise RuntimeError("GLM SQG W4A8 weight inventory changed after planning")
    if len(runtime_tensors) != len(expected_runtime):
        raise RuntimeError("GLM SQG W4A8 scratch inventory changed after planning")
    if any(
        actual.data_ptr() != expected.data_ptr()
        for actual, expected in zip(weight_tensors, expected_weights, strict=True)
    ):
        raise RuntimeError("GLM SQG W4A8 custom op received different weight storage")
    if any(
        actual.data_ptr() != expected.data_ptr()
        for actual, expected in zip(runtime_tensors, expected_runtime, strict=True)
    ):
        raise RuntimeError("GLM SQG W4A8 custom op received different scratch storage")
    _run_impl(hidden_states, topk_weights, topk_ids, weights, runtime)


@_run_op.register_fake
def _run_op_fake(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weight_tensors: list[torch.Tensor],
    runtime_tensors: list[torch.Tensor],
    weights_id: int,
    runtime_id: int,
) -> None:
    del (
        hidden_states,
        topk_weights,
        topk_ids,
        weight_tensors,
        runtime_tensors,
        weights_id,
        runtime_id,
    )


def run(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weights: GLMSQGW4A8Weights,
    runtime: GLMSQGW4A8Runtime,
) -> torch.Tensor:
    """Run through an opaque compile-safe op and return its stable output view."""

    tokens = int(hidden_states.shape[0])
    _run_op(
        hidden_states,
        topk_weights,
        topk_ids,
        _weight_tensors(weights),
        _runtime_tensors(runtime),
        id(weights),
        id(runtime),
    )
    return runtime.output[:tokens]


_WEIGHT_REGISTRY = weakref.WeakValueDictionary()
_RUNTIME_REGISTRY = weakref.WeakValueDictionary()


__all__ = [
    "GLMSQGW4A8Runtime",
    "GLMSQGW4A8Weights",
    "glm_route_packed_w4a8_kernel_contract",
    "prepare_runtime",
    "prepare_weights",
    "run",
    "validate_glm_route_packed_w4a8_acceptance_kernel",
]
