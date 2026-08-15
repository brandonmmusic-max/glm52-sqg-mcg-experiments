"""Exact GLM route transforms around compact trellis projections."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from b12x.moe._shared.kernels.w4a16.kernel import (
    _run_trellis_dense_hadamard128,
)


@triton.jit
def _glm_route_silu_kernel(
    gate,
    up,
    route_experts,
    gate_svh,
    up_svh,
    output,
    width: tl.constexpr,
    block_n: tl.constexpr,
):
    route = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_n + tl.arange(0, block_n)
    mask = offsets < width
    expert = tl.load(route_experts + route).to(tl.int64)
    gate_value = tl.load(gate + route * width + offsets, mask=mask, other=0.0)
    up_value = tl.load(up + route * width + offsets, mask=mask, other=0.0)
    gate_scale = tl.load(
        gate_svh + expert * width + offsets, mask=mask, other=0.0
    )
    up_scale = tl.load(up_svh + expert * width + offsets, mask=mask, other=0.0)
    gate_value = gate_value.to(tl.float32) * gate_scale.to(tl.float32)
    up_value = up_value.to(tl.float32) * up_scale.to(tl.float32)
    activated = gate_value * tl.sigmoid(gate_value) * up_value
    tl.store(output + route * width + offsets, activated, mask=mask)


@triton.jit
def _glm_route_scale_kernel(
    source,
    route_experts,
    scale_table,
    output,
    width: tl.constexpr,
    block_n: tl.constexpr,
):
    route = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_n + tl.arange(0, block_n)
    mask = offsets < width
    expert = tl.load(route_experts + route).to(tl.int64)
    values = tl.load(source + route * width + offsets, mask=mask, other=0.0)
    scales = tl.load(
        scale_table + expert * width + offsets, mask=mask, other=0.0
    )
    tl.store(output + route * width + offsets, values * scales, mask=mask)


@triton.jit
def _glm_topk_weighted_sum_kernel(
    routes,
    topk_weights,
    output,
    width: tl.constexpr,
    topk: tl.constexpr,
    block_n: tl.constexpr,
):
    token = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_n + tl.arange(0, block_n)
    mask = offsets < width
    accumulator = tl.zeros((block_n,), dtype=tl.float32)
    for route_in_token in tl.static_range(topk):
        route = token * topk + route_in_token
        values = tl.load(
            routes + route * width + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        weight = tl.load(topk_weights + route).to(tl.float32)
        accumulator += values * weight
    tl.store(output + token * width + offsets, accumulator, mask=mask)


@triton.jit
def _glm_h512_finish_kernel(
    source_h128,
    output,
    width: tl.constexpr,
    block_n: tl.constexpr,
):
    """Finish normalized H512 as H4 over four normalized H128 chunks."""

    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_n + tl.arange(0, block_n)
    group_base = (offsets // 512) * 512
    within = offsets % 128
    mask = offsets < width
    row_base = row * width
    x0 = tl.load(
        source_h128 + row_base + group_base + within,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x1 = tl.load(
        source_h128 + row_base + group_base + 128 + within,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x2 = tl.load(
        source_h128 + row_base + group_base + 256 + within,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x3 = tl.load(
        source_h128 + row_base + group_base + 384 + within,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    quarter = (offsets % 512) // 128
    y0 = (x0 + x1 + x2 + x3) * 0.5
    y1 = (x0 - x1 + x2 - x3) * 0.5
    y2 = (x0 + x1 - x2 - x3) * 0.5
    y3 = (x0 - x1 - x2 + x3) * 0.5
    value = tl.where(
        quarter == 0,
        y0,
        tl.where(quarter == 1, y1, tl.where(quarter == 2, y2, y3)),
    )
    tl.store(output + row_base + offsets, value, mask=mask)


@triton.jit
def _glm_coupled_pack_scaled_kernel(
    gate,
    up,
    route_experts,
    gate_svh,
    up_svh,
    output,
    width: tl.constexpr,
    block_n: tl.constexpr,
):
    route = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_n + tl.arange(0, block_n)
    projection_offsets = offsets % width
    is_up = offsets >= width
    mask = offsets < 2 * width
    expert = tl.load(route_experts + route).to(tl.int64)
    gate_value = tl.load(
        gate + route * width + projection_offsets,
        mask=mask & ~is_up,
        other=0.0,
    ).to(tl.float32)
    up_value = tl.load(
        up + route * width + projection_offsets,
        mask=mask & is_up,
        other=0.0,
    ).to(tl.float32)
    gate_scale = tl.load(
        gate_svh + expert * width + projection_offsets,
        mask=mask & ~is_up,
        other=1.0,
    ).to(tl.float32)
    up_scale = tl.load(
        up_svh + expert * width + projection_offsets,
        mask=mask & is_up,
        other=1.0,
    ).to(tl.float32)
    value = tl.where(is_up, up_value * up_scale, gate_value * gate_scale)
    tl.store(output + route * (2 * width) + offsets, value, mask=mask)


@triton.jit
def _glm_coupled_silu_post_sign_kernel(
    recovered,
    route_experts,
    pre_signs,
    post_signs,
    output,
    width: tl.constexpr,
    block_n: tl.constexpr,
):
    route = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_n + tl.arange(0, block_n)
    mask = offsets < width
    expert = tl.load(route_experts + route).to(tl.int64)
    gate_index = 2 * offsets
    up_index = gate_index + 1
    recovered_base = route * (2 * width)
    signs_base = expert * (2 * width)
    gate = tl.load(
        recovered + recovered_base + gate_index, mask=mask, other=0.0
    ).to(tl.float32)
    up = tl.load(
        recovered + recovered_base + up_index, mask=mask, other=0.0
    ).to(tl.float32)
    gate *= tl.load(pre_signs + signs_base + gate_index, mask=mask, other=1.0)
    up *= tl.load(pre_signs + signs_base + up_index, mask=mask, other=1.0)
    post = tl.load(
        post_signs + expert * width + offsets, mask=mask, other=1.0
    ).to(tl.float32)
    activated = gate * tl.sigmoid(gate) * up * post
    tl.store(output + route * width + offsets, activated, mask=mask)


def run_glm_coupled_residual_h512(
    source: torch.Tensor,
    h128_scratch: torch.Tensor,
    output: torch.Tensor,
    *,
    ones: torch.Tensor,
) -> torch.Tensor:
    """Apply updated-QSRT's normalized block-H512 residual boundary."""

    if source.ndim != 2 or int(source.shape[1]) % 512:
        raise ValueError("coupled residual rows require width divisible by 512")
    rows, width = (int(value) for value in source.shape)
    device = source.device
    for name, value in (
        ("source", source),
        ("h128_scratch", h128_scratch),
        ("output", output),
    ):
        if (
            value.shape != source.shape
            or value.dtype != torch.float16
            or value.device != device
            or not value.is_contiguous()
        ):
            raise TypeError(
                f"{name} must be contiguous FP16 {tuple(source.shape)} on {device}"
            )
    if (
        ones.shape != (width,)
        or ones.dtype != torch.float16
        or ones.device != device
        or not ones.is_contiguous()
    ):
        raise TypeError(f"ones must be contiguous FP16 [{width}] on {device}")
    _run_trellis_dense_hadamard128(
        source, h128_scratch, ones, scale_before=False
    )
    block_n = 256
    _glm_h512_finish_kernel[(rows, triton.cdiv(width, block_n))](
        h128_scratch,
        output,
        width=width,
        block_n=block_n,
        num_warps=4,
    )
    return output


def run_glm_coupled_gate_up_output_transform_silu(
    gate_transformed: torch.Tensor,
    up_transformed: torch.Tensor,
    route_experts: torch.Tensor,
    gate_svh: torch.Tensor,
    up_svh: torch.Tensor,
    pre_signs: torch.Tensor,
    post_signs: torch.Tensor,
    gate_hadamard: torch.Tensor,
    up_hadamard: torch.Tensor,
    pre_scaled: torch.Tensor,
    pre_hadamard: torch.Tensor,
    activation_signed: torch.Tensor,
    output: torch.Tensor,
    *,
    ones_intermediate: torch.Tensor,
    ones_preactivation: torch.Tensor,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> torch.Tensor:
    """Close updated-QSRT's interleaved H128 SiLU activation boundary.

    The encoded full-width preactivation vector is stored as one gate half
    followed by one up half. Tensor parallelism slices both halves before the
    two local projections execute. A TP rank must therefore gather the local
    gate/up halves, restore the full encoded order, and select the contiguous
    interleaved interval that corresponds to its down-projection partition.
    """

    if gate_transformed.shape != up_transformed.shape or gate_transformed.ndim != 2:
        raise ValueError("coupled gate/up outputs must be aligned rank-2 tensors")
    routes, width = (int(value) for value in gate_transformed.shape)
    tp_rank = int(tp_rank)
    tp_size = int(tp_size)
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError("coupled TP rank must identify one tensor-parallel shard")
    device = gate_transformed.device
    for name, value, expected in (
        ("gate_hadamard", gate_hadamard, (routes, width)),
        ("up_hadamard", up_hadamard, (routes, width)),
        ("pre_scaled", pre_scaled, (routes, 2 * width)),
        ("pre_hadamard", pre_hadamard, (routes, 2 * width)),
        ("activation_signed", activation_signed, (routes, width)),
        ("output", output, (routes, width)),
    ):
        if (
            tuple(value.shape) != expected
            or value.dtype != torch.float16
            or value.device != device
            or not value.is_contiguous()
        ):
            raise TypeError(
                f"{name} must be contiguous FP16 {expected} on {device}"
            )
    if (
        pre_signs.ndim != 2
        or tuple(pre_signs.shape) != (int(gate_svh.shape[0]), 2 * width)
        or post_signs.ndim != 2
        or tuple(post_signs.shape) != (int(gate_svh.shape[0]), width)
        or pre_signs.dtype != torch.float16
        or post_signs.dtype != torch.float16
        or pre_signs.device != device
        or post_signs.device != device
    ):
        raise TypeError("coupled rotation signs differ from the expert geometry")
    _run_trellis_dense_hadamard128(
        gate_transformed,
        gate_hadamard,
        ones_intermediate,
        scale_before=False,
    )
    _run_trellis_dense_hadamard128(
        up_transformed,
        up_hadamard,
        ones_intermediate,
        scale_before=False,
    )
    block_n = 256
    _glm_coupled_pack_scaled_kernel[(routes, triton.cdiv(2 * width, block_n))](
        gate_hadamard,
        up_hadamard,
        route_experts,
        gate_svh,
        up_svh,
        pre_scaled,
        width=width,
        block_n=block_n,
        num_warps=4,
    )
    if tp_size > 1:
        from vllm.distributed import get_tp_group

        gathered = get_tp_group().all_gather(pre_scaled, dim=1)
        expected = (routes, 2 * width * tp_size)
        if tuple(gathered.shape) != expected:
            raise RuntimeError(
                "coupled TP gate/up gather differs: "
                f"{tuple(gathered.shape)} != {expected}"
            )
        full_encoded = (
            gathered.view(routes, tp_size, 2, width)
            .permute(0, 2, 1, 3)
            .reshape(routes, 2 * width * tp_size)
        )
        start = 2 * tp_rank * width
        pre_scaled.copy_(full_encoded[:, start : start + 2 * width])
    _run_trellis_dense_hadamard128(
        pre_scaled,
        pre_hadamard,
        ones_preactivation,
        scale_before=False,
    )
    _glm_coupled_silu_post_sign_kernel[(
        routes,
        triton.cdiv(width, block_n),
    )](
        pre_hadamard,
        route_experts,
        pre_signs,
        post_signs,
        activation_signed,
        width=width,
        block_n=block_n,
        num_warps=4,
    )
    _run_trellis_dense_hadamard128(
        activation_signed,
        output,
        ones_intermediate,
        scale_before=False,
    )
    return output


def run_glm_gate_up_output_transform_silu(
    gate_transformed: torch.Tensor,
    up_transformed: torch.Tensor,
    route_experts: torch.Tensor,
    gate_svh: torch.Tensor,
    up_svh: torch.Tensor,
    gate_hadamard: torch.Tensor,
    up_hadamard: torch.Tensor,
    output: torch.Tensor,
    *,
    ones: torch.Tensor,
) -> torch.Tensor:
    """Invert gate/up H128 bases, apply expert scales, then exact SwiGLU."""

    if gate_transformed.shape != up_transformed.shape or gate_transformed.ndim != 2:
        raise ValueError("gate/up transformed outputs must be aligned rank-2 tensors")
    routes, width = (int(value) for value in gate_transformed.shape)
    device = gate_transformed.device
    for name, tensor in (
        ("gate_transformed", gate_transformed),
        ("up_transformed", up_transformed),
        ("gate_hadamard", gate_hadamard),
        ("up_hadamard", up_hadamard),
        ("output", output),
    ):
        if (
            tensor.shape != gate_transformed.shape
            or tensor.dtype != torch.float16
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise TypeError(
                f"{name} must be contiguous FP16 {tuple(gate_transformed.shape)} "
                f"on {device}"
            )
    if (
        route_experts.shape != (routes,)
        or route_experts.dtype != torch.int32
        or route_experts.device != device
        or not route_experts.is_contiguous()
    ):
        raise TypeError("route_experts must be contiguous int32 [routes]")
    if gate_svh.shape != up_svh.shape or gate_svh.ndim != 2:
        raise ValueError("gate/up svh tables must be aligned [experts, intermediate]")
    for name, tensor in (("gate_svh", gate_svh), ("up_svh", up_svh)):
        if (
            int(tensor.shape[1]) != width
            or tensor.dtype != torch.float16
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise TypeError(f"{name} must be contiguous FP16 [experts,{width}]")
    if (
        ones.shape != (width,)
        or ones.dtype != torch.float16
        or ones.device != device
        or not ones.is_contiguous()
    ):
        raise TypeError(f"ones must be contiguous FP16 [{width}] on {device}")

    _run_trellis_dense_hadamard128(
        gate_transformed,
        gate_hadamard,
        ones,
        scale_before=False,
    )
    _run_trellis_dense_hadamard128(
        up_transformed,
        up_hadamard,
        ones,
        scale_before=False,
    )
    block_n = 256
    _glm_route_silu_kernel[(routes, triton.cdiv(width, block_n))](
        gate_hadamard,
        up_hadamard,
        route_experts,
        gate_svh,
        up_svh,
        output,
        width=width,
        block_n=block_n,
        num_warps=4,
    )
    return output


def run_glm_down_input_transform(
    activation: torch.Tensor,
    route_experts: torch.Tensor,
    down_suh: torch.Tensor,
    scaled: torch.Tensor,
    rotated: torch.Tensor,
    *,
    ones: torch.Tensor,
) -> torch.Tensor:
    """Apply expert-private down ``suh`` followed by normalized H128."""

    if activation.ndim != 2:
        raise ValueError("down activation must be rank 2")
    routes, width = (int(value) for value in activation.shape)
    device = activation.device
    for name, tensor in (
        ("activation", activation),
        ("scaled", scaled),
        ("rotated", rotated),
    ):
        if (
            tensor.shape != activation.shape
            or tensor.dtype != torch.float16
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise TypeError(
                f"{name} must be contiguous FP16 {tuple(activation.shape)} on {device}"
            )
    if (
        route_experts.shape != (routes,)
        or route_experts.dtype != torch.int32
        or route_experts.device != device
        or not route_experts.is_contiguous()
    ):
        raise TypeError("route_experts must be contiguous int32 [routes]")
    if (
        down_suh.ndim != 2
        or int(down_suh.shape[1]) != width
        or down_suh.dtype != torch.float16
        or down_suh.device != device
        or not down_suh.is_contiguous()
    ):
        raise TypeError(f"down_suh must be contiguous FP16 [experts,{width}]")
    if (
        ones.shape != (width,)
        or ones.dtype != torch.float16
        or ones.device != device
        or not ones.is_contiguous()
    ):
        raise TypeError(f"ones must be contiguous FP16 [{width}] on {device}")

    block_n = 256
    _glm_route_scale_kernel[(routes, triton.cdiv(width, block_n))](
        activation,
        route_experts,
        down_suh,
        scaled,
        width=width,
        block_n=block_n,
        num_warps=4,
    )
    _run_trellis_dense_hadamard128(
        scaled,
        rotated,
        ones,
        scale_before=True,
    )
    return rotated


def run_glm_down_output_transform_sum(
    down_transformed: torch.Tensor,
    topk_weights: torch.Tensor,
    down_svh: torch.Tensor,
    down_canonical: torch.Tensor,
    output: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    """Invert down H128, apply shared ``svh``, and sum signed top-k routes."""

    if down_transformed.ndim != 2:
        raise ValueError("down transformed output must be rank 2")
    routes, width = (int(value) for value in down_transformed.shape)
    topk = int(topk)
    if topk <= 0 or routes % topk:
        raise ValueError("route count must be positive and divisible by topk")
    tokens = routes // topk
    device = down_transformed.device
    if (
        down_transformed.dtype != torch.float16
        or not down_transformed.is_cuda
        or not down_transformed.is_contiguous()
        or down_canonical.shape != down_transformed.shape
        or down_canonical.dtype != torch.float16
        or down_canonical.device != device
        or not down_canonical.is_contiguous()
    ):
        raise TypeError("down buffers must be aligned contiguous CUDA FP16")
    if (
        down_svh.shape != (width,)
        or down_svh.dtype != torch.float16
        or down_svh.device != device
        or not down_svh.is_contiguous()
    ):
        raise TypeError(f"down_svh must be contiguous FP16 [{width}]")
    if (
        topk_weights.shape != (tokens, topk)
        or topk_weights.dtype != torch.float32
        or topk_weights.device != device
        or not topk_weights.is_contiguous()
    ):
        raise TypeError(f"topk_weights must be contiguous FP32 [{tokens},{topk}]")
    if (
        output.shape != (tokens, width)
        or output.dtype not in (torch.float16, torch.bfloat16)
        or output.device != device
        or not output.is_contiguous()
    ):
        raise TypeError(
            f"output must be contiguous FP16/BF16 [{tokens},{width}] on {device}"
        )

    _run_trellis_dense_hadamard128(
        down_transformed,
        down_canonical,
        down_svh,
        scale_before=False,
    )
    block_n = 256
    _glm_topk_weighted_sum_kernel[(tokens, triton.cdiv(width, block_n))](
        down_canonical,
        topk_weights,
        output,
        width=width,
        topk=topk,
        block_n=block_n,
        num_warps=4,
    )
    return output


__all__ = [
    "run_glm_coupled_gate_up_output_transform_silu",
    "run_glm_coupled_residual_h512",
    "run_glm_down_input_transform",
    "run_glm_down_output_transform_sum",
    "run_glm_gate_up_output_transform_silu",
]
