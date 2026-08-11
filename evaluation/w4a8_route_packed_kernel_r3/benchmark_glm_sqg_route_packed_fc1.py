"""Measure the real mixed-rate route-packed GLM SQG W4A8 FC1 path.

The benchmark consumes one sealed atoms-v2 layer and captured GLM routes.  It
preserves the independent K3/K4 assignment of every gate and up tensor and
times the complete hybrid FC1 prefix:

    route pack -> shared suh/H128 -> MXFP8 A8 -> packed gate/up FP8 MMA
    -> expert-private svh/H128 -> exact SiLU(gate) * up

The result is deliberately labelled an FC1 prefix, not a complete MoE layer:
the selected quality contract keeps the down projection on A16, whose packed
projection and routed sum are a separate integration stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Any

import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F

from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x.gemm import trellis_linear
from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm
from b12x.moe._shared.kernels.glm_trellis_transform import (
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


_KIND = "b12x_glm52_sqg_route_packed_w4a8_fc1"
_LAYER = 77
_HIDDEN = 6144
_INTERMEDIATE = 2048
_EXPERTS = 256
_TOPK = 8
_BLOCK_M = 128


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[int(0.10 * (len(ordered) - 1))],
        "p90_us": ordered[int(0.90 * (len(ordered) - 1))],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _time_graph(graph: torch.cuda.CUDAGraph, samples: int) -> list[float]:
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) * 1000.0 for start, end in events]


def _load_rates(manifest: dict[str, Any], projection: str) -> list[int]:
    bit_map = manifest.get("bit_map")
    if not isinstance(bit_map, dict):
        raise ValueError("layer manifest has no bit_map")
    rates = []
    for expert in range(_EXPERTS):
        key = f"model.layers.{_LAYER}.mlp.experts.{expert}.{projection}"
        bits = int(bit_map[key])
        if bits not in (3, 4):
            raise ValueError(f"{key} is K{bits}; this path supports K3/K4")
        rates.append(bits)
    return rates


def _load_projection(
    handle,
    manifest: dict[str, Any],
    projection: str,
    device: torch.device,
) -> tuple[GLMRoutePackedW4A8Projection, torch.Tensor, list[int]]:
    rates = _load_rates(manifest, projection)
    trellises: list[torch.Tensor] = []
    svh: list[torch.Tensor] = []
    for expert in range(_EXPERTS):
        prefix = f"model.layers.{_LAYER}.mlp.experts.{expert}.{projection}"
        trellises.append(handle.get_tensor(f"{prefix}.trellis").to(device))
        svh.append(handle.get_tensor(f"{prefix}.svh"))
    prepared = prepare_glm_route_packed_w4a8_projection(
        trellises,
        rates,
        size_k=_HIDDEN,
        size_n=_INTERMEDIATE,
    )
    svh_table = torch.stack(svh).to(device=device, dtype=torch.float16).contiguous()
    return prepared, svh_table, rates


def _load_inputs(
    capture_layer: Path,
    *,
    m: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    capture_manifest_path = capture_layer / "layer_manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text())
    if (
        int(capture_manifest["layer"]) != _LAYER
        or int(capture_manifest["hidden"]) != _HIDDEN
        or int(capture_manifest["topk"]) != _TOPK
        or int(capture_manifest["tokens"]) < m
    ):
        raise ValueError("capture does not satisfy the sealed GLM layer-77 contract")
    total = int(capture_manifest["tokens"])
    hidden = np.memmap(
        capture_layer / "hidden.bf16.bin",
        mode="r",
        dtype="u2",
        shape=(total, _HIDDEN),
    )
    ids = np.memmap(
        capture_layer / "topk_ids.u8.bin",
        mode="r",
        dtype="u1",
        shape=(total, _TOPK),
    )
    source_words = np.array(hidden[:m], copy=True)
    source = (
        torch.from_numpy(source_words)
        .view(torch.bfloat16)
        .to(device=device, dtype=torch.float16)
        .contiguous()
    )
    topk_ids = (
        torch.from_numpy(np.array(ids[:m], copy=True))
        .to(device=device, dtype=torch.int32)
        .contiguous()
    )
    return source, topk_ids, capture_manifest


def _make_route_workspaces(
    m: int, device: torch.device
) -> dict[str, torch.Tensor]:
    numel_capacity, packed_capacity, block_capacity = route_pack_capacity(
        m * _TOPK,
        _BLOCK_M,
        _EXPERTS,
        topk=_TOPK,
    )
    return {
        "numel_capacity": torch.tensor(numel_capacity),
        "packed_route_indices": torch.empty(
            packed_capacity, dtype=torch.int32, device=device
        ),
        "block_expert_ids": torch.empty(
            block_capacity, dtype=torch.int32, device=device
        ),
        "packed_route_count": torch.empty(1, dtype=torch.int32, device=device),
        "expert_offsets": torch.empty(
            _EXPERTS + 1, dtype=torch.int32, device=device
        ),
        "expert_counts": torch.empty(_EXPERTS, dtype=torch.int32, device=device),
    }


def _pack(topk_ids: torch.Tensor, workspaces: dict[str, torch.Tensor]):
    return pack_topk_routes_by_expert(
        topk_ids,
        _BLOCK_M,
        _EXPERTS,
        packed_route_indices=workspaces["packed_route_indices"],
        block_expert_ids=workspaces["block_expert_ids"],
        packed_route_count=workspaces["packed_route_count"],
        expert_offsets=workspaces["expert_offsets"],
        expert_counts=workspaces["expert_counts"],
    )


def _buffers(m: int, device: torch.device) -> dict[str, Any]:
    routes = m * _TOPK
    return {
        "rotated": torch.empty((m, _HIDDEN), dtype=torch.float16, device=device),
        "quantized": empty_mxfp8_rows_for_dense_gemm(m, _HIDDEN, device=device),
        "gate": torch.empty(
            (routes, _INTERMEDIATE), dtype=torch.float16, device=device
        ),
        "up": torch.empty(
            (routes, _INTERMEDIATE), dtype=torch.float16, device=device
        ),
        "gate_h": torch.empty(
            (routes, _INTERMEDIATE), dtype=torch.float16, device=device
        ),
        "up_h": torch.empty(
            (routes, _INTERMEDIATE), dtype=torch.float16, device=device
        ),
        "activated": torch.empty(
            (routes, _INTERMEDIATE), dtype=torch.float16, device=device
        ),
        "ones": torch.ones(_INTERMEDIATE, dtype=torch.float16, device=device),
    }


def _run_prefix(
    source: torch.Tensor,
    topk_ids: torch.Tensor,
    route_workspaces: dict[str, torch.Tensor],
    buffers: dict[str, Any],
    gate: GLMRoutePackedW4A8Projection,
    up: GLMRoutePackedW4A8Projection,
    gate_up_suh: torch.Tensor,
    gate_svh: torch.Tensor,
    up_svh: torch.Tensor,
) -> torch.Tensor:
    packed, block_experts, _ = _pack(topk_ids, route_workspaces)
    _run_trellis_dense_hadamard128(
        source,
        buffers["rotated"],
        gate_up_suh,
        scale_before=True,
    )
    quantized = buffers["quantized"]
    quantize_mxfp8_rows_cute(
        buffers["rotated"],
        quantized.values,
        quantized.scale_rows,
        quantized.scale_mma,
        value_order="trellis_native_mma",
    )
    run_glm_route_packed_w4a8_projection(
        quantized,
        gate,
        packed,
        block_experts,
        buffers["gate"],
        topk=_TOPK,
        shared_input=True,
    )
    run_glm_route_packed_w4a8_projection(
        quantized,
        up,
        packed,
        block_experts,
        buffers["up"],
        topk=_TOPK,
        shared_input=True,
    )
    return run_glm_gate_up_output_transform_silu(
        buffers["gate"],
        buffers["up"],
        topk_ids.reshape(-1),
        gate_svh,
        up_svh,
        buffers["gate_h"],
        buffers["up_h"],
        buffers["activated"],
        ones=buffers["ones"],
    )


def _capture_graph(callable_) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    eager = callable_().clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = callable_()
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(captured, eager):
        raise RuntimeError("eager and CUDA-graph results differ")
    return graph, eager


def _dense_buffers(source: torch.Tensor) -> dict[str, Any]:
    m = int(source.shape[0])
    return {
        "output": torch.empty(
            (m, _INTERMEDIATE), dtype=torch.float16, device=source.device
        ),
        "rotated_f16": torch.empty_like(source),
        "quantized": empty_mxfp8_rows_for_dense_gemm(
            m, _HIDDEN, device=source.device
        ),
        "gemm_output_f16": torch.empty(
            (m, _INTERMEDIATE), dtype=torch.float16, device=source.device
        ),
        "m_tile_rows": 64,
    }


def _validate_against_dense(
    handle,
    manifest: dict[str, Any],
    source: torch.Tensor,
    topk_ids: torch.Tensor,
    buffers: dict[str, Any],
    gate_up_suh: torch.Tensor,
) -> dict[str, Any]:
    flat_ids = topk_ids.reshape(-1)
    rate_pairs: dict[tuple[int, int], int] = {}
    gate_rates = _load_rates(manifest, "gate_proj")
    up_rates = _load_rates(manifest, "up_proj")
    for route, expert_tensor in enumerate(flat_ids.detach().cpu().tolist()):
        expert = int(expert_tensor)
        rate_pairs.setdefault((gate_rates[expert], up_rates[expert]), route)
        if len(rate_pairs) == 4:
            break
    rows: list[dict[str, Any]] = []
    for (gate_bits, up_bits), route in sorted(rate_pairs.items()):
        expert = int(flat_ids[route])
        token = route // _TOPK
        dense_outputs = {}
        for projection in ("gate_proj", "up_proj"):
            prefix = f"model.layers.{_LAYER}.mlp.experts.{expert}.{projection}"
            prepared = trellis_linear.prepare_weight(
                handle.get_tensor(f"{prefix}.trellis").to(source.device),
                gate_up_suh,
                handle.get_tensor(f"{prefix}.svh").to(source.device),
                codebook="sqg_xor_cheb_t12",
                params_dtype=torch.float16,
            )
            one = source[token : token + 1]
            dense_outputs[projection] = trellis_linear.run_uniform_w4a8(
                one, prepared, **_dense_buffers(one)
            )[0].clone()
        packed_gate = (
            buffers["gate_h"][route].float()
            * handle.get_tensor(
                f"model.layers.{_LAYER}.mlp.experts.{expert}.gate_proj.svh"
            ).to(source.device).float()
        ).half()
        packed_up = (
            buffers["up_h"][route].float()
            * handle.get_tensor(
                f"model.layers.{_LAYER}.mlp.experts.{expert}.up_proj.svh"
            ).to(source.device).float()
        ).half()
        dense_activation = (
            F.silu(dense_outputs["gate_proj"].float())
            * dense_outputs["up_proj"].float()
        ).half()
        packed_activation = buffers["activated"][route]
        comparisons = {}
        for name, candidate, reference in (
            ("gate", packed_gate, dense_outputs["gate_proj"]),
            ("up", packed_up, dense_outputs["up_proj"]),
            ("activation", packed_activation, dense_activation),
        ):
            delta = candidate.float() - reference.float()
            denom = max(float(reference.float().square().sum()), 1.0e-30)
            comparisons[name] = {
                "nmse": float(delta.square().sum()) / denom,
                "max_abs": float(delta.abs().max()),
            }
            torch.testing.assert_close(
                candidate, reference, rtol=4.0e-3, atol=3.0e-2
            )
        rows.append(
            {
                "route": route,
                "token": token,
                "expert": expert,
                "gate_bits": gate_bits,
                "up_bits": up_bits,
                "comparisons": comparisons,
            }
        )
    return {"rate_pairs_found": len(rate_pairs), "rows": rows, "passed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-shard", type=Path, required=True)
    parser.add_argument("--layer-manifest", type=Path, required=True)
    parser.add_argument("--capture-layer", type=Path, required=True)
    parser.add_argument("--m", type=int, choices=(3072, 4096), required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.samples < 1 or args.warmup < 1:
        raise SystemExit("samples and warmup must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major != 12 or minor not in (0, 1):
        raise SystemExit(f"SM120/SM121 required, got SM{major}{minor}")

    manifest = json.loads(args.layer_manifest.read_text())
    if manifest.get("bit_histogram") != {"3": 384, "4": 384}:
        raise SystemExit("sealed layer does not preserve the 384/384 K3/K4 census")
    device = torch.device("cuda", torch.cuda.current_device())
    source, topk_ids, capture_manifest = _load_inputs(
        args.capture_layer, m=args.m, device=device
    )
    print(f"loaded M={args.m} captured inputs and routes", flush=True)

    with safe_open(args.layer_shard, framework="pt", device="cpu") as handle:
        gate, gate_svh, gate_rates = _load_projection(
            handle, manifest, "gate_proj", device
        )
        print("loaded sealed gate K3/K4 pools", flush=True)
        up, up_svh, up_rates = _load_projection(
            handle, manifest, "up_proj", device
        )
        print("loaded sealed up K3/K4 pools", flush=True)
        gate_up_suh = handle.get_tensor(
            f"model.layers.{_LAYER}.mlp.experts.r7_shared.gate_up_suh"
        ).to(device=device, dtype=torch.float16).contiguous()
        route_workspaces = _make_route_workspaces(args.m, device)
        buffers = _buffers(args.m, device)

        call = lambda: _run_prefix(  # noqa: E731
            source,
            topk_ids,
            route_workspaces,
            buffers,
            gate,
            up,
            gate_up_suh,
            gate_svh,
            up_svh,
        )
        graph, eager = _capture_graph(call)
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize()

        def prep_call() -> torch.Tensor:
            _pack(topk_ids, route_workspaces)
            _run_trellis_dense_hadamard128(
                source,
                buffers["rotated"],
                gate_up_suh,
                scale_before=True,
            )
            quantized = buffers["quantized"]
            quantize_mxfp8_rows_cute(
                buffers["rotated"],
                quantized.values,
                quantized.scale_rows,
                quantized.scale_mma,
                value_order="trellis_native_mma",
            )
            return buffers["rotated"]

        packed, block_experts, _ = _pack(topk_ids, route_workspaces)
        prep_call()

        def projection_call() -> torch.Tensor:
            run_glm_route_packed_w4a8_projection(
                buffers["quantized"],
                gate,
                packed,
                block_experts,
                buffers["gate"],
                topk=_TOPK,
                shared_input=True,
            )
            return run_glm_route_packed_w4a8_projection(
                buffers["quantized"],
                up,
                packed,
                block_experts,
                buffers["up"],
                topk=_TOPK,
                shared_input=True,
            )

        def post_call() -> torch.Tensor:
            return run_glm_gate_up_output_transform_silu(
                buffers["gate"],
                buffers["up"],
                topk_ids.reshape(-1),
                gate_svh,
                up_svh,
                buffers["gate_h"],
                buffers["up_h"],
                buffers["activated"],
                ones=buffers["ones"],
            )

        prep_graph, _ = _capture_graph(prep_call)
        projection_graph, _ = _capture_graph(projection_call)
        post_graph, _ = _capture_graph(post_call)
        for _ in range(args.warmup):
            prep_graph.replay()
            projection_graph.replay()
            post_graph.replay()
        torch.cuda.synchronize()
        validation = _validate_against_dense(
            handle,
            manifest,
            source,
            topk_ids,
            buffers,
            gate_up_suh,
        )
        print("dense oracle validation passed", flush=True)
        samples = _time_graph(graph, args.samples)
        prep_samples = _time_graph(prep_graph, args.samples)
        projection_samples = _time_graph(projection_graph, args.samples)
        post_samples = _time_graph(post_graph, args.samples)

    worktree = Path(__file__).resolve().parents[1]
    props = torch.cuda.get_device_properties(device)
    route_counts = torch.bincount(
        topk_ids.reshape(-1).long(), minlength=_EXPERTS
    ).cpu()
    payload = {
        "kind": _KIND,
        "schema_version": 1,
        "status": "complete",
        "provenance": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "worktree": str(worktree),
            "git_commit": _git_commit(worktree),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "layer_shard": str(args.layer_shard.resolve()),
            "layer_shard_sha256_from_manifest": manifest.get("shard_sha256"),
            "layer_manifest": str(args.layer_manifest.resolve()),
            "layer_manifest_sha256": _sha256(args.layer_manifest),
            "capture_layer": str(args.capture_layer.resolve()),
            "capture_run_uuid": capture_manifest.get("capture_run_uuid"),
            "w4a8_kernel": os.environ.get("B12X_GLM_W4A8_KERNEL", "m64n256"),
            "w4a8_v2_blocks": os.environ.get("B12X_GLM_W4A8_V2_BLOCKS", "4"),
        },
        "contract": {
            "layer": _LAYER,
            "m": args.m,
            "routes": args.m * _TOPK,
            "topk": _TOPK,
            "num_experts": _EXPERTS,
            "hidden": _HIDDEN,
            "intermediate": _INTERMEDIATE,
            "route_block": _BLOCK_M,
            "activation": "exact silu(gate) * up",
            "weight_endpoint": "native SQG labels decoded directly to E4M3 MMA",
            "activation_endpoint": "MXFP8 E4M3 rows with per-K32 E8M0 scales",
            "input_transform": "shared gate_up_suh then normalized H128",
            "output_transform": "normalized H128 then expert-private gate/up svh",
            "independent_per_tensor_rates": True,
            "uniform_k3": False,
            "dense_weight_materialization": False,
            "caller_owned_fixed_workspace": True,
            "cuda_graph_replay": True,
            "scope": "route-packed W4A8 gate/up FC1 plus exact SwiGLU prefix",
            "excluded": [
                "down A16 projection",
                "router-weighted expert sum",
                "residual and non-MoE layer work",
            ],
        },
        "rate_census": {
            "gate": {str(bits): gate_rates.count(bits) for bits in (3, 4)},
            "up": {str(bits): up_rates.count(bits) for bits in (3, 4)},
        },
        "route_distribution": {
            "active_experts": int((route_counts > 0).sum()),
            "min": int(route_counts.min()),
            "median": float(route_counts.float().median()),
            "max": int(route_counts.max()),
        },
        "validation": {
            "eager_graph_bit_exact": bool(torch.equal(buffers["activated"], eager)),
            "all_finite": bool(torch.isfinite(eager).all()),
            "dense_oracle": validation,
        },
        "timing": {
            "samples": args.samples,
            "warmup": args.warmup,
            "cuda_graph_full_prefix": _summary(samples),
            "component_graphs": {
                "route_pack_input_transform_mxfp8": _summary(prep_samples),
                "mixed_rate_gate_up_projection": _summary(projection_samples),
                "output_transform_exact_swiglu": _summary(post_samples),
            },
        },
        "device": {
            "name": props.name,
            "sm_count": props.multi_processor_count,
            "capability": [major, minor],
        },
        "interpretation": {
            "serving_acceptance": False,
            "reason": (
                "this closes and times the actual route-packed mixed-rate hybrid "
                "FC1 prefix; the independently rated compact A16 down projection "
                "and routed sum must still be integrated before a layer speed gate"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload["timing"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
