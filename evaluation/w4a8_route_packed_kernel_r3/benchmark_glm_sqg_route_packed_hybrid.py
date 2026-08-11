"""Test 8c-layer: route-packed GLM SQG hybrid versus dispatch-matched A16.

Both arms consume the same sealed atoms-v2 K3/K4 payload, captured top-8
routes, exact GLM SiLU, transforms, and routed sum.  The control executes all
three projections through compact W4A16.  The candidate changes only gate/up
to direct-E4M3 W4A8; down remains compact A16, matching the selected Test 8b
quality contract.
"""

from __future__ import annotations

import argparse
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

from benchmark_glm_sqg_route_packed_fc1 import (
    _BLOCK_M,
    _EXPERTS,
    _HIDDEN,
    _INTERMEDIATE,
    _LAYER,
    _TOPK,
    _buffers as _fc1_buffers,
    _load_inputs,
    _load_projection,
    _make_route_workspaces,
    _run_prefix,
    _sha256,
    _summary,
)
from b12x.gemm import trellis_linear
from b12x.moe._shared.kernels.glm_trellis_transform import (
    run_glm_down_input_transform,
    run_glm_down_output_transform_sum,
    run_glm_gate_up_output_transform_silu,
)
from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm
from b12x.moe._shared.kernels.glm_trellis_w4a8 import (
    prepare_glm_route_packed_w4a8_projection,
    run_glm_route_packed_w4a8_projection,
)
from b12x.moe._shared.kernels.glm_trellis_w4a16 import (
    prepare_glm_route_packed_w4a16_runtime,
    run_glm_route_packed_w4a16_projection,
)
from b12x.moe._shared.kernels.w4a16.kernel import (
    _run_trellis_dense_hadamard128,
)


_KIND = "b12x_glm52_sqg_route_packed_hybrid_layer_benchmark"


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


def _load_rates(manifest: dict[str, Any], projection: str) -> list[int]:
    bit_map = manifest["bit_map"]
    return [
        int(
            bit_map[
                f"model.layers.{_LAYER}.mlp.experts.{expert}.{projection}"
            ]
        )
        for expert in range(_EXPERTS)
    ]


def _load_down_projection(
    handle,
    manifest: dict[str, Any],
    device: torch.device,
):
    rates = _load_rates(manifest, "down_proj")
    if any(bits not in (3, 4) for bits in rates):
        raise ValueError("hybrid down path requires independent K3/K4 tensors")
    trellises = []
    down_suh = []
    for expert in range(_EXPERTS):
        prefix = f"model.layers.{_LAYER}.mlp.experts.{expert}.down_proj"
        trellises.append(handle.get_tensor(f"{prefix}.trellis").to(device))
        down_suh.append(handle.get_tensor(f"{prefix}.suh"))
    prepared = prepare_glm_route_packed_w4a8_projection(
        trellises,
        rates,
        size_k=_INTERMEDIATE,
        size_n=_HIDDEN,
    )
    suh_table = (
        torch.stack(down_suh).to(device=device, dtype=torch.float16).contiguous()
    )
    return prepared, suh_table, rates


def _load_topk_weights(
    capture_layer: Path,
    *,
    total_rows: int,
    m: int,
    device: torch.device,
) -> torch.Tensor:
    weights = np.memmap(
        capture_layer / "topk_weights.f32le.bin",
        mode="r",
        dtype="<f4",
        shape=(total_rows, _TOPK),
    )
    return (
        torch.from_numpy(np.array(weights[:m], copy=True))
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )


def _hybrid_buffers(m: int, device: torch.device) -> dict[str, Any]:
    result = _fc1_buffers(m, device)
    routes = m * _TOPK
    result.update(
        {
            "down_scaled": torch.empty(
                (routes, _INTERMEDIATE), dtype=torch.float16, device=device
            ),
            "down_rotated": torch.empty(
                (routes, _INTERMEDIATE), dtype=torch.float16, device=device
            ),
            "down_projected": torch.empty(
                (routes, _HIDDEN), dtype=torch.float16, device=device
            ),
            "down_canonical": torch.empty(
                (routes, _HIDDEN), dtype=torch.float16, device=device
            ),
            "output": torch.empty(
                (m, _HIDDEN), dtype=torch.bfloat16, device=device
            ),
            "down_quantized": empty_mxfp8_rows_for_dense_gemm(
                m * _TOPK, _INTERMEDIATE, device=device
            ),
        }
    )
    return result


def _run_down(
    buffers: dict[str, Any],
    route_experts: torch.Tensor,
    topk_weights: torch.Tensor,
    down,
    down_suh: torch.Tensor,
    down_svh: torch.Tensor,
    down_runtime,
) -> torch.Tensor:
    run_glm_down_input_transform(
        buffers["activated"],
        route_experts,
        down_suh,
        buffers["down_scaled"],
        buffers["down_rotated"],
        ones=buffers["ones"],
    )
    run_glm_route_packed_w4a16_projection(
        buffers["down_rotated"],
        route_experts,
        down,
        buffers["down_projected"],
        down_runtime,
    )
    return run_glm_down_output_transform_sum(
        buffers["down_projected"],
        topk_weights,
        down_svh,
        buffers["down_canonical"],
        buffers["output"],
        topk=_TOPK,
    )


def _run_down_a8(
    buffers: dict[str, Any],
    route_workspaces: dict[str, torch.Tensor],
    route_experts: torch.Tensor,
    topk_weights: torch.Tensor,
    down,
    down_suh: torch.Tensor,
    down_svh: torch.Tensor,
) -> torch.Tensor:
    """SPEED-ONLY full-W4A8 down arm: quantizes ``act`` to E4M3.

    Test 8b measured ~+19% routed-function damage from act-A8; this arm
    exists to price the speed side of that decision and is never a quality
    acceptance path.
    """
    run_glm_down_input_transform(
        buffers["activated"],
        route_experts,
        down_suh,
        buffers["down_scaled"],
        buffers["down_rotated"],
        ones=buffers["ones"],
    )
    quantized = buffers["down_quantized"]
    quantize_mxfp8_rows_cute(
        buffers["down_rotated"],
        quantized.values,
        quantized.scale_rows,
        quantized.scale_mma,
        value_order="trellis_native_mma",
    )
    run_glm_route_packed_w4a8_projection(
        quantized,
        down,
        route_workspaces["packed_route_indices"],
        route_workspaces["block_expert_ids"],
        buffers["down_projected"],
        topk=_TOPK,
        shared_input=False,
    )
    return run_glm_down_output_transform_sum(
        buffers["down_projected"],
        topk_weights,
        down_svh,
        buffers["down_canonical"],
        buffers["output"],
        topk=_TOPK,
    )


def _capture(callable_) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    eager = callable_().clone()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(eager).all()):
        raise RuntimeError("non-finite eager layer output")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = callable_()
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(captured, eager):
        raise RuntimeError("eager and CUDA-graph layer outputs differ")
    return graph, eager


def _time_abba(
    control: torch.cuda.CUDAGraph,
    hybrid: torch.cuda.CUDAGraph,
    *,
    samples: int,
) -> dict[str, list[float]]:
    if samples < 200 or samples % 2:
        raise ValueError("samples must be even and at least 200")
    graphs = {"a16": control, "hybrid": hybrid}
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    for cycle in range(samples // 2):
        order = (
            ("a16", "hybrid", "hybrid", "a16")
            if cycle % 2 == 0
            else ("hybrid", "a16", "a16", "hybrid")
        )
        for arm in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[arm].replay()
            end.record()
            events.append((arm, start, end))
    torch.cuda.synchronize()
    result = {"a16": [], "hybrid": []}
    for arm, start, end in events:
        result[arm].append(float(start.elapsed_time(end)) * 1000.0)
    return result


def _distortion(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate_f32 = candidate.float()
    reference_f32 = reference.float()
    delta = candidate_f32 - reference_f32
    denominator = max(float(reference_f32.square().sum()), 1.0e-30)
    cosine_denominator = max(
        float(torch.linalg.vector_norm(candidate_f32))
        * float(torch.linalg.vector_norm(reference_f32)),
        1.0e-30,
    )
    return {
        "nmse": float(delta.square().sum()) / denominator,
        "rmse": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "cosine_similarity": float((candidate_f32 * reference_f32).sum())
        / cosine_denominator,
    }


def _dense_down_oracle(
    handle,
    manifest: dict[str, Any],
    buffers: dict[str, Any],
    route_experts: torch.Tensor,
    down_svh: torch.Tensor,
) -> dict[str, Any]:
    rates = _load_rates(manifest, "down_proj")
    rows = []
    for bits in (3, 4):
        route = next(
            index
            for index, expert in enumerate(route_experts.detach().cpu().tolist())
            if rates[int(expert)] == bits
        )
        expert = int(route_experts[route])
        prefix = f"model.layers.{_LAYER}.mlp.experts.{expert}.down_proj"
        prepared = trellis_linear.prepare_weight(
            handle.get_tensor(f"{prefix}.trellis").to(route_experts.device),
            handle.get_tensor(f"{prefix}.suh").to(route_experts.device),
            down_svh,
            codebook="sqg_xor_cheb_t12",
            params_dtype=torch.float16,
        )
        source = buffers["activated"][route : route + 1]
        dense_buffers = {
            "output": torch.empty(
                (1, _HIDDEN), dtype=torch.float16, device=route_experts.device
            ),
            "gemm_output": torch.empty(
                (1, _HIDDEN), dtype=torch.float16, device=route_experts.device
            ),
            "c_tmp": torch.empty(
                64 * _HIDDEN, dtype=torch.float32, device=route_experts.device
            ),
            "rotated_f16": torch.empty_like(source),
        }
        dense = trellis_linear.run(source, prepared, **dense_buffers)[0].clone()
        packed = buffers["down_canonical"][route]
        torch.testing.assert_close(packed, dense, rtol=4.0e-3, atol=3.0e-2)
        rows.append(
            {
                "route": route,
                "expert": expert,
                "bits": bits,
                "distortion": _distortion(packed, dense),
            }
        )
    return {"passed": True, "rows": rows}


def _bootstrap_ratio(
    a16: list[float], hybrid: list[float], *, seed: int = 20260811
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    a16_array = np.asarray(a16)
    hybrid_array = np.asarray(hybrid)
    indices = generator.integers(0, len(a16), size=(10000, len(a16)))
    ratios = np.median(a16_array[indices], axis=1) / np.median(
        hybrid_array[indices], axis=1
    )
    low, high = np.quantile(ratios, (0.025, 0.975))
    return {
        "a16_over_hybrid": statistics.median(a16) / statistics.median(hybrid),
        "bootstrap_ci95_low": float(low),
        "bootstrap_ci95_high": float(high),
        "replicates": 10000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-shard", type=Path, required=True)
    parser.add_argument("--layer-manifest", type=Path, required=True)
    parser.add_argument("--capture-layer", type=Path, required=True)
    parser.add_argument("--m", type=int, choices=(3072, 4096), required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--moe-prefill-fraction", type=float, default=0.31)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.samples < 200 or args.samples % 2 or args.warmup < 1:
        raise SystemExit("samples must be even >=200 and warmup positive")
    if not 0.0 <= args.moe_prefill_fraction <= 1.0:
        raise SystemExit("MoE fraction must lie in [0,1]")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major != 12 or minor not in (0, 1):
        raise SystemExit(f"SM120/SM121 required, got SM{major}{minor}")

    manifest = json.loads(args.layer_manifest.read_text())
    if manifest.get("bit_histogram") != {"3": 384, "4": 384}:
        raise SystemExit("layer is not the sealed mixed K3/K4 atoms-v2 artifact")
    device = torch.device("cuda", torch.cuda.current_device())
    source, topk_ids, capture_manifest = _load_inputs(
        args.capture_layer, m=args.m, device=device
    )
    topk_weights = _load_topk_weights(
        args.capture_layer,
        total_rows=int(capture_manifest["tokens"]),
        m=args.m,
        device=device,
    )
    route_experts = topk_ids.reshape(-1)
    routes = args.m * _TOPK
    print(f"loaded captured M={args.m}, routes={routes}", flush=True)

    with safe_open(args.layer_shard, framework="pt", device="cpu") as handle:
        gate, gate_svh, gate_rates = _load_projection(
            handle, manifest, "gate_proj", device
        )
        up, up_svh, up_rates = _load_projection(
            handle, manifest, "up_proj", device
        )
        down, down_suh, down_rates = _load_down_projection(
            handle, manifest, device
        )
        gate_up_suh = handle.get_tensor(
            f"model.layers.{_LAYER}.mlp.experts.r7_shared.gate_up_suh"
        ).to(device=device, dtype=torch.float16).contiguous()
        down_svh = handle.get_tensor(
            f"model.layers.{_LAYER}.mlp.experts.r7_shared.down_svh"
        ).to(device=device, dtype=torch.float16).contiguous()
        print("loaded all three sealed independent-rate projections", flush=True)

        hybrid_route_workspaces = _make_route_workspaces(args.m, device)
        buffers = _hybrid_buffers(args.m, device)
        gate_a16_runtime = prepare_glm_route_packed_w4a16_runtime(
            gate, routes=routes, topk=_TOPK
        )
        print("compiled gate A16 control", flush=True)
        up_a16_runtime = prepare_glm_route_packed_w4a16_runtime(
            up, routes=routes, topk=_TOPK
        )
        print("compiled up A16 control", flush=True)
        down_runtime = prepare_glm_route_packed_w4a16_runtime(
            down, routes=routes, topk=1
        )
        print("compiled compact A16 down", flush=True)

        def run_hybrid() -> torch.Tensor:
            _run_prefix(
                source,
                topk_ids,
                hybrid_route_workspaces,
                buffers,
                gate,
                up,
                gate_up_suh,
                gate_svh,
                up_svh,
            )
            return _run_down(
                buffers,
                route_experts,
                topk_weights,
                down,
                down_suh,
                down_svh,
                down_runtime,
            )

        def run_a16() -> torch.Tensor:
            _run_trellis_dense_hadamard128(
                source,
                buffers["rotated"],
                gate_up_suh,
                scale_before=True,
            )
            run_glm_route_packed_w4a16_projection(
                buffers["rotated"],
                route_experts,
                gate,
                buffers["gate"],
                gate_a16_runtime,
            )
            run_glm_route_packed_w4a16_projection(
                buffers["rotated"],
                route_experts,
                up,
                buffers["up"],
                up_a16_runtime,
            )
            run_glm_gate_up_output_transform_silu(
                buffers["gate"],
                buffers["up"],
                route_experts,
                gate_svh,
                up_svh,
                buffers["gate_h"],
                buffers["up_h"],
                buffers["activated"],
                ones=buffers["ones"],
            )
            return _run_down(
                buffers,
                route_experts,
                topk_weights,
                down,
                down_suh,
                down_svh,
                down_runtime,
            )

        down_a8_enabled = os.environ.get("GLM_HYBRID_DOWN_A8", "") == "1"

        def run_full_w4a8() -> torch.Tensor:
            _run_prefix(
                source,
                topk_ids,
                hybrid_route_workspaces,
                buffers,
                gate,
                up,
                gate_up_suh,
                gate_svh,
                up_svh,
            )
            return _run_down_a8(
                buffers,
                hybrid_route_workspaces,
                route_experts,
                topk_weights,
                down,
                down_suh,
                down_svh,
            )

        control_graph, control_output = _capture(run_a16)
        hybrid_graph, hybrid_output = _capture(run_hybrid)
        full_graph = None
        full_output = None
        if down_a8_enabled:
            full_graph, full_output = _capture(run_full_w4a8)
        for _ in range(args.warmup):
            control_graph.replay()
            hybrid_graph.replay()
            if full_graph is not None:
                full_graph.replay()
        torch.cuda.synchronize()

        run_hybrid()
        torch.cuda.synchronize()
        dense_oracle = _dense_down_oracle(
            handle, manifest, buffers, route_experts, down_svh
        )
        token_sample = min(args.m, 32)
        signed_reference = (
            buffers["down_canonical"][: token_sample * _TOPK]
            .float()
            .reshape(token_sample, _TOPK, _HIDDEN)
            .mul(topk_weights[:token_sample, :, None])
            .sum(dim=1)
            .to(torch.bfloat16)
        )
        torch.testing.assert_close(
            buffers["output"][:token_sample],
            signed_reference,
            rtol=3.0e-3,
            atol=3.0e-2,
        )
        print("dense down and signed top-8 closure passed", flush=True)
        timings = _time_abba(
            control_graph, hybrid_graph, samples=args.samples
        )
        full_timings = None
        full_distortion = None
        if full_graph is not None:
            full_timings = _time_abba(
                control_graph, full_graph, samples=args.samples
            )
            full_graph.replay()
            torch.cuda.synchronize()
            full_distortion = _distortion(
                full_output.float(), control_output.float()
            )

    ratio = _bootstrap_ratio(timings["a16"], timings["hybrid"])
    moe_speedup = ratio["a16_over_hybrid"]
    end_to_end = 1.0 / (
        (1.0 - args.moe_prefill_fraction)
        + args.moe_prefill_fraction / moe_speedup
    )
    worktree = Path(__file__).resolve().parents[1]
    props = torch.cuda.get_device_properties(device)
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
            "routes": routes,
            "topk": _TOPK,
            "hidden": _HIDDEN,
            "intermediate": _INTERMEDIATE,
            "experts": _EXPERTS,
            "route_block": _BLOCK_M,
            "control": "route-packed compact SQG W4A16 gate/up/down",
            "candidate": "route-packed SQG W4A8 gate/up plus compact SQG W4A16 down",
            "activation": "exact silu(gate) * up",
            "router_sum": "FP32 signed top-8 weighted sum to BF16",
            "rates": "independent per tensor K3/K4; no uniform K3",
            "transforms": "topology-neutral shared/private suh/svh and H128",
            "dense_weight_materialization": False,
            "caller_owned_fixed_workspace": True,
            "cuda_graph_replay": True,
            "timing": "balanced ABBA with alternating orientation",
        },
        "rate_census": {
            projection: {
                str(bits): rates.count(bits) for bits in (3, 4)
            }
            for projection, rates in (
                ("gate", gate_rates),
                ("up", up_rates),
                ("down", down_rates),
            )
        },
        "validation": {
            "a16_eager_graph_bit_exact": True,
            "hybrid_eager_graph_bit_exact": True,
            "all_finite": True,
            "dense_down_oracle": dense_oracle,
            "signed_top8_sample_tokens": token_sample,
            "signed_top8_sample_passed": True,
            "hybrid_vs_a16_layer_output": _distortion(
                hybrid_output, control_output
            ),
        },
        "timing": {
            "samples_per_arm": args.samples,
            "warmup": args.warmup,
            "a16_control": _summary(timings["a16"]),
            "hybrid": _summary(timings["hybrid"]),
            "speedup": ratio,
            "moe_prefill_fraction": args.moe_prefill_fraction,
            "amdahl_projected_end_to_end_prefill_speedup": end_to_end,
        },
        "full_w4a8_speed_only_arm": None
        if full_timings is None
        else {
            "quality_status": (
                "NOT quality-qualified: act-A8 measured ~+19% routed damage "
                "in Test 8b; this arm prices speed only"
            ),
            "a16_control": _summary(full_timings["a16"]),
            "full_w4a8": _summary(full_timings["hybrid"]),
            "speedup": _bootstrap_ratio(
                full_timings["a16"], full_timings["hybrid"]
            ),
            "amdahl_projected_end_to_end_prefill_speedup": 1.0
            / (
                (1.0 - args.moe_prefill_fraction)
                + args.moe_prefill_fraction
                / (
                    statistics.median(full_timings["a16"])
                    / statistics.median(full_timings["hybrid"])
                )
            ),
            "full_w4a8_vs_a16_layer_output": full_distortion,
        },
        "gate": {
            "quality": "report relative to dispatch-matched A16; KLD remains final acceptance",
            "speed_threshold": 1.15,
            "projected_end_to_end_pass": end_to_end >= 1.15,
        },
        "device": {
            "name": props.name,
            "sm_count": props.multi_processor_count,
            "capability": [major, minor],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload["timing"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
