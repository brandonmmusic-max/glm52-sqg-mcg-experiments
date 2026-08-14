#!/usr/bin/env python3
"""Routed W4A8 ground truth: runtime expert FFN vs ORIGINAL BF16 weights.

The reference *decoder* in glm52_fresh_sqg cannot be trusted here (it misses
the original BF16 weight by ~161% on a K6 tensor the runtime reproduces to
4%), so the only sound oracle is the published BF16 checkpoint itself.

Runs the real route-packed GLM SQG W4A8 layer with ONE expert (tp_size=1, so
the whole logical tensor is live) and compares against the BF16 expert FFN
computed in fp32:  silu(x @ Wg) * (x @ Wu) @ Wd.

Expected: rel-RMSE in the few-percent range for K3/K4 experts plus MXFP8
activation quantization. A layout/epilogue defect lands near 100%.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

MODEL = Path("/model")
BF16 = Path("/bf16/model-00075-of-00282.safetensors")
LAYER = 3
EXPERT = 0


def load(weight_map: dict, name: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(str(MODEL / weight_map[name]), framework="pt", device="cpu") as h:
        return h.get_tensor(name)


def rel_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    denominator = expected.float().square().mean().sqrt().clamp_min(1e-20)
    return float(difference.square().mean().sqrt() / denominator)


def main() -> None:
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    quant = json.loads((MODEL / "quantization_config.json").read_text())
    base = f"model.layers.{LAYER}.mlp.experts"

    gate_t = load(weight_map, f"{base}.{EXPERT}.gate_proj.trellis")
    up_t = load(weight_map, f"{base}.{EXPERT}.up_proj.trellis")
    down_t = load(weight_map, f"{base}.{EXPERT}.down_proj.trellis")
    shared_suh = load(weight_map, f"{base}.r7_shared.gate_up_suh")
    shared_down_svh = load(weight_map, f"{base}.r7_shared.down_svh")
    gate_svh = load(weight_map, f"{base}.{EXPERT}.gate_proj.svh")
    up_svh = load(weight_map, f"{base}.{EXPERT}.up_proj.svh")
    down_suh = load(weight_map, f"{base}.{EXPERT}.down_proj.suh")
    bits = {
        "gate": gate_t.shape[2] // 16,
        "up": up_t.shape[2] // 16,
        "down": down_t.shape[2] // 16,
    }

    from safetensors import safe_open

    with safe_open(str(BF16), framework="pt", device="cpu") as h:
        w_gate = h.get_tensor(f"{base}.{EXPERT}.gate_proj.weight").t().float()
        w_up = h.get_tensor(f"{base}.{EXPERT}.up_proj.weight").t().float()
        w_down = h.get_tensor(f"{base}.{EXPERT}.down_proj.weight").t().float()
    hidden, intermediate = w_gate.shape
    print(f"layer {LAYER} expert {EXPERT} bits {bits} hidden {hidden} inter {intermediate}")

    tokens = 32
    generator = torch.Generator().manual_seed(20260812)
    x = (torch.randn((tokens, hidden), generator=generator) * 0.4).to(torch.bfloat16)
    xf = x.float()
    y_ref = (torch.nn.functional.silu(xf @ w_gate) * (xf @ w_up)) @ w_down

    from b12x.moe import glm_sqg_w4a8 as api

    api.validate_glm_route_packed_w4a8_acceptance_kernel()
    target = quant["glm_sqg_w4a8"]["down_targets"][str(LAYER)]
    weights = api.prepare_weights(
        gate_trellis=[gate_t.cuda()], gate_bits=[bits["gate"]],
        up_trellis=[up_t.cuda()], up_bits=[bits["up"]],
        down_trellis=[down_t.cuda()], down_bits=[bits["down"]],
        gate_up_suh=shared_suh.cuda(),
        gate_svh=gate_svh.unsqueeze(0).contiguous().cuda(),
        up_svh=up_svh.unsqueeze(0).contiguous().cuda(),
        down_suh=down_suh.unsqueeze(0).contiguous().cuda(),
        down_svh=shared_down_svh.cuda(),
        hidden_size=hidden,
        intermediate_size=intermediate,
        global_intermediate_size=intermediate,
        tp_rank=0, tp_size=1,
        derived_down_target_id=target["derived_down_target_id"],
        down_target_beta=target["down_target_beta"],
    )
    runtime = api.prepare_runtime(
        weights, max_tokens=tokens, topk=1, output_dtype=torch.bfloat16
    )
    topk_ids = torch.zeros((tokens, 1), dtype=torch.int32, device="cuda")
    topk_weights = torch.ones((tokens, 1), dtype=torch.float32, device="cuda")
    y_gpu = api.run(x.cuda(), topk_weights, topk_ids, weights, runtime).cpu().float()
    torch.cuda.synchronize()

    # Also score the two intermediate stages so a defect can be localized.
    verdict = {
        "layer": LAYER,
        "expert": EXPERT,
        "bits": bits,
        "expert_ffn_rel_rmse_vs_original_bf16": rel_rmse(y_gpu, y_ref),
        "output_rms_gpu": float(y_gpu.square().mean().sqrt()),
        "output_rms_bf16": float(y_ref.square().mean().sqrt()),
        "finite": bool(torch.isfinite(y_gpu).all()),
    }
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
