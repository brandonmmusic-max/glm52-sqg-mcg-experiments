#!/usr/bin/env python3
"""Quantify the MXFP8 activation (A8) cost against A16, same weights.

Both paths consume the IDENTICAL stored SQG K3/K4 trellis payloads for one
routed expert. Only the activation precision differs:

  A16: b12x.gemm.trellis_linear.run() per projection (generic dense trellis,
       FP16 activations), FFN composed outside the kernel.
  A8:  the production route-packed GLM SQG W4A8 layer (MXFP8/E4M3 on h and
       act, FP32 accumulation).

Both are scored against the ORIGINAL published BF16 expert weights, so the
difference between the two rel-RMSE values is the activation-quantization
cost that the older W4A16 EXL3 comparison point never paid.
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

    tokens = 32
    generator = torch.Generator().manual_seed(20260812)
    x = (torch.randn((tokens, hidden), generator=generator) * 0.4).to(torch.bfloat16)
    xf = x.float()
    y_ref = (torch.nn.functional.silu(xf @ w_gate) * (xf @ w_up)) @ w_down

    from b12x.gemm.trellis_linear import api as dense_api

    def a16(trellis, suh, svh, activation):
        prepared = dense_api.prepare_weight(
            trellis.cuda(), suh.cuda(), svh.cuda(),
            codebook="sqg_xor_cheb_t12", params_dtype=torch.float16,
        )
        out = dense_api.run(activation.to(torch.float16).cuda(), prepared)
        return out[:, : svh.numel()].float().cpu()

    gate_a16 = a16(gate_t, shared_suh, gate_svh, x)
    up_a16 = a16(up_t, shared_suh, up_svh, x)
    hidden_a16 = torch.nn.functional.silu(gate_a16) * up_a16
    y_a16 = a16(down_t, down_suh, shared_down_svh, hidden_a16)

    import os

    from b12x.moe import glm_sqg_w4a8 as moe_api

    moe_api.validate_glm_route_packed_w4a8_acceptance_kernel()
    endpoint = getattr(moe_api, "glm_sqg_act_endpoint", lambda: "a8")()
    print(f"[oracle] production-path act endpoint = {endpoint} "
          f"(B12X_GLM_SQG_ACT_ENDPOINT={os.environ.get('B12X_GLM_SQG_ACT_ENDPOINT', 'unset')})")
    target = quant["glm_sqg_w4a8"]["down_targets"][str(LAYER)]
    weights = moe_api.prepare_weights(
        gate_trellis=[gate_t.cuda()], gate_bits=[bits["gate"]],
        up_trellis=[up_t.cuda()], up_bits=[bits["up"]],
        down_trellis=[down_t.cuda()], down_bits=[bits["down"]],
        gate_up_suh=shared_suh.cuda(),
        gate_svh=gate_svh.unsqueeze(0).contiguous().cuda(),
        up_svh=up_svh.unsqueeze(0).contiguous().cuda(),
        down_suh=down_suh.unsqueeze(0).contiguous().cuda(),
        down_svh=shared_down_svh.cuda(),
        hidden_size=hidden, intermediate_size=intermediate,
        global_intermediate_size=intermediate, tp_rank=0, tp_size=1,
        derived_down_target_id=target["derived_down_target_id"],
        down_target_beta=target["down_target_beta"],
    )
    runtime = moe_api.prepare_runtime(
        weights, max_tokens=tokens, topk=1, output_dtype=torch.bfloat16
    )
    y_a8 = moe_api.run(
        x.cuda(),
        torch.ones((tokens, 1), dtype=torch.float32, device="cuda"),
        torch.zeros((tokens, 1), dtype=torch.int32, device="cuda"),
        weights, runtime,
    ).cpu().float()
    torch.cuda.synchronize()

    e_a16 = rel_rmse(y_a16, y_ref)
    e_a8 = rel_rmse(y_a8, y_ref)
    print(json.dumps({
        "layer": LAYER, "expert": EXPERT, "bits": bits,
        "a16_ffn_rel_rmse_vs_bf16": e_a16,
        "a8_ffn_rel_rmse_vs_bf16": e_a8,
        "a8_over_a16_ratio": (e_a8 / e_a16) if e_a16 else None,
        "a8_vs_a16_direct_rel_rmse": rel_rmse(y_a8, y_a16),
        "production_path_act_endpoint": endpoint,
        "rms": {
            "bf16": float(y_ref.square().mean().sqrt()),
            "a16": float(y_a16.square().mean().sqrt()),
            "a8": float(y_a8.square().mean().sqrt()),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
