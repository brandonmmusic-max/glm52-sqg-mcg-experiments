#!/usr/bin/env python3
"""Numeric codec oracle: GPU runtime vs canonical reference decode.

Part 1 (K6 dense): pick one real non-routed K6 matrix, reconstruct the
effective weight with glm52_fresh_sqg.reference.decode_stored_fp16, and
compare `x @ W_ref` against api.run_sqg_k6_w6a16(x, prepared) on GPU.

Part 2 (routed W4A8): pick one routed expert's gate/up/down, reconstruct the
reference expert FFN output silu(x@Wg)*(x@Wu) @ Wd, and compare against the
route-packed GLM W4A8 layer run with a single expert.

Both parts report relative RMSE; healthy quant paths land in the low
percent range, layout mismatches land near 100%.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/reference-src")
from glm52_fresh_sqg.reference import decode_stored_fp16, relative_rmse  # noqa: E402

MODEL = Path("/model")


def read_header(path: Path) -> dict:
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


def load_tensor(weight_map: dict, name: str) -> torch.Tensor:
    from safetensors import safe_open

    shard = MODEL / weight_map[name]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def sqg_codebook() -> torch.Tensor:
    # Full 16-bit state -> raw E4M3 label table straight from the codec's
    # defining function (the kernel's 4 KiB T12 staircase is derived from it).
    from b12x._lib.quant.sqg_e4m3 import decode_sqg_cheb_normal_e4m3_ranks_torch

    table = decode_sqg_cheb_normal_e4m3_ranks_torch(
        torch.arange(1 << 16, dtype=torch.int64)
    ).flatten()
    if table.dtype == torch.float8_e4m3fn:
        table = table.view(torch.uint8)
    if table.dtype != torch.uint8 or table.numel() != (1 << 16):
        raise SystemExit(f"bad codebook: {table.dtype} {table.numel()}")
    return table.cpu()


def part1_k6_dense(index: dict) -> float:
    weight_map = index["weight_map"]
    name = next(
        n for n in weight_map
        if n.endswith(".trellis") and ".mlp.experts." not in n
        and "layers.3." in n
    )
    prefix = name.removesuffix(".trellis")
    print(f"[K6] {prefix}")
    trellis = load_tensor(weight_map, name)
    suh = load_tensor(weight_map, prefix + ".suh")
    svh = load_tensor(weight_map, prefix + ".svh")
    bits = trellis.shape[2] // 16
    assert bits == 6, bits
    weight_ref = decode_stored_fp16(
        trellis, suh, svh, bits=6, codebook_e4m3=sqg_codebook()
    )
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn((8, trellis.shape[0] * 16), generator=generator).to(torch.float16)
    y_ref = (x.float() @ weight_ref.float()).to(torch.float32)

    from b12x.gemm.trellis_linear import api

    prepared = api.prepare_weight(
        trellis.cuda(), suh.cuda(), svh.cuda(),
        codebook="sqg_xor_cheb_t12", params_dtype=torch.float16,
    )
    y_gpu = api.run_sqg_k6_w6a16(x.cuda(), prepared)[:, : y_ref.shape[1]]
    y_generic = api.run(x.cuda(), prepared)[:, : y_ref.shape[1]]
    torch.cuda.synchronize()
    error = relative_rmse(y_gpu.cpu(), y_ref)
    cross = relative_rmse(y_gpu.cpu(), y_generic.cpu().float())
    print(f"[K6] rel RMSE gpu-vs-ref: {error:.6f}  k6-vs-genericW4A16: {cross:.6f}")
    return error


def part2_routed_w4a8(index: dict, quant: dict) -> float:
    weight_map = index["weight_map"]
    layer, expert = 3, 0
    base = f"model.layers.{layer}.mlp.experts"
    gate_t = load_tensor(weight_map, f"{base}.{expert}.gate_proj.trellis")
    up_t = load_tensor(weight_map, f"{base}.{expert}.up_proj.trellis")
    down_t = load_tensor(weight_map, f"{base}.{expert}.down_proj.trellis")
    shared_suh = load_tensor(weight_map, f"{base}.r7_shared.gate_up_suh")
    shared_down_svh = load_tensor(weight_map, f"{base}.r7_shared.down_svh")
    gate_svh = load_tensor(weight_map, f"{base}.{expert}.gate_proj.svh")
    up_svh = load_tensor(weight_map, f"{base}.{expert}.up_proj.svh")
    down_suh = load_tensor(weight_map, f"{base}.{expert}.down_proj.suh")
    codebook = sqg_codebook()
    bits = {
        "gate": gate_t.shape[2] // 16,
        "up": up_t.shape[2] // 16,
        "down": down_t.shape[2] // 16,
    }
    print(f"[W4A8] layer {layer} expert {expert} bits {bits}")
    weight_gate = decode_stored_fp16(gate_t, shared_suh, gate_svh, bits=bits["gate"], codebook_e4m3=codebook)
    weight_up = decode_stored_fp16(up_t, shared_suh, up_svh, bits=bits["up"], codebook_e4m3=codebook)
    weight_down = decode_stored_fp16(down_t, down_suh, shared_down_svh, bits=bits["down"], codebook_e4m3=codebook)
    generator = torch.Generator().manual_seed(4321)
    tokens = 8
    x = (torch.randn((tokens, weight_gate.shape[0]), generator=generator) * 0.5).to(torch.bfloat16)
    xf = x.float()
    hidden_ref = torch.nn.functional.silu(xf @ weight_gate.float()) * (xf @ weight_up.float())
    y_ref = hidden_ref @ weight_down.float()

    # GPU: single-expert route-packed layer with the FULL logical tensors
    # (tp_size=1) so the reference composition matches exactly.
    from b12x.moe import glm_sqg_w4a8 as api

    target = quant["glm_sqg_w4a8"]["down_targets"][str(layer)]
    weights = api.prepare_weights(
        gate_trellis=[gate_t.cuda()], gate_bits=[bits["gate"]],
        up_trellis=[up_t.cuda()], up_bits=[bits["up"]],
        down_trellis=[down_t.cuda()], down_bits=[bits["down"]],
        gate_up_suh=shared_suh.cuda(),
        gate_svh=gate_svh.unsqueeze(0).contiguous().cuda(),
        up_svh=up_svh.unsqueeze(0).contiguous().cuda(),
        down_suh=down_suh.unsqueeze(0).contiguous().cuda(),
        down_svh=shared_down_svh.cuda(),
        hidden_size=weight_gate.shape[0],
        intermediate_size=weight_gate.shape[1],
        global_intermediate_size=weight_gate.shape[1],
        tp_rank=0, tp_size=1,
        derived_down_target_id=target["derived_down_target_id"],
        down_target_beta=target["down_target_beta"],
    )
    runtime = api.prepare_runtime(weights, max_tokens=tokens, topk=1, output_dtype=torch.bfloat16)
    topk_ids = torch.zeros((tokens, 1), dtype=torch.int32, device="cuda")
    topk_weights = torch.ones((tokens, 1), dtype=torch.float32, device="cuda")
    y_gpu = api.run(x.cuda(), topk_weights, topk_ids, weights, runtime)
    torch.cuda.synchronize()
    error = relative_rmse(y_gpu.cpu(), y_ref)
    print(f"[W4A8] rel RMSE gpu-vs-ref: {error:.6f}")
    return error


def main() -> None:
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    quant = json.loads((MODEL / "quantization_config.json").read_text())
    e_k6 = part1_k6_dense(index)
    e_moe = part2_routed_w4a8(index, quant)
    verdict = {
        "k6_dense_rel_rmse": e_k6,
        "w4a8_routed_rel_rmse": e_moe,
        "k6_ok": e_k6 < 0.05,
        "w4a8_ok": e_moe < 0.08,
    }
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
