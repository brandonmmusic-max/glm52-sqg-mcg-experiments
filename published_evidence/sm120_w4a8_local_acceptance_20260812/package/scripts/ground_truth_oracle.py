#!/usr/bin/env python3
"""Ground-truth oracle: GPU-decoded weight vs ORIGINAL BF16 weight.

Extracts the effective weight matrix from the runtime K6 path by probing with
identity blocks, then compares three ways:
  W_gpu  vs W_orig   -> runtime decode correctness (expected ~K6 quant error)
  W_ref  vs W_orig   -> my reference-decoder correctness
  W_gpu  vs W_ref    -> which one is wrong when they disagree
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/reference-src")
from glm52_fresh_sqg.reference import decode_stored_fp16, relative_rmse  # noqa: E402

MODEL = Path("/model")
NAME = "model.layers.3.mlp.shared_experts.down_proj"


def load(weight_map: dict, name: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(str(MODEL / weight_map[name]), framework="pt", device="cpu") as h:
        return h.get_tensor(name)


def codebook() -> torch.Tensor:
    from b12x._lib.quant.sqg_e4m3 import decode_sqg_cheb_normal_e4m3_ranks_torch

    table = decode_sqg_cheb_normal_e4m3_ranks_torch(
        torch.arange(1 << 16, dtype=torch.int64)
    ).flatten()
    if table.dtype == torch.float8_e4m3fn:
        table = table.view(torch.uint8)
    return table.cpu()


def main() -> None:
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    trellis = load(weight_map, NAME + ".trellis")
    suh = load(weight_map, NAME + ".suh")
    svh = load(weight_map, NAME + ".svh")
    k = trellis.shape[0] * 16
    n = trellis.shape[1] * 16

    from safetensors import safe_open

    with safe_open("/bf16/model-00079-of-00282.safetensors", framework="pt", device="cpu") as h:
        w_orig = h.get_tensor(NAME + ".weight")  # HF layout [out, in]
    print("orig", tuple(w_orig.shape), "packed K,N =", k, n)
    # x @ W convention: W is [K=in, N=out]; HF weight is [out, in] -> transpose
    w_orig_kn = w_orig.t().float()
    logical_k, logical_n = w_orig_kn.shape

    w_ref = decode_stored_fp16(trellis, suh, svh, bits=6, codebook_e4m3=codebook())

    from b12x.gemm.trellis_linear import api

    prepared = api.prepare_weight(
        trellis.cuda(), suh.cuda(), svh.cuda(),
        codebook="sqg_xor_cheb_t12", params_dtype=torch.float16,
    )
    rows = []
    eye = torch.eye(k, dtype=torch.float16)
    for start in range(0, k, 256):
        block = eye[start : start + 256].cuda()
        rows.append(api.run_sqg_k6_w6a16(block, prepared).cpu().float())
    w_gpu = torch.cat(rows, dim=0)[:, :n]
    torch.cuda.synchronize()

    w_gpu_l = w_gpu[:logical_k, :logical_n]
    w_ref_l = w_ref.float()[:logical_k, :logical_n]
    print(json.dumps({
        "gpu_vs_orig": relative_rmse(w_gpu_l, w_orig_kn),
        "ref_vs_orig": relative_rmse(w_ref_l, w_orig_kn),
        "gpu_vs_ref": relative_rmse(w_gpu_l, w_ref_l),
    }, indent=2))


if __name__ == "__main__":
    main()
