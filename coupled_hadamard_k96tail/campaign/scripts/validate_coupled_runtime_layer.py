#!/usr/bin/env python3
"""Run a selected coupled layer through production B12X and an A16 oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "kquant"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qsrt-root", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--result-json", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if str(args.qsrt_root.resolve()) not in sys.path:
        sys.path.insert(0, str(args.qsrt_root.resolve()))
    import torch
    from safetensors import safe_open

    from b12x.moe import glm_sqg_w4a8 as api
    from kquant.sqg_e4m3 import sqg_xor_cheb_t12_bytes
    from qsrt.qsrt_coupled import CoupledHadamardSpec, coupled_execution
    from src.fresh_pipeline_common import atomic_json, sha256_file
    from src.glm52_fresh_sqg.reference import decode_stored_fp16

    manifest = json.loads(args.manifest.read_text())
    manifest_schema = manifest.get("schema")
    if (
        manifest_schema
        not in {
            "glm52-coupled-selected-layer-runtime-v1",
            "glm52-coupled-selected-layer-runtime-v2",
            "glm52-coupled-selected-layer-runtime-v3",
        }
        or manifest.get("complete") is not True
        or sha256_file(args.shard) != manifest.get("shard_sha256")
    ):
        raise ValueError("coupled runtime-layer manifest differs")
    census = manifest.get("bit_census")
    if (
        not isinstance(census, dict)
        or census.get("total") != 768
        or census.get("k3", -1) + census.get("k4", -1) != 768
        or census.get("k4") not in (48, 72, 96)
        or manifest.get("bits_per_weight")
        != (3 * census["k3"] + 4 * census["k4"]) / 768.0
    ):
        raise ValueError("coupled runtime-layer rate census differs")
    profile_binding = manifest.get("final_profile_binding")
    if (
        manifest_schema == "glm52-coupled-selected-layer-runtime-v3"
        and (
            not isinstance(profile_binding, dict)
            or profile_binding.get("schema")
            != "glm52-updated-qsrt-coupled-final-profile-binding-v1"
            or profile_binding.get("no_b300_owner_speed_rescue") is not True
        )
    ):
        raise ValueError("coupled runtime no-shortcut profile binding differs")
    layer = int(manifest["layer"])
    draws = [int(manifest["selected_draws"][str(expert)]) for expert in range(256)]
    device = torch.device(args.device)
    base = f"model.layers.{layer}.mlp.experts"
    with safe_open(str(args.shard), framework="pt", device="cpu") as handle:
        shared_suh = handle.get_tensor(f"{base}.r7_shared.gate_up_suh")
        shared_down_svh = handle.get_tensor(f"{base}.r7_shared.down_svh")
        trellis: dict[str, list[torch.Tensor]] = {
            projection: [] for projection in ("gate_proj", "up_proj", "down_proj")
        }
        scales: dict[str, list[torch.Tensor]] = {
            "gate_proj": [],
            "up_proj": [],
            "down_proj": [],
        }
        bits: dict[str, list[int]] = {
            projection: [] for projection in trellis
        }
        for expert in range(256):
            for projection in trellis:
                prefix = f"{base}.{expert}.{projection}"
                marker = handle.get_tensor(f"{prefix}.sqg")
                if int(marker.reshape(()).item()) & 0xFFFFFFFF != 0x53514731:
                    raise ValueError(f"{prefix}: SQG marker differs")
                current = handle.get_tensor(f"{prefix}.trellis")
                current_bits = int(current.shape[2]) // 16
                if current_bits not in (3, 4):
                    raise ValueError(f"{prefix}: stored rate differs")
                trellis[projection].append(current)
                bits[projection].append(current_bits)
                scale_name = "suh" if projection == "down_proj" else "svh"
                scales[projection].append(handle.get_tensor(f"{prefix}.{scale_name}"))

    gate_trellis = [value.to(device) for value in trellis["gate_proj"]]
    up_trellis = [value.to(device) for value in trellis["up_proj"]]
    down_trellis = [value.to(device) for value in trellis["down_proj"]]
    gate_svh = torch.stack(scales["gate_proj"]).contiguous().to(device)
    up_svh = torch.stack(scales["up_proj"]).contiguous().to(device)
    down_suh = torch.stack(scales["down_proj"]).contiguous().to(device)
    shared_suh_gpu = shared_suh.contiguous().to(device)
    shared_down_gpu = shared_down_svh.contiguous().to(device)
    hidden = int(shared_suh.numel())
    intermediate = int(gate_svh.shape[1])
    prepared = api.prepare_weights(
        gate_trellis=gate_trellis,
        gate_bits=bits["gate_proj"],
        up_trellis=up_trellis,
        up_bits=bits["up_proj"],
        down_trellis=down_trellis,
        down_bits=bits["down_proj"],
        gate_up_suh=shared_suh_gpu,
        gate_svh=gate_svh,
        up_svh=up_svh,
        down_suh=down_suh,
        down_svh=shared_down_gpu,
        hidden_size=hidden,
        intermediate_size=intermediate,
        global_intermediate_size=intermediate,
        tp_rank=0,
        tp_size=1,
        coupled_rotation_draws=torch.tensor(
            draws, dtype=torch.uint8, device=device
        ).contiguous(),
    )
    runtime = api.prepare_runtime(
        prepared,
        max_tokens=args.tokens,
        topk=args.topk,
        output_dtype=torch.bfloat16,
    )
    generator = torch.Generator(device="cpu").manual_seed(20260813)
    hidden_states = (
        torch.randn((args.tokens, hidden), generator=generator)
        .mul_(0.4)
        .to(torch.bfloat16)
        .to(device)
    )
    # Deterministically span experts with all-K3 and mixed K4 projection tuples.
    route_values = torch.tensor(
        [(token * 67 + slot * 29) % 256 for token in range(args.tokens) for slot in range(args.topk)],
        dtype=torch.int32,
    ).reshape(args.tokens, args.topk)
    topk_ids = route_values.to(device)
    topk_weights = torch.softmax(
        torch.randn((args.tokens, args.topk), generator=generator), dim=-1
    ).to(device)
    actual = api.run(hidden_states, topk_weights, topk_ids, prepared, runtime).float()
    torch.cuda.synchronize(device)

    lut = {
        bit: sqg_xor_cheb_t12_bytes(bit).to(device).contiguous() for bit in (3, 4)
    }
    reference = torch.zeros_like(actual)
    for token in range(args.tokens):
        row = hidden_states[token : token + 1].float()
        for slot in range(args.topk):
            expert = int(route_values[token, slot])
            decoded_gate = decode_stored_fp16(
                gate_trellis[expert],
                shared_suh_gpu,
                gate_svh[expert],
                bits=bits["gate_proj"][expert],
                codebook_e4m3=lut[bits["gate_proj"][expert]],
            )
            decoded_up = decode_stored_fp16(
                up_trellis[expert],
                shared_suh_gpu,
                up_svh[expert],
                bits=bits["up_proj"][expert],
                codebook_e4m3=lut[bits["up_proj"][expert]],
            )
            decoded_down = decode_stored_fp16(
                down_trellis[expert],
                down_suh[expert],
                shared_down_gpu,
                bits=bits["down_proj"][expert],
                codebook_e4m3=lut[bits["down_proj"][expert]],
            )
            spec = CoupledHadamardSpec(
                residual_block_size=512,
                preactivation_block_size=128,
                postactivation_block_size=128,
                residual_draw=0,
                intermediate_draw=draws[expert],
                activation="silu",
            )
            weights = (
                decoded_gate.T.contiguous(),
                decoded_up.T.contiguous(),
                decoded_down.T.contiguous(),
            )
            expert_output = coupled_execution(weights, spec).execute(row, weights)
            reference[token].add_(expert_output[0], alpha=float(topk_weights[token, slot]))

    difference = actual - reference
    relative_rmse = float(
        difference.square().mean().sqrt()
        / reference.square().mean().sqrt().clamp_min(1e-20)
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(actual, reference, dim=1).mean()
    )
    result = {
        "schema": "glm52-coupled-selected-layer-b12x-oracle-v2",
        "complete": True,
        "layer": layer,
        "runtime_layer_manifest_schema": manifest_schema,
        "runtime_layer_manifest_id": manifest.get("manifest_id"),
        "runtime_layer_shard_sha256": manifest.get("shard_sha256"),
        "bit_census": census,
        "bits_per_weight": manifest["bits_per_weight"],
        "selected_beta": manifest.get("selected_beta"),
        "final_profile_binding": profile_binding,
        "tokens": args.tokens,
        "topk": args.topk,
        "production_endpoint": "route_packed_direct_e4m3_w4a8",
        "reference_endpoint": "decoded_fp16_weights_a16_updated_qsrt_coupled",
        "relative_rmse": relative_rmse,
        "mean_cosine": cosine,
        "max_abs": float(difference.abs().max()),
        "finite": bool(torch.isfinite(actual).all()),
        "nonzero": bool((actual != 0).any()),
        "pass_limits": {"relative_rmse_max": 0.20, "mean_cosine_min": 0.98},
        "pass": bool(
            torch.isfinite(actual).all()
            and (actual != 0).any()
            and relative_rmse <= 0.20
            and cosine >= 0.98
        ),
    }
    if args.result_json:
        atomic_json(args.result_json, result, overwrite=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit("coupled production runtime differs from the A16 oracle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
