#!/usr/bin/env python3
"""One-layer GPU probe: prove TP4 resharding + native W4A8 kernel execution.

Loads one routed layer of the published r7-layout checkpoint from disk,
reproduces exl3.py's TP-local intermediate-axis slicing for every rank,
builds the six K3/K4 pools through b12x.moe.glm_sqg_w4a8.prepare_weights
(which validates the per-projection K3/K4 partitions, pool byte counts, scale
finiteness, and topology closure), plans the sealed PR11 runtime, and runs the
actual route-packed W4A8 kernel with random routing. Passes only if every
rank's output is finite and the layer's K3/K4 census matches the sealed
384/384 contract.

Runs inside the SM120 image on one GPU. No weights are modified.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open


def load_layer(model_root: Path, layer: int, experts: int):
    index = json.loads((model_root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    handles: dict[str, object] = {}

    def tensor(name: str) -> torch.Tensor:
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(
                str(model_root / shard), framework="pt", device="cpu"
            )
        return handles[shard].get_tensor(name)

    base = f"model.layers.{layer}.mlp.experts"
    shared_suh = tensor(f"{base}.r7_shared.gate_up_suh")
    shared_down_svh = tensor(f"{base}.r7_shared.down_svh")
    data: dict[str, list[torch.Tensor]] = {
        "gate_trellis": [],
        "gate_svh": [],
        "up_trellis": [],
        "up_svh": [],
        "down_trellis": [],
        "down_suh": [],
    }
    markers = Counter()
    for expert in range(experts):
        for projection, trellis_key, vector_key, vector_name in (
            ("gate_proj", "gate_trellis", "gate_svh", "svh"),
            ("up_proj", "up_trellis", "up_svh", "svh"),
            ("down_proj", "down_trellis", "down_suh", "suh"),
        ):
            prefix = f"{base}.{expert}.{projection}"
            data[trellis_key].append(tensor(f"{prefix}.trellis"))
            data[vector_key].append(tensor(f"{prefix}.{vector_name}"))
            marker = tensor(f"{prefix}.sqg")
            value = int(marker.reshape(()).item()) & 0xFFFFFFFF
            if value != 0x53514731:
                raise SystemExit(f"{prefix}: bad SQG sentinel 0x{value:08x}")
            markers[projection] += 1
    return shared_suh, shared_down_svh, data, markers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/model")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()

    from b12x.moe import glm_sqg_w4a8 as api

    api.validate_glm_route_packed_w4a8_acceptance_kernel()

    model_root = Path(args.model)
    quant = json.loads((model_root / "quantization_config.json").read_text())
    contract = quant["glm_sqg_w4a8"]
    census = contract["per_layer_bit_census"]

    shared_suh, shared_down_svh, data, markers = load_layer(
        model_root, args.layer, args.experts
    )
    hidden = shared_suh.numel()
    global_intermediate = data["gate_svh"][0].numel()
    local = global_intermediate // args.tp_size
    if local % 128:
        raise SystemExit(
            f"local intermediate {local} is not 128-aligned for TP{args.tp_size}"
        )

    bits = {
        key: [int(t.shape[2]) // 16 for t in data[key]]
        for key in ("gate_trellis", "up_trellis", "down_trellis")
    }
    layer_census = Counter()
    for key in bits:
        layer_census.update(bits[key])
    if (
        layer_census[3] != census["k3"]
        or layer_census[4] != census["k4"]
        or sum(layer_census.values()) != census["total"]
    ):
        raise SystemExit(
            f"layer {args.layer} census {dict(layer_census)} != sealed {census}"
        )

    device = torch.device("cuda:0")
    results = []
    for rank in range(args.tp_size):
        start = rank * local
        gate_trellis = [
            t.narrow(1, start // 16, local // 16).contiguous().to(device)
            for t in data["gate_trellis"]
        ]
        up_trellis = [
            t.narrow(1, start // 16, local // 16).contiguous().to(device)
            for t in data["up_trellis"]
        ]
        down_trellis = [
            t.narrow(0, start // 16, local // 16).contiguous().to(device)
            for t in data["down_trellis"]
        ]
        gate_svh = torch.stack(
            [t.narrow(0, start, local) for t in data["gate_svh"]]
        ).contiguous().to(device)
        up_svh = torch.stack(
            [t.narrow(0, start, local) for t in data["up_svh"]]
        ).contiguous().to(device)
        down_suh = torch.stack(
            [t.narrow(0, start, local) for t in data["down_suh"]]
        ).contiguous().to(device)

        weights = api.prepare_weights(
            gate_trellis=gate_trellis,
            gate_bits=bits["gate_trellis"],
            up_trellis=up_trellis,
            up_bits=bits["up_trellis"],
            down_trellis=down_trellis,
            down_bits=bits["down_trellis"],
            gate_up_suh=shared_suh.to(device),
            gate_svh=gate_svh,
            up_svh=up_svh,
            down_suh=down_suh,
            down_svh=shared_down_svh.to(device),
            hidden_size=hidden,
            intermediate_size=local,
            global_intermediate_size=global_intermediate,
            tp_rank=rank,
            tp_size=args.tp_size,
            derived_down_target_id=contract["down_targets"][str(args.layer)][
                "derived_down_target_id"
            ],
            down_target_beta=contract["down_targets"][str(args.layer)][
                "down_target_beta"
            ],
        )
        runtime = api.prepare_runtime(
            weights,
            max_tokens=args.tokens,
            topk=args.topk,
            output_dtype=torch.bfloat16,
        )
        generator = torch.Generator(device="cpu").manual_seed(20260812 + rank)
        hidden_states = (
            torch.randn(
                (args.tokens, hidden), generator=generator, dtype=torch.float32
            )
            .to(torch.bfloat16)
            .to(device)
        )
        topk_ids = torch.randint(
            0,
            args.experts,
            (args.tokens, args.topk),
            generator=generator,
            dtype=torch.int32,
        ).to(device)
        topk_weights = torch.softmax(
            torch.randn(
                (args.tokens, args.topk), generator=generator, dtype=torch.float32
            ),
            dim=-1,
        ).to(device)
        output = api.run(hidden_states, topk_weights, topk_ids, weights, runtime)
        torch.cuda.synchronize(device)
        finite = bool(torch.isfinite(output).all())
        magnitude = float(output.float().abs().mean())
        nonzero = bool((output != 0).any())
        results.append(
            {
                "tp_rank": rank,
                "finite": finite,
                "nonzero": nonzero,
                "mean_abs": magnitude,
                "k3": layer_census[3],
                "k4": layer_census[4],
            }
        )
        del weights, runtime, output, gate_trellis, up_trellis, down_trellis
        torch.cuda.empty_cache()
        if not finite or not nonzero:
            raise SystemExit(f"rank {rank}: non-finite or all-zero kernel output")

    receipt = {
        "schema": "glm52-sqg-w4a8-sm120-gpu-probe-v1",
        "layer": args.layer,
        "experts": args.experts,
        "hidden_size": hidden,
        "global_intermediate": global_intermediate,
        "local_intermediate": local,
        "tp_size": args.tp_size,
        "sqg_marker_count": sum(markers.values()),
        "census": dict(layer_census),
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "ranks": results,
        "pass": True,
    }
    print(json.dumps(receipt, indent=2))
    if args.result_json:
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.result_json).write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
