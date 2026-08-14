#!/usr/bin/env python3
"""CPU-only raw weight NMSE comparison for matched GLM MCG and SQG tensors.

MCG is independently unpacked and decoded from the protected checkpoint.  Each
CPU reconstruction must close against the production reconstruction hash.
SQG uses the source-relative RMSE recorded by its independent packed-byte
closure; squaring that value yields raw NMSE.  Both arms are bound to the same
official BF16 tensor payload before a row is accepted.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np
import torch
from safetensors import safe_open


DEFAULT_MCG_ROOT = Path(
    "/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED"
)
DEFAULT_SQG_ROOT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-encode-absfix-r2"
)
DEFAULT_BF16_ROOT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/bf16_layers"
)
DEFAULT_OUTPUT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/"
    "raw_encoded_nmse_sqg_vs_mcg.json"
)
MCG_MULT = np.uint64(0xCBAC1FED)
MCG_MASK = np.uint32(0x8FFF8FFF)
MCG_XOR = np.uint32(0x3B603B60)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _mcg_lut() -> torch.Tensor:
    indices = np.arange(1 << 16, dtype=np.uint64)
    products = ((indices * MCG_MULT) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    products = ((products & MCG_MASK) ^ MCG_XOR).astype(np.uint32)
    low = (products & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    high = (
        (products >> np.uint32(16)) & np.uint32(0xFFFF)
    ).astype(np.uint16).view(np.float16)
    values = (low.astype(np.float16) + high.astype(np.float16)).astype(np.float16)
    result = torch.from_numpy(np.ascontiguousarray(values))
    if result.shape != (1 << 16,) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("MCG CPU LUT construction failed")
    return result


def _payload_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(
        memoryview(value.view(torch.uint8).numpy()).cast("B")
    ).hexdigest()


def _load_reference_functions(experiment_root: Path):
    import sys

    kquant_root = experiment_root / "kquant"
    for path in (str(kquant_root), str(experiment_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from kquant.exl3_reference import decode_exl3_weight
    from src.glm52_fresh_sqg.reference import unpack_trellis_states

    return decode_exl3_weight, unpack_trellis_states


def _load_sqg_manifests(sqg_root: Path, layer: int) -> dict[int, dict[str, Any]]:
    directory = sqg_root / f"layer_{layer:03d}" / "experts"
    paths = sorted(directory.glob(f"layer-{layer:03d}-expert-*.json"))
    if len(paths) != 256:
        raise RuntimeError(
            f"layer {layer}: expected 256 canonical SQG expert manifests, got {len(paths)}"
        )
    result: dict[int, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        expert = int(payload["expert"])
        if expert in result:
            raise RuntimeError(f"layer {layer}: duplicate SQG expert {expert}")
        if not payload.get("complete"):
            raise RuntimeError(f"layer {layer} expert {expert}: incomplete SQG manifest")
        result[expert] = payload
    if set(result) != set(range(256)):
        raise RuntimeError(f"layer {layer}: SQG expert census is not exactly 0..255")
    return result


def _read_tensor(path: Path, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _source_exl(
    source_hf: torch.Tensor,
    permutation: torch.Tensor,
    projection: str,
) -> torch.Tensor:
    if projection in ("gate_proj", "up_proj"):
        return source_hf.index_select(0, permutation).T.contiguous().float()
    if projection == "down_proj":
        return source_hf.index_select(1, permutation).T.contiguous().float()
    raise ValueError(f"unsupported projection: {projection}")


def _float32_nmse(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = actual.float() - expected.float()
    relative_rmse = float(
        difference.square().mean().sqrt()
        / expected.float().square().mean().sqrt().clamp_min(1.0e-20)
    )
    return relative_rmse * relative_rmse, relative_rmse


def _compare_layer(
    layer: int,
    mcg_root_raw: str,
    sqg_root_raw: str,
    bf16_root_raw: str,
    experiment_root_raw: str,
    torch_threads: int,
    limit_experts: int | None,
) -> list[dict[str, Any]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    mcg_root = Path(mcg_root_raw)
    sqg_root = Path(sqg_root_raw)
    bf16_root = Path(bf16_root_raw)
    experiment_root = Path(experiment_root_raw)
    decode_exl3_weight, unpack_trellis_states = _load_reference_functions(
        experiment_root
    )
    lut = _mcg_lut()

    sidecar_path = mcg_root / f"r7-experts-layer-{layer:03d}.json"
    payload_path = mcg_root / f"r7-experts-layer-{layer:03d}.safetensors"
    sidecar = json.loads(sidecar_path.read_text())
    sqg_manifests = _load_sqg_manifests(sqg_root, layer)
    expert_count = 256 if limit_experts is None else min(256, limit_experts)
    rows: list[dict[str, Any]] = []

    with safe_open(payload_path, framework="pt", device="cpu") as mcg_payload:
        for expert in range(expert_count):
            permutation = torch.tensor(
                sidecar["permutations"][str(expert)]["new_to_old"],
                dtype=torch.long,
            )
            if permutation.shape != (2048,) or not torch.equal(
                torch.sort(permutation).values, torch.arange(2048)
            ):
                raise RuntimeError(
                    f"layer {layer} expert {expert}: invalid MCG permutation"
                )
            sqg_expert = sqg_manifests[expert]

            for projection in PROJECTIONS:
                prefix = (
                    f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                )
                bits = int(sidecar["bit_map"][prefix])
                sqg_tensor = sqg_expert["tensor_manifests"][prefix]
                sqg_bits = int(sqg_tensor["bits"])
                if bits != sqg_bits or bits not in (3, 4):
                    raise RuntimeError(
                        f"{prefix}: MCG/SQG rate mismatch {bits} != {sqg_bits}"
                    )

                mcg_source = sidecar["tensor_provenance"][prefix]
                sqg_source = sqg_tensor["source"]
                if mcg_source["bf16_sha256"] != sqg_source["tensor_payload_sha256"]:
                    raise RuntimeError(f"{prefix}: BF16 source payload binding differs")
                if mcg_source["source_name"] != sqg_source["tensor_name"]:
                    raise RuntimeError(f"{prefix}: BF16 source tensor name differs")

                packed = mcg_payload.get_tensor(prefix + ".trellis")
                vector_refs = sidecar["vector_refs"][prefix]
                suh = mcg_payload.get_tensor(vector_refs["suh"])
                svh = mcg_payload.get_tensor(vector_refs["svh"])
                states = unpack_trellis_states(packed, bits)
                decoded = decode_exl3_weight(
                    states,
                    suh,
                    svh,
                    codebook_values=lut,
                    bits=bits,
                )
                observed_hash = _payload_sha256(decoded.half())
                expected_hash = sidecar["roundtrip_hashes"][prefix][
                    "reconstruction_sha256"
                ]
                if observed_hash != expected_hash:
                    raise RuntimeError(
                        f"{prefix}: CPU MCG reconstruction hash differs: "
                        f"{observed_hash} != {expected_hash}"
                    )

                source_path = bf16_root / mcg_source["source_shard"]
                source_hf = _read_tensor(source_path, mcg_source["source_name"])
                source = _source_exl(source_hf, permutation, projection)
                if source.shape != decoded.shape:
                    raise RuntimeError(
                        f"{prefix}: source/decode shape mismatch "
                        f"{tuple(source.shape)} != {tuple(decoded.shape)}"
                    )

                mcg_nmse, mcg_rrmse = _float32_nmse(decoded, source)
                sqg_rrmse = float(
                    sqg_tensor["decoded_closure"]["source_relative_rmse"]
                )
                sqg_nmse = sqg_rrmse * sqg_rrmse
                source_ss = float(source.double().square().sum().item())
                ratio = sqg_nmse / mcg_nmse
                rows.append(
                    {
                        "tensor": prefix,
                        "layer": layer,
                        "expert": expert,
                        "projection": projection,
                        "bits": bits,
                        "elements": int(source.numel()),
                        "source_sum_squares": source_ss,
                        "source_payload_sha256": mcg_source["bf16_sha256"],
                        "mcg": {
                            "nmse": mcg_nmse,
                            "relative_rmse": mcg_rrmse,
                            "sse": mcg_nmse * source_ss,
                            "reconstruction_sha256": observed_hash,
                        },
                        "sqg": {
                            "nmse": sqg_nmse,
                            "relative_rmse": sqg_rrmse,
                            "sse": sqg_nmse * source_ss,
                            "reconstruction_sha256": sqg_tensor[
                                "decoded_closure"
                            ]["decoded_exl_sha256"],
                            "closure_implementation": sqg_tensor[
                                "decoded_closure"
                            ]["implementation"],
                        },
                        "sqg_over_mcg_nmse": ratio,
                        "sqg_nmse_reduction_percent": (1.0 - ratio) * 100.0,
                        "winner": "sqg" if sqg_nmse < mcg_nmse else (
                            "mcg" if mcg_nmse < sqg_nmse else "tie"
                        ),
                    }
                )
                del packed, states, decoded, source_hf, source
    return rows


def _summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot summarize an empty comparison")
    source_ss = sum(row["source_sum_squares"] for row in materialized)
    mcg_sse = sum(row["mcg"]["sse"] for row in materialized)
    sqg_sse = sum(row["sqg"]["sse"] for row in materialized)
    mcg_values = [row["mcg"]["nmse"] for row in materialized]
    sqg_values = [row["sqg"]["nmse"] for row in materialized]
    ratios = [row["sqg_over_mcg_nmse"] for row in materialized]
    winners = defaultdict(int)
    for row in materialized:
        winners[row["winner"]] += 1
    energy_mcg = mcg_sse / source_ss
    energy_sqg = sqg_sse / source_ss
    return {
        "tensors": len(materialized),
        "elements": sum(row["elements"] for row in materialized),
        "source_sum_squares": source_ss,
        "energy_weighted_nmse": {
            "mcg": energy_mcg,
            "sqg": energy_sqg,
            "sqg_over_mcg": energy_sqg / energy_mcg,
            "sqg_reduction_percent": (1.0 - energy_sqg / energy_mcg) * 100.0,
        },
        "macro_mean_nmse": {
            "mcg": statistics.fmean(mcg_values),
            "sqg": statistics.fmean(sqg_values),
            "sqg_over_mcg": statistics.fmean(sqg_values)
            / statistics.fmean(mcg_values),
        },
        "median_tensor_nmse": {
            "mcg": statistics.median(mcg_values),
            "sqg": statistics.median(sqg_values),
        },
        "paired_ratio": {
            "median_sqg_over_mcg": statistics.median(ratios),
            "geometric_mean_sqg_over_mcg": math.exp(
                statistics.fmean(math.log(value) for value in ratios)
            ),
        },
        "winner_counts": dict(sorted(winners.items())),
        "sqg_win_rate": winners["sqg"] / len(materialized),
    }


def _group_summaries(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    def order(item: tuple[str, list[dict[str, Any]]]) -> tuple[int, int | str]:
        try:
            return (0, int(item[0]))
        except ValueError:
            return (1, item[0])

    return {key: _summarize(value) for key, value in sorted(groups.items(), key=order)}


def _markdown_table(groups: dict[str, Any], heading: str) -> list[str]:
    lines = [
        f"### {heading}",
        "",
        "| Group | Tensors | MCG NMSE | SQG NMSE | SQG/MCG | SQG reduction | SQG wins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    def order(item: tuple[str, Any]) -> tuple[int, int | str]:
        try:
            return (0, int(item[0]))
        except ValueError:
            return (1, item[0])

    for key, value in sorted(groups.items(), key=order):
        energy = value["energy_weighted_nmse"]
        lines.append(
            f"| {key} | {value['tensors']:,} | {energy['mcg']:.9e} | "
            f"{energy['sqg']:.9e} | {energy['sqg_over_mcg']:.6f} | "
            f"{energy['sqg_reduction_percent']:+.3f}% | "
            f"{value['winner_counts'].get('sqg', 0):,} ({value['sqg_win_rate']:.1%}) |"
        )
    lines.append("")
    return lines


def _render_report(result: dict[str, Any]) -> str:
    overall = result["summary"]["overall"]
    energy = overall["energy_weighted_nmse"]
    mcg_wins = [row for row in result["tensors"] if row["winner"] == "mcg"]
    mcg_wins_by_layer = Counter(row["layer"] for row in mcg_wins)
    mcg_wins_by_bits = Counter(row["bits"] for row in mcg_wins)
    lines = [
        "# Raw encoded-weight NMSE: SQG versus MCG",
        "",
        "Lower is better. This is an encoded-weight distortion comparison, not KLD.",
        "",
        "## Overall result",
        "",
        f"- Matched tensors: **{overall['tensors']:,}**",
        f"- MCG energy-weighted NMSE: **`{energy['mcg']:.9e}`**",
        f"- SQG energy-weighted NMSE: **`{energy['sqg']:.9e}`**",
        f"- SQG/MCG: **`{energy['sqg_over_mcg']:.6f}`**",
        f"- SQG raw-NMSE reduction: **`{energy['sqg_reduction_percent']:+.3f}%`**",
        f"- SQG tensor wins: **{overall['winner_counts'].get('sqg', 0):,} / "
        f"{overall['tensors']:,} ({overall['sqg_win_rate']:.1%})**",
        f"- MCG exceptions: **{len(mcg_wins):,}**; "
        f"{mcg_wins_by_bits.get(3, 0)} were K3 and "
        f"{mcg_wins_by_bits.get(4, 0)} were K4",
        "- MCG exceptions by layer: "
        + ", ".join(
            f"L{layer}={mcg_wins_by_layer.get(layer, 0)}"
            for layer in result["layers"]
        ),
        "",
    ]
    lines.extend(_markdown_table(result["summary"]["by_bits"], "By rate"))
    lines.extend(
        _markdown_table(result["summary"]["by_projection"], "By projection")
    )
    lines.extend(_markdown_table(result["summary"]["by_layer"], "By layer"))
    lines.extend(
        [
            "## Method",
            "",
            "- MCG was unpacked and decoded entirely on CPU from its production trellis bytes and stored FP16 scale vectors.",
            "- Every MCG decoded tensor was required to match its sealed production reconstruction SHA-256 byte-for-byte.",
            "- SQG NMSE is the square of `source_relative_rmse` from the independent packed-byte SQG closure recorded during encoding.",
            "- Each pair was required to bind to the same official BF16 tensor payload and the same K3/K4 assignment.",
            "- Aggregate NMSE is `sum(SSE) / sum(BF16 weight energy)`; per-tensor macro and paired statistics are retained in the JSON.",
            "- No GPU was visible to this process.",
            "",
            "## Interpretation limit",
            "",
            "Raw weight NMSE does not include Hessian sensitivity, routing, error compounding, activation quantization, or runtime dispatch. It answers whether the stored SQG weights are geometrically closer to BF16 than the stored MCG weights, not whether full-model KLD is lower.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcg-root", type=Path, default=DEFAULT_MCG_ROOT)
    parser.add_argument("--sqg-root", type=Path, default=DEFAULT_SQG_ROOT)
    parser.add_argument("--bf16-root", type=Path, default=DEFAULT_BF16_ROOT)
    parser.add_argument(
        "--experiment-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 28, 52, 77])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads-per-worker", type=int, default=12)
    parser.add_argument("--limit-experts", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0 or args.torch_threads_per_worker <= 0:
        raise ValueError("worker and thread counts must be positive")
    if args.limit_experts is not None and args.limit_experts <= 0:
        raise ValueError("--limit-experts must be positive")
    if len(set(args.layers)) != len(args.layers):
        raise ValueError("layer list contains duplicates")

    context = multiprocessing.get_context("spawn")
    rows: list[dict[str, Any]] = []
    max_workers = min(args.workers, len(args.layers))
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as pool:
        futures = {
            pool.submit(
                _compare_layer,
                layer,
                str(args.mcg_root.resolve()),
                str(args.sqg_root.resolve()),
                str(args.bf16_root.resolve()),
                str(args.experiment_root.resolve()),
                args.torch_threads_per_worker,
                args.limit_experts,
            ): layer
            for layer in args.layers
        }
        for future in as_completed(futures):
            layer = futures[future]
            layer_rows = future.result()
            rows.extend(layer_rows)
            print(f"layer {layer}: compared {len(layer_rows)} tensors", flush=True)

    rows.sort(key=lambda row: (row["layer"], row["expert"], row["projection"]))
    result = {
        "schema": "glm52-raw-encoded-nmse-sqg-vs-mcg-v1",
        "device": "cpu",
        "gpu_visible": False,
        "layers": sorted(args.layers),
        "mcg_root": str(args.mcg_root.resolve()),
        "sqg_root": str(args.sqg_root.resolve()),
        "bf16_root": str(args.bf16_root.resolve()),
        "comparison_contract": {
            "metric": "sum((decoded - bf16)^2) / sum(bf16^2)",
            "matched_official_bf16_payload_required": True,
            "matched_k3_k4_assignment_required": True,
            "mcg_independent_cpu_decode": True,
            "mcg_all_reconstruction_hashes_closed": True,
            "sqg_metric_source": "independent_pytorch_packed_fp16_v1 closure",
        },
        "summary": {
            "overall": _summarize(rows),
            "by_bits": _group_summaries(rows, "bits"),
            "by_projection": _group_summaries(rows, "projection"),
            "by_layer": _group_summaries(rows, "layer"),
        },
        "tensors": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report_path = args.output.with_suffix(".md")
    report_path.write_text(_render_report(result))
    print(_render_report(result), flush=True)
    print(f"json: {args.output}", flush=True)
    print(f"report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
