#!/usr/bin/env python3
"""CPU-only matched GLM MCG/SQG E4M3 endpoint distortion test.

The FP8 weight operand is the regularized trellis label, before the existing
Hadamard rotations and ``suh``/``svh`` scales are applied.  Consequently this
test rounds the MCG *codebook labels* to finite E4M3 and then runs the normal
EXL reconstruction.  It deliberately does not round the final reconstructed
weight, which would model a different and unusable dataflow.

For each matched tensor the four reported arms are:

* MCG A16: the sealed production FP16 codebook decode;
* MCG E4M3: the same states/transforms/scales with the MCG LUT rounded to E4M3;
* SQG A16: the stored SQG labels widened from their native E4M3 bytes; and
* SQG native E4M3: the same raw E4M3 bytes supplied explicitly.

The SQG A16/native pair must close exactly.  MCG endpoint error is decomposed
as ``||e + d||^2 - ||e||^2 = ||d||^2 + 2<e,d>`` so an apparently helpful or
harmful rounding result is not mistaken for independent additive noise.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

import torch
from safetensors import safe_open


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from scripts.compare_raw_encoded_nmse import (
    DEFAULT_MCG_ROOT,
    PROJECTIONS,
    _load_reference_functions,
    _load_sqg_manifests,
    _mcg_lut,
    _payload_sha256,
    _read_tensor,
    _source_exl,
)


DEFAULT_SQG_ROOT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-contig-late-final-a025-r1"
)
DEFAULT_SQG_PREPARATION_ROOT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-contig-late-a025-r1"
)
DEFAULT_BF16_ROOT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/bf16_contiguous_late"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results/e4m3_endpoint_distortion_late_r1.json"
CODEBOOK = "sqg_xor_cheb_t12"


def round_finite_e4m3(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return widened finite-E4M3 values and their raw bytes."""

    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError("E4M3 input must be finite floating point")
    raw = values.float().to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    widened = raw.view(torch.float8_e4m3fn).to(torch.float16).contiguous()
    if not bool(torch.isfinite(widened).all()):
        raise RuntimeError("E4M3 conversion produced a non-finite label")
    return widened, raw


def error_decomposition(
    reference: torch.Tensor,
    baseline: torch.Tensor,
    endpoint: torch.Tensor,
) -> dict[str, float]:
    """Decompose the endpoint SSE change relative to ``reference``."""

    # Retain FP32 element arithmetic (the actual reference scoring regime) and
    # use FP64 only for reductions.  Expanding every 12.6M-element tensor to
    # FP64 roughly doubles memory traffic without changing the endpoint being
    # tested.
    reference32 = reference.float()
    error = baseline.float() - reference32
    delta = endpoint.float() - baseline.float()
    source_ss = float(reference32.square().sum(dtype=torch.float64).item())
    baseline_sse = float(error.square().sum(dtype=torch.float64).item())
    endpoint_sse = float((error + delta).square().sum(dtype=torch.float64).item())
    conversion_sse = float(delta.square().sum(dtype=torch.float64).item())
    cross_term = float((2.0 * error * delta).sum(dtype=torch.float64).item())
    closure_residual = endpoint_sse - baseline_sse - conversion_sse - cross_term
    tolerance = max(1.0e-7, 2.0e-12 * max(endpoint_sse, baseline_sse, 1.0))
    if abs(closure_residual) > tolerance:
        raise RuntimeError(
            "endpoint error decomposition failed: "
            f"residual={closure_residual} tolerance={tolerance}"
        )
    return {
        "source_sum_squares": source_ss,
        "baseline_sse": baseline_sse,
        "endpoint_sse": endpoint_sse,
        "endpoint_minus_baseline_sse": endpoint_sse - baseline_sse,
        "conversion_sse": conversion_sse,
        "cross_term": cross_term,
        "closure_residual": closure_residual,
    }


def _load_endpoint_functions(experiment_root: Path):
    for path in (str(experiment_root / "kquant"), str(experiment_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from kquant.sqg_e4m3 import sqg_xor_cheb_t12_bytes
    from src.glm52_fresh_sqg.reference import tensor_sha256

    return sqg_xor_cheb_t12_bytes, tensor_sha256


def _sqg_permutation(
    sqg_preparation_root: Path, layer: int, expert: int
) -> torch.Tensor:
    path = (
        sqg_preparation_root
        / f"layer_{layer:03d}"
        / "preparation"
        / "permutations"
        / f"expert-{expert:03d}.json"
    )
    payload = json.loads(path.read_text())
    result = torch.tensor(payload["new_to_old"], dtype=torch.long)
    if result.shape != (2048,) or not torch.equal(
        torch.sort(result).values, torch.arange(2048)
    ):
        raise RuntimeError(f"layer {layer} expert {expert}: invalid SQG permutation")
    return result


def _compare_layer(
    layer: int,
    mcg_root_raw: str,
    sqg_root_raw: str,
    sqg_preparation_root_raw: str,
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
    sqg_preparation_root = Path(sqg_preparation_root_raw)
    bf16_root = Path(bf16_root_raw)
    experiment_root = Path(experiment_root_raw)
    decode_exl3_weight, unpack_trellis_states = _load_reference_functions(
        experiment_root
    )
    sqg_bytes_fn, tensor_sha256 = _load_endpoint_functions(experiment_root)

    mcg_lut = _mcg_lut()
    mcg_e4m3_lut, mcg_e4m3_bytes = round_finite_e4m3(mcg_lut)
    if torch.equal(mcg_lut, mcg_e4m3_lut):
        raise RuntimeError("MCG LUT unexpectedly already lies on the E4M3 grid")

    sqg_luts: dict[int, torch.Tensor] = {}
    sqg_bytes: dict[int, torch.Tensor] = {}
    for bits in (3, 4):
        raw = sqg_bytes_fn(bits).contiguous()
        widened = raw.view(torch.float8_e4m3fn).to(torch.float16).contiguous()
        rerounded, rerounded_raw = round_finite_e4m3(widened)
        if not torch.equal(raw, rerounded_raw) or not torch.equal(widened, rerounded):
            raise RuntimeError(f"SQG K{bits} labels do not close on E4M3 bytes")
        sqg_luts[bits] = widened
        sqg_bytes[bits] = raw

    sidecar_path = mcg_root / f"r7-experts-layer-{layer:03d}.json"
    payload_path = mcg_root / f"r7-experts-layer-{layer:03d}.safetensors"
    sidecar = json.loads(sidecar_path.read_text())
    sqg_manifests = _load_sqg_manifests(sqg_root, layer)
    expert_count = 256 if limit_experts is None else min(256, limit_experts)
    rows: list[dict[str, Any]] = []

    with safe_open(payload_path, framework="pt", device="cpu") as mcg_payload:
        for expert in range(expert_count):
            mcg_permutation = torch.tensor(
                sidecar["permutations"][str(expert)]["new_to_old"],
                dtype=torch.long,
            )
            if mcg_permutation.shape != (2048,) or not torch.equal(
                torch.sort(mcg_permutation).values, torch.arange(2048)
            ):
                raise RuntimeError(
                    f"layer {layer} expert {expert}: invalid MCG permutation"
                )
            sqg_permutation = _sqg_permutation(
                sqg_preparation_root, layer, expert
            )
            sqg_manifest = sqg_manifests[expert]
            sqg_shard = (
                sqg_root
                / f"layer_{layer:03d}"
                / "experts"
                / sqg_manifest["shard"]
            )

            with safe_open(sqg_shard, framework="pt", device="cpu") as sqg_payload:
                for projection in PROJECTIONS:
                    prefix = (
                        f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                    )
                    bits = int(sidecar["bit_map"][prefix])
                    sqg_tensor = sqg_manifest["tensor_manifests"][prefix]
                    if bits != int(sqg_tensor["bits"]) or bits not in (3, 4):
                        raise RuntimeError(f"{prefix}: K3/K4 assignment differs")

                    mcg_source = sidecar["tensor_provenance"][prefix]
                    sqg_source = sqg_tensor["source"]
                    if (
                        mcg_source["bf16_sha256"]
                        != sqg_source["tensor_payload_sha256"]
                        or mcg_source["source_name"] != sqg_source["tensor_name"]
                    ):
                        raise RuntimeError(f"{prefix}: official BF16 binding differs")

                    source_hf = _read_tensor(
                        bf16_root / mcg_source["source_shard"],
                        mcg_source["source_name"],
                    )
                    mcg_source_exl = _source_exl(
                        source_hf, mcg_permutation, projection
                    )
                    sqg_source_exl = _source_exl(
                        source_hf, sqg_permutation, projection
                    )

                    mcg_states = unpack_trellis_states(
                        mcg_payload.get_tensor(prefix + ".trellis"), bits
                    )
                    vector_refs = sidecar["vector_refs"][prefix]
                    mcg_suh = mcg_payload.get_tensor(vector_refs["suh"])
                    mcg_svh = mcg_payload.get_tensor(vector_refs["svh"])
                    mcg_a16 = decode_exl3_weight(
                        mcg_states,
                        mcg_suh,
                        mcg_svh,
                        codebook_values=mcg_lut,
                        bits=bits,
                    )
                    mcg_hash = _payload_sha256(mcg_a16.half())
                    expected_mcg_hash = sidecar["roundtrip_hashes"][prefix][
                        "reconstruction_sha256"
                    ]
                    if mcg_hash != expected_mcg_hash:
                        raise RuntimeError(f"{prefix}: sealed MCG decode hash differs")
                    mcg_endpoint = decode_exl3_weight(
                        mcg_states,
                        mcg_suh,
                        mcg_svh,
                        codebook_values=mcg_e4m3_lut,
                        bits=bits,
                    )

                    sqg_states = unpack_trellis_states(
                        sqg_payload.get_tensor(prefix + ".trellis"), bits
                    )
                    sqg_suh = sqg_payload.get_tensor(prefix + ".suh")
                    sqg_svh = sqg_payload.get_tensor(prefix + ".svh")
                    sqg_a16 = decode_exl3_weight(
                        sqg_states, sqg_suh, sqg_svh, bits=bits
                    )
                    sqg_hash = tensor_sha256(sqg_a16)
                    expected_sqg_hash = sqg_tensor["decoded_closure"][
                        "decoded_exl_sha256"
                    ]
                    if sqg_hash != expected_sqg_hash:
                        raise RuntimeError(f"{prefix}: sealed SQG decode hash differs")
                    # The default SQG decoder above already widens the exact
                    # byte table verified at worker startup.  Re-running the
                    # full Hadamard/scaling transform with the same table is
                    # byte-neutral and would add one third to this 3,072-tensor
                    # CPU pass, so the native endpoint is the exact same tensor.
                    sqg_endpoint = sqg_a16

                    mcg_metrics = error_decomposition(
                        mcg_source_exl, mcg_a16, mcg_endpoint
                    )
                    sqg_metrics = error_decomposition(
                        sqg_source_exl, sqg_a16, sqg_endpoint
                    )
                    if (
                        sqg_metrics["conversion_sse"] != 0.0
                        or sqg_metrics["endpoint_minus_baseline_sse"] != 0.0
                    ):
                        raise RuntimeError(f"{prefix}: SQG endpoint is not exact")

                    rows.append(
                        {
                            "tensor": prefix,
                            "layer": layer,
                            "expert": expert,
                            "projection": projection,
                            "bits": bits,
                            "elements": int(source_hf.numel()),
                            "source_payload_sha256": mcg_source["bf16_sha256"],
                            "mcg": {
                                **mcg_metrics,
                                "a16_nmse": mcg_metrics["baseline_sse"]
                                / mcg_metrics["source_sum_squares"],
                                "e4m3_nmse": mcg_metrics["endpoint_sse"]
                                / mcg_metrics["source_sum_squares"],
                                "conversion_nmse": mcg_metrics["conversion_sse"]
                                / mcg_metrics["source_sum_squares"],
                                "a16_reconstruction_sha256": mcg_hash,
                            },
                            "sqg": {
                                **sqg_metrics,
                                "a16_nmse": sqg_metrics["baseline_sse"]
                                / sqg_metrics["source_sum_squares"],
                                "native_e4m3_nmse": sqg_metrics["endpoint_sse"]
                                / sqg_metrics["source_sum_squares"],
                                "conversion_nmse": 0.0,
                                "a16_reconstruction_sha256": sqg_hash,
                                "native_endpoint_exact": True,
                            },
                        }
                    )

                    del (
                        source_hf,
                        mcg_source_exl,
                        sqg_source_exl,
                        mcg_states,
                        mcg_a16,
                        mcg_endpoint,
                        sqg_states,
                        sqg_a16,
                        sqg_endpoint,
                    )
    return rows


def _summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    if not values:
        raise ValueError("cannot summarize an empty endpoint comparison")
    source_ss = sum(row["mcg"]["source_sum_squares"] for row in values)
    mcg_a16_sse = sum(row["mcg"]["baseline_sse"] for row in values)
    mcg_endpoint_sse = sum(row["mcg"]["endpoint_sse"] for row in values)
    mcg_conversion_sse = sum(row["mcg"]["conversion_sse"] for row in values)
    mcg_cross = sum(row["mcg"]["cross_term"] for row in values)
    sqg_a16_sse = sum(row["sqg"]["baseline_sse"] for row in values)
    sqg_endpoint_sse = sum(row["sqg"]["endpoint_sse"] for row in values)
    endpoint_delta = mcg_endpoint_sse - mcg_a16_sse
    closure = endpoint_delta - mcg_conversion_sse - mcg_cross
    return {
        "tensors": len(values),
        "elements": sum(row["elements"] for row in values),
        "source_sum_squares": source_ss,
        "nmse": {
            "mcg_a16": mcg_a16_sse / source_ss,
            "mcg_e4m3": mcg_endpoint_sse / source_ss,
            "sqg_a16": sqg_a16_sse / source_ss,
            "sqg_native_e4m3": sqg_endpoint_sse / source_ss,
        },
        "mcg_endpoint": {
            "bf16_error_change_percent": endpoint_delta / mcg_a16_sse * 100.0,
            "conversion_energy_over_bf16_percent": mcg_conversion_sse
            / mcg_a16_sse
            * 100.0,
            "cross_term_over_bf16_percent": mcg_cross / mcg_a16_sse * 100.0,
            "conversion_nmse": mcg_conversion_sse / source_ss,
            "cross_term_normalized": mcg_cross / source_ss,
            "decomposition_closure_residual": closure,
            "tensors_bf16_error_worsened": sum(
                row["mcg"]["endpoint_sse"] > row["mcg"]["baseline_sse"]
                for row in values
            ),
            "tensors_bf16_error_improved": sum(
                row["mcg"]["endpoint_sse"] < row["mcg"]["baseline_sse"]
                for row in values
            ),
        },
        "sqg_endpoint": {
            "all_native_endpoints_exact": all(
                row["sqg"]["native_endpoint_exact"] for row in values
            ),
            "maximum_conversion_sse": max(
                row["sqg"]["conversion_sse"] for row in values
            ),
        },
        "paired": {
            "median_mcg_endpoint_error_ratio": statistics.median(
                row["mcg"]["endpoint_sse"] / row["mcg"]["baseline_sse"]
                for row in values
            ),
            "geometric_mean_mcg_endpoint_error_ratio": math.exp(
                statistics.fmean(
                    math.log(
                        row["mcg"]["endpoint_sse"]
                        / row["mcg"]["baseline_sse"]
                    )
                    for row in values
                )
            ),
        },
    }


def _group(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: _summarize(group) for key, group in sorted(groups.items())}


def _render(result: dict[str, Any]) -> str:
    overall = result["summary"]["overall"]
    nmse = overall["nmse"]
    endpoint = overall["mcg_endpoint"]
    lines = [
        "# Test 8a: matched E4M3 endpoint distortion",
        "",
        "This is a CPU-only weight-operand endpoint test, not KLD or W4A8 activation quality.",
        "",
        "## Overall",
        "",
        f"- Matched tensors: **{overall['tensors']:,}**",
        f"- MCG A16 NMSE: **`{nmse['mcg_a16']:.9e}`**",
        f"- MCG label-to-E4M3 NMSE: **`{nmse['mcg_e4m3']:.9e}`**",
        f"- MCG BF16-error change: **`{endpoint['bf16_error_change_percent']:+.4f}%`**",
        f"- MCG conversion energy relative to its A16 BF16 error: **`{endpoint['conversion_energy_over_bf16_percent']:+.4f}%`**",
        f"- MCG cross-term relative to its A16 BF16 error: **`{endpoint['cross_term_over_bf16_percent']:+.4f}%`**",
        f"- SQG A16 NMSE: **`{nmse['sqg_a16']:.9e}`**",
        f"- SQG native-E4M3 NMSE: **`{nmse['sqg_native_e4m3']:.9e}`**",
        f"- SQG exact native endpoints: **{overall['sqg_endpoint']['all_native_endpoints_exact']}**",
        "",
        "## By rate",
        "",
        "| Rate | Tensors | MCG A16 | MCG E4M3 | MCG error change | SQG native E4M3 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for bits, summary in result["summary"]["by_bits"].items():
        n = summary["nmse"]
        e = summary["mcg_endpoint"]
        lines.append(
            f"| K{bits} | {summary['tensors']:,} | {n['mcg_a16']:.9e} | "
            f"{n['mcg_e4m3']:.9e} | {e['bf16_error_change_percent']:+.4f}% | "
            f"{n['sqg_native_e4m3']:.9e} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The MCG conversion-energy term is the cost of moving its realized labels to E4M3. The BF16-error change also includes a signed cross-term with the existing quantization error, so it need not equal the conversion energy and can even improve individual tensors by chance.",
            "",
            "SQG's zero conversion term establishes only its exact weight-label endpoint. It does not remove E4M3 activation error, prove W4A8 KLD parity, or demonstrate a speedup.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcg-root", type=Path, default=DEFAULT_MCG_ROOT)
    parser.add_argument("--sqg-root", type=Path, default=DEFAULT_SQG_ROOT)
    parser.add_argument(
        "--sqg-preparation-root",
        type=Path,
        default=DEFAULT_SQG_PREPARATION_ROOT,
    )
    parser.add_argument("--bf16-root", type=Path, default=DEFAULT_BF16_ROOT)
    parser.add_argument(
        "--experiment-root", type=Path, default=EXPERIMENT_ROOT
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[74, 75, 76, 77])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads-per-worker", type=int, default=10)
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
                str(args.sqg_preparation_root.resolve()),
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
        "schema": "glm52-e4m3-endpoint-distortion-v1",
        "device": "cpu",
        "gpu_visible": False,
        "layers": sorted(args.layers),
        "mcg_root": str(args.mcg_root.resolve()),
        "sqg_root": str(args.sqg_root.resolve()),
        "sqg_preparation_root": str(args.sqg_preparation_root.resolve()),
        "bf16_root": str(args.bf16_root.resolve()),
        "comparison_contract": {
            "mcg_endpoint": "regularized MCG LUT rounded RNE to finite E4M3 before Hadamard/scales",
            "sqg_endpoint": "native raw sqg_xor_cheb_t12 E4M3 label bytes before Hadamard/scales",
            "final_reconstructed_weight_rounding_forbidden": True,
            "matched_official_bf16_payload_required": True,
            "matched_k3_k4_assignment_required": True,
            "all_a16_decode_hashes_closed": True,
            "sqg_a16_native_e4m3_exact_required": True,
        },
        "mcg_lut": {
            "fp16_unique_labels": int(torch.unique(_mcg_lut()).numel()),
            "e4m3_unique_labels": int(
                torch.unique(round_finite_e4m3(_mcg_lut())[1]).numel()
            ),
        },
        "summary": {
            "overall": _summarize(rows),
            "by_bits": _group(rows, "bits"),
            "by_projection": _group(rows, "projection"),
            "by_layer": _group(rows, "layer"),
        },
        "tensors": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report_path = args.output.with_suffix(".md")
    report_path.write_text(_render(result))
    print(_render(result), flush=True)
    print(f"json: {args.output}", flush=True)
    print(f"report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
