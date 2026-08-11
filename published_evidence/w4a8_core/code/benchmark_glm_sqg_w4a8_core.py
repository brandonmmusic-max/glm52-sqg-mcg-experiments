"""Test 8c-core: native SQG W4A8 versus SQG W4A16 on GLM-5.2 shapes.

This benchmark consumes checkpoint-native K3/K4 tensors from one encoded GLM
layer and post-LayerNorm hidden rows plus exact top-8 routes from its capture.
It never materializes a dense weight.  Each arm runs the same compact tensor,
stored transforms, and routed input rows.  The W4A8 arm adds the real input
Hadamard/scaling, MXFP8 row quantization, direct SQG-to-E4M3 decode, FP8 MMA,
and output transform.  The W4A16 arm is the compact FP16 control used by the
GLM trellis runtime (the BF16 capture rows cross the runtime's FP16 boundary
before either arm).

This is deliberately a dense *core* benchmark, not a routed-layer kernel.  It
uses actual per-expert route-row histograms and reports a clearly labelled
serial-call projection.  It cannot establish serving speed until a route-
packed GLM layer path is timed end to end.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F

from b12x.gemm import trellis_linear
from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm


_RESULT_KIND = "b12x_glm52_sqg_w4a8_core_benchmark"
_SCHEMA_VERSION = 1
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_GLOBAL_M_DEFAULT = (1, 128, 512, 1024, 2048, 3072, 4096)
_QUANTILES_DEFAULT = (0.5, 0.9, 1.0)
_ROLE_IDS = {"fit": 0, "selection": 1, "holdout": 2}
_BIT_KEY = re.compile(r"\.experts\.(?P<expert>\d+)\.(?P<projection>[^.]+)$")


@dataclass(frozen=True)
class RouteCase:
    global_m: int
    projection: str
    bits: int
    route_quantile: float
    expert: int
    expert_rows: int
    row_indices: tuple[int, ...]

    @property
    def key(self) -> str:
        q = int(round(self.route_quantile * 100))
        return (
            f"m{self.global_m}-{self.projection}-k{self.bits}-"
            f"p{q:03d}-e{self.expert:03d}-r{self.expert_rows}"
        )


def _parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not parsed or min(parsed) <= 0:
        raise argparse.ArgumentTypeError("values must be positive integers")
    return parsed


def _parse_quantiles(value: str) -> tuple[float, ...]:
    parsed = tuple(
        dict.fromkeys(float(item.strip()) for item in value.split(",") if item.strip())
    )
    if not parsed or min(parsed) <= 0.0 or max(parsed) > 1.0:
        raise argparse.ArgumentTypeError("quantiles must lie in (0, 1]")
    return parsed


@functools.lru_cache(maxsize=None)
def _sha256(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _load_rate_map(manifest_path: Path, *, layer: int) -> dict[int, dict[str, int]]:
    payload = json.loads(manifest_path.read_text())
    bit_map = payload.get("bit_map")
    if not isinstance(bit_map, dict):
        raise ValueError(f"{manifest_path} has no bit_map object")
    rates: dict[int, dict[str, int]] = {expert: {} for expert in range(256)}
    prefix = f"model.layers.{layer}.mlp.experts."
    for key, raw_bits in bit_map.items():
        if not key.startswith(prefix):
            continue
        match = _BIT_KEY.search(key)
        if match is None:
            continue
        expert = int(match.group("expert"))
        projection = match.group("projection")
        bits = int(raw_bits)
        if expert not in rates or projection not in _PROJECTIONS:
            raise ValueError(f"unexpected bit-map key {key!r}")
        if bits not in (3, 4):
            raise ValueError(f"Test 8c requires K3/K4 only, got K{bits} for {key}")
        rates[expert][projection] = bits
    missing = [
        (expert, projection)
        for expert in range(256)
        for projection in _PROJECTIONS
        if projection not in rates[expert]
    ]
    if missing:
        raise ValueError(f"bit map is missing {len(missing)} expert tensors")
    histogram = {
        bits: sum(
            rates[expert][projection] == bits
            for expert in range(256)
            for projection in _PROJECTIONS
        )
        for bits in (3, 4)
    }
    if histogram != {3: 384, 4: 384}:
        raise ValueError(
            "layer must preserve the exact topology-neutral K3/K4 census; "
            f"got {histogram}"
        )
    return rates


def _nearest_quantile(values: np.ndarray, quantile: float) -> int:
    if values.size == 0:
        raise ValueError("cannot take a quantile of an empty route set")
    try:
        return int(np.quantile(values, quantile, method="nearest"))
    except TypeError:  # NumPy < 1.22
        return int(np.quantile(values, quantile, interpolation="nearest"))


def _route_summary(counts: np.ndarray) -> dict[str, Any]:
    active = counts[counts > 0]
    result: dict[str, Any] = {
        "counts": [int(value) for value in counts],
        "active_experts": int(active.size),
        "inactive_experts": int((counts == 0).sum()),
        "total_routes": int(counts.sum()),
    }
    if active.size:
        result.update(
            {
                "active_min": int(active.min()),
                "active_p50": _nearest_quantile(active, 0.5),
                "active_p90": _nearest_quantile(active, 0.9),
                "active_max": int(active.max()),
            }
        )
    return result


def _build_route_plan(
    topk_ids: np.ndarray,
    source_rows: np.ndarray,
    rates: dict[int, dict[str, int]],
    *,
    global_ms: Iterable[int],
    quantiles: Iterable[float],
) -> tuple[list[RouteCase], dict[str, Any]]:
    topk_ids = np.asarray(topk_ids)
    source_rows = np.asarray(source_rows, dtype=np.int64)
    if topk_ids.ndim != 2 or int(topk_ids.shape[1]) != 8:
        raise ValueError(f"topk_ids must have shape [rows,8], got {topk_ids.shape}")
    if source_rows.ndim != 1 or source_rows.size != topk_ids.shape[0]:
        raise ValueError("source_rows must map every topk row to one capture row")
    cases: list[RouteCase] = []
    histograms: dict[str, Any] = {}
    for global_m in global_ms:
        if global_m > topk_ids.shape[0]:
            raise ValueError(
                f"global M={global_m} exceeds available route rows {topk_ids.shape[0]}"
            )
        prefix = np.asarray(topk_ids[:global_m], dtype=np.int64)
        counts = np.bincount(prefix.reshape(-1), minlength=256)[:256]
        global_record: dict[str, Any] = {
            "all_experts": _route_summary(counts),
            "by_projection_rate": {},
        }
        for projection in _PROJECTIONS:
            for bits in (3, 4):
                expert_ids = np.array(
                    [
                        expert
                        for expert in range(256)
                        if rates[expert][projection] == bits
                    ],
                    dtype=np.int64,
                )
                rate_counts = counts[expert_ids]
                label = f"{projection}_k{bits}"
                global_record["by_projection_rate"][label] = {
                    "expert_ids": expert_ids.tolist(),
                    **_route_summary(rate_counts),
                }
                active_mask = rate_counts > 0
                active_experts = expert_ids[active_mask]
                active_counts = rate_counts[active_mask]
                if active_counts.size == 0:
                    continue
                for quantile in quantiles:
                    target = _nearest_quantile(active_counts, float(quantile))
                    distance = np.abs(active_counts - target)
                    expert = int(active_experts[np.flatnonzero(distance == distance.min())[0]])
                    local_rows = np.flatnonzero(np.any(prefix == expert, axis=1))
                    if local_rows.size != int(counts[expert]):
                        raise ValueError(
                            "top-8 routes contain a duplicate expert within one token; "
                            "the dense core contract requires one row per expert route"
                        )
                    rows = source_rows[local_rows]
                    cases.append(
                        RouteCase(
                            global_m=int(global_m),
                            projection=projection,
                            bits=bits,
                            route_quantile=float(quantile),
                            expert=expert,
                            expert_rows=int(rows.size),
                            row_indices=tuple(int(row) for row in rows),
                        )
                    )
        histograms[str(global_m)] = global_record
    return cases, histograms


def _tensor_prefix(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}"


def _expert_path(sqg_root: Path, layer: int, expert: int) -> Path:
    return (
        sqg_root
        / f"layer_{layer:03d}"
        / "experts"
        / f"layer-{layer:03d}-expert-{expert:03d}.safetensors"
    )


def _load_weight(
    sqg_root: Path,
    *,
    layer: int,
    expert: int,
    projection: str,
    device: torch.device,
):
    path = _expert_path(sqg_root, layer, expert)
    prefix = _tensor_prefix(layer, expert, projection)
    with safe_open(path, framework="pt", device="cpu") as handle:
        trellis = handle.get_tensor(f"{prefix}.trellis").to(device)
        suh = handle.get_tensor(f"{prefix}.suh").to(device)
        svh = handle.get_tensor(f"{prefix}.svh").to(device)
    prepared = trellis_linear.prepare_weight(
        trellis,
        suh,
        svh,
        codebook="sqg_xor_cheb_t12",
        params_dtype=torch.float16,
    )
    return prepared, path


def _load_hidden_rows(
    hidden_words: np.memmap,
    row_indices: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    words = np.array(hidden_words[np.asarray(row_indices, dtype=np.int64)], copy=True)
    # GLM checkpoints/captures are BF16, but the deployed trellis A16 path
    # prepares and executes this boundary in FP16.  Use that same boundary in
    # both benchmark arms so the comparison changes only activation/weight MMA.
    source = (
        torch.from_numpy(words)
        .view(torch.bfloat16)
        .to(device=device, dtype=torch.float16)
    )
    return source.contiguous()


def _buffers_a16(x: torch.Tensor, out_features: int) -> dict[str, torch.Tensor]:
    m, _ = x.shape
    # The dense planner uses M48 or M64 route tiles. Supply the larger rounded
    # route extent so graph capture cannot fall back to an internal allocation.
    route_slots = max(((m + 47) // 48) * 48, ((m + 63) // 64) * 64)
    return {
        "output": torch.empty((m, out_features), dtype=x.dtype, device=x.device),
        "gemm_output": torch.empty(
            (m, out_features), dtype=torch.float16, device=x.device
        ),
        "c_tmp": torch.empty(
            (max(route_slots * out_features, 1),),
            dtype=torch.float32,
            device=x.device,
        ),
        "rotated_f16": torch.empty(
            x.shape, dtype=torch.float16, device=x.device
        ),
    }


def _buffers_a8(x: torch.Tensor, out_features: int) -> dict[str, Any]:
    m, k = (int(value) for value in x.shape)
    return {
        "output": torch.empty((m, out_features), dtype=x.dtype, device=x.device),
        "rotated_f16": torch.empty(
            x.shape, dtype=torch.float16, device=x.device
        ),
        "quantized": empty_mxfp8_rows_for_dense_gemm(m, k, device=x.device),
        "gemm_output_f16": torch.empty(
            (m, out_features), dtype=torch.float16, device=x.device
        ),
    }


def _make_down_source(
    hidden: torch.Tensor,
    sqg_root: Path,
    *,
    layer: int,
    expert: int,
    device: torch.device,
    upstream: str,
) -> tuple[torch.Tensor, list[object], dict[str, Any]]:
    keepalive: list[object] = []
    outputs: dict[str, torch.Tensor] = {}
    rates: dict[str, int] = {}
    for projection in ("gate_proj", "up_proj"):
        prepared, path = _load_weight(
            sqg_root,
            layer=layer,
            expert=expert,
            projection=projection,
            device=device,
        )
        rates[projection] = int(prepared.trellis_bits)
        if upstream == "w4a8":
            buffers = _buffers_a8(hidden, int(prepared.out_features))
            output = trellis_linear.run_uniform_w4a8(hidden, prepared, **buffers)
        else:
            buffers = _buffers_a16(hidden, int(prepared.out_features))
            output = trellis_linear.run(hidden, prepared, **buffers)
        outputs[projection] = output.clone()
        keepalive.extend((prepared, buffers, output, path))
    torch.cuda.synchronize(device)
    activation = (F.silu(outputs["gate_proj"].float()) * outputs["up_proj"].float())
    activation = activation.to(torch.float16).contiguous()
    return activation, keepalive, {"upstream": upstream, "rates": rates}


def _comparison(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    cand = candidate.float()
    ref = reference.float()
    delta = cand - ref
    ref_energy = max(float(ref.square().sum()), torch.finfo(torch.float32).tiny)
    denominator = max(
        float(torch.linalg.vector_norm(cand)) * float(torch.linalg.vector_norm(ref)),
        torch.finfo(torch.float32).tiny,
    )
    return {
        "nmse_w4a8_vs_w4a16": float(delta.square().sum()) / ref_energy,
        "rmse": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "cosine_similarity": float((cand * ref).sum()) / denominator,
    }


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90_us": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _ratio_bootstrap(
    w4a16: list[float],
    w4a8: list[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if len(w4a16) != len(w4a8) or not w4a16:
        raise ValueError("timing arms must have the same nonzero sample count")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(w4a16), size=(replicates, len(w4a16)))
    a16 = np.asarray(w4a16, dtype=np.float64)
    a8 = np.asarray(w4a8, dtype=np.float64)
    ratios = np.median(a16[indices], axis=1) / np.median(a8[indices], axis=1)
    low, high = np.quantile(ratios, [0.025, 0.975])
    return {
        "w4a16_over_w4a8_speedup": statistics.median(w4a16)
        / statistics.median(w4a8),
        "bootstrap_ci95_low": float(low),
        "bootstrap_ci95_high": float(high),
        "bootstrap_replicates": int(replicates),
    }


def _time_abba(
    graphs: dict[str, torch.cuda.CUDAGraph],
    *,
    samples_per_arm: int,
) -> dict[str, list[float]]:
    if samples_per_arm < 200 or samples_per_arm % 2:
        raise ValueError("samples_per_arm must be even and at least 200")
    samples = {"w4a16": [], "w4a8": []}
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    for cycle in range(samples_per_arm // 2):
        order = (
            ("w4a16", "w4a8", "w4a8", "w4a16")
            if cycle % 2 == 0
            else ("w4a8", "w4a16", "w4a16", "w4a8")
        )
        for arm in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[arm].replay()
            end.record()
            events.append((arm, start, end))
    torch.cuda.synchronize()
    for arm, start, end in events:
        samples[arm].append(float(start.elapsed_time(end)) * 1000.0)
    return samples


def _capture_case(
    source: torch.Tensor,
    weight,
    *,
    warmup: int,
) -> tuple[dict[str, torch.cuda.CUDAGraph], list[object], dict[str, Any]]:
    a16_buffers = _buffers_a16(source, int(weight.out_features))
    a8_buffers = _buffers_a8(source, int(weight.out_features))
    a16_eager = trellis_linear.run(source, weight, **a16_buffers).clone()
    a8_eager = trellis_linear.run_uniform_w4a8(source, weight, **a8_buffers).clone()
    torch.cuda.synchronize(source.device)
    if not bool(torch.isfinite(a16_eager).all()) or not bool(torch.isfinite(a8_eager).all()):
        raise RuntimeError("non-finite output before graph capture")

    graph_a16 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_a16):
        a16_graph_output = trellis_linear.run(source, weight, **a16_buffers)
    graph_a8 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_a8):
        a8_graph_output = trellis_linear.run_uniform_w4a8(source, weight, **a8_buffers)
    for _ in range(warmup):
        graph_a16.replay()
        graph_a8.replay()
    torch.cuda.synchronize(source.device)
    if not torch.equal(a16_graph_output, a16_eager):
        raise RuntimeError("W4A16 eager and graph outputs differ")
    if not torch.equal(a8_graph_output, a8_eager):
        raise RuntimeError("W4A8 eager and graph outputs differ")
    keepalive: list[object] = [
        source,
        weight,
        a16_buffers,
        a8_buffers,
        a16_eager,
        a8_eager,
        a16_graph_output,
        a8_graph_output,
    ]
    return (
        {"w4a16": graph_a16, "w4a8": graph_a8},
        keepalive,
        {
            **_comparison(a8_eager, a16_eager),
            "w4a16_eager_graph_bit_exact": True,
            "w4a8_eager_graph_bit_exact": True,
        },
    )


def _end_to_end_speedup(moe_speedup: float, moe_fraction: float) -> float:
    if moe_speedup <= 0.0 or not 0.0 <= moe_fraction <= 1.0:
        raise ValueError("invalid Amdahl projection inputs")
    return 1.0 / ((1.0 - moe_fraction) + moe_fraction / moe_speedup)


def _project_serial_core(
    results: list[dict[str, Any]],
    histograms: dict[str, Any],
    *,
    moe_fraction: float,
) -> dict[str, Any]:
    projections: dict[str, Any] = {}
    for global_m, histogram in histograms.items():
        selected = [row for row in results if row["case"]["global_m"] == int(global_m)]
        arm_totals = {"w4a16": 0.0, "w4a8": 0.0}
        bindings: list[dict[str, Any]] = []
        for projection in _PROJECTIONS:
            for bits in (3, 4):
                label = f"{projection}_k{bits}"
                group = [
                    row
                    for row in selected
                    if row["case"]["projection"] == projection
                    and row["case"]["bits"] == bits
                ]
                counts = histogram["by_projection_rate"][label]["counts"]
                if not group:
                    continue
                for count in counts:
                    if count <= 0:
                        continue
                    nearest = min(
                        group,
                        key=lambda row: abs(row["case"]["expert_rows"] - count),
                    )
                    arm_totals["w4a16"] += nearest["timing"]["w4a16"]["median_us"]
                    arm_totals["w4a8"] += nearest["timing"]["w4a8"]["median_us"]
                    bindings.append(
                        {
                            "projection": projection,
                            "bits": bits,
                            "route_rows": count,
                            "timed_case": nearest["case"]["key"],
                        }
                    )
        if arm_totals["w4a8"] <= 0.0:
            continue
        speedup = arm_totals["w4a16"] / arm_totals["w4a8"]
        projections[global_m] = {
            "method": (
                "serial sum over active expert/projection calls; each actual route "
                "count is mapped to the nearest measured quantile case of the same "
                "projection and rate"
            ),
            "w4a16_us": arm_totals["w4a16"],
            "w4a8_us": arm_totals["w4a8"],
            "moe_core_speedup": speedup,
            "amdahl_end_to_end_speedup_at_declared_moe_fraction": _end_to_end_speedup(
                speedup, moe_fraction
            ),
            "moe_fraction": moe_fraction,
            "not_serving_acceptance": True,
            "bindings": bindings,
        }
    return projections


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _git_commit(worktree: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqg-root", type=Path, required=True)
    parser.add_argument("--capture-layer", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=77)
    parser.add_argument("--role", choices=("all", *_ROLE_IDS), default="all")
    parser.add_argument(
        "--global-m", type=_parse_int_list, default=_GLOBAL_M_DEFAULT
    )
    parser.add_argument(
        "--route-quantiles", type=_parse_quantiles, default=_QUANTILES_DEFAULT
    )
    parser.add_argument("--down-upstream", choices=("w4a8", "w4a16"), default="w4a8")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--moe-prefill-fraction", type=float, default=0.31)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="seal only the CPU route/tensor plan; do not initialize CUDA",
    )
    args = parser.parse_args()

    if args.samples < 200 or args.samples % 2:
        raise SystemExit("--samples must be even and at least 200")
    if args.warmup < 1 or args.bootstrap_replicates < 1:
        raise SystemExit("warmup and bootstrap replicates must be positive")
    if not 0.0 <= args.moe_prefill_fraction <= 1.0:
        raise SystemExit("--moe-prefill-fraction must lie in [0,1]")

    manifest_path = (
        args.sqg_root
        / f"layer_{args.layer:03d}"
        / "final"
        / f"fresh-sqg-layer-{args.layer:03d}.json"
    )
    capture_manifest_path = args.capture_layer / "layer_manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text())
    if int(capture_manifest.get("layer", -1)) != args.layer:
        raise SystemExit("capture layer manifest does not match --layer")
    if int(capture_manifest.get("hidden", -1)) != 6144:
        raise SystemExit("Test 8c GLM contract requires hidden size 6144")
    rates = _load_rate_map(manifest_path, layer=args.layer)
    total_rows = int(capture_manifest["tokens"])
    topk = int(capture_manifest["topk"])
    if topk != 8:
        raise SystemExit(f"Test 8c GLM contract requires top-8, got {topk}")
    all_ids = np.memmap(
        args.capture_layer / "topk_ids.u8.bin",
        mode="r",
        dtype="u1",
        shape=(total_rows, topk),
    )
    if args.role == "all":
        source_rows = np.arange(total_rows, dtype=np.int64)
    else:
        role_ids = np.memmap(
            args.capture_layer / "role_ids.u8.bin",
            mode="r",
            dtype="u1",
            shape=(total_rows,),
        )
        source_rows = np.flatnonzero(role_ids == _ROLE_IDS[args.role]).astype(np.int64)
    selected_ids = np.asarray(all_ids[source_rows])
    cases, histograms = _build_route_plan(
        selected_ids,
        source_rows,
        rates,
        global_ms=args.global_m,
        quantiles=args.route_quantiles,
    )

    worktree = Path(__file__).resolve().parents[1]
    payload: dict[str, Any] = {
        "kind": _RESULT_KIND,
        "schema_version": _SCHEMA_VERSION,
        "status": "planned" if args.dry_run else "running",
        "provenance": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "worktree": str(worktree),
            "git_commit": _git_commit(worktree),
            "sqg_root": str(args.sqg_root.resolve()),
            "layer_manifest": str(manifest_path.resolve()),
            "layer_manifest_sha256": _sha256(manifest_path),
            "capture_layer": str(args.capture_layer.resolve()),
            "capture_manifest": str(capture_manifest_path.resolve()),
            "capture_manifest_sha256": _sha256(capture_manifest_path),
            "capture_run_uuid": capture_manifest.get("capture_run_uuid"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "contract": {
            "layer": args.layer,
            "hidden": 6144,
            "intermediate": 2048,
            "num_experts": 256,
            "topk": 8,
            "role": args.role,
            "global_m": list(args.global_m),
            "route_quantiles": list(args.route_quantiles),
            "samples_per_arm": args.samples,
            "timing_order": "balanced ABBA with alternating cycle orientation",
            "warmup": args.warmup,
            "cuda_graph_replay": True,
            "caller_owned_fixed_workspace": True,
            "weight_path": "checkpoint-native compact SQG K3/K4 in both arms",
            "capture_to_runtime_boundary": "BF16 capture rows cast once to FP16 for both arms",
            "w4a16_path": "input transform -> compact decode -> FP16 MMA -> output transform",
            "w4a8_path": (
                "input transform -> MXFP8 K32 quantization -> direct compact SQG "
                "decode to E4M3 MMA registers -> FP8 MMA -> output transform"
            ),
            "dense_weight_materialization": False,
            "down_activation_source": args.down_upstream,
            "limitations": [
                "dense per-tensor core, not route-packed layer execution",
                "serial-call projection does not model routed scheduling or fusion",
                "one measured tensor is selected at each route-count quantile",
                "down activation is generated outside the timed down-projection graph",
            ],
        },
        "rate_census": {
            str(bits): sum(
                rates[expert][projection] == bits
                for expert in range(256)
                for projection in _PROJECTIONS
            )
            for bits in (3, 4)
        },
        "route_histograms": histograms,
        "cases": [{**asdict(case), "key": case.key} for case in cases],
        "results": [],
    }
    if args.dry_run:
        _write_new(args.output, payload)
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required unless --dry-run is supplied")
    major, minor = torch.cuda.get_device_capability()
    if major != 12 or minor not in (0, 1):
        raise SystemExit(f"SM120/SM121 is required, got SM{major}{minor}")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    payload["device"] = {
        "name": properties.name,
        "sm": f"{major}{minor}",
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory": properties.total_memory,
        "uuid": str(getattr(properties, "uuid", "unavailable")),
    }
    kernel_sources = {
        "w4a8": worktree / "b12x/gemm/trellis_linear/w4a8.py",
        "w4a16": worktree / "b12x/moe/_shared/kernels/w4a16/kernel.py",
    }
    payload["kernel_identity"] = {
        arm: {
            "path": str(path),
            "sha256": _sha256(path),
        }
        for arm, path in kernel_sources.items()
    }
    payload["kernel_identity"]["w4a8"].update(
        {
            "entry_point": "trellis_linear.run_uniform_w4a8",
            "compile_spec": "gemm.trellis_w4a8_dense:v3",
            "native_uniform_bits": [3, 4],
        }
    )
    payload["kernel_identity"]["w4a16"].update(
        {"entry_point": "trellis_linear.run", "compact_native": True}
    )

    hidden_words = np.memmap(
        args.capture_layer / "hidden.bf16.bin",
        mode="r",
        dtype="<u2",
        shape=(total_rows, 6144),
    )
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        hidden = _load_hidden_rows(hidden_words, case.row_indices, device=device)
        down_meta = None
        if case.projection == "down_proj":
            source, upstream_keepalive, down_meta = _make_down_source(
                hidden,
                args.sqg_root,
                layer=args.layer,
                expert=case.expert,
                device=device,
                upstream=args.down_upstream,
            )
            # The generated activation owns its bytes; upstream temporaries
            # only need to survive through this construction call.
            del upstream_keepalive
        else:
            source = hidden
        weight, weight_path = _load_weight(
            args.sqg_root,
            layer=args.layer,
            expert=case.expert,
            projection=case.projection,
            device=device,
        )
        if int(weight.trellis_bits) != case.bits:
            raise RuntimeError("planned and loaded tensor rates disagree")
        expected_shape = (
            (6144, 2048)
            if case.projection != "down_proj"
            else (2048, 6144)
        )
        if (int(weight.in_features), int(weight.out_features)) != expected_shape:
            raise RuntimeError(
                f"{case.key} has unexpected GLM shape "
                f"{weight.in_features}x{weight.out_features}"
            )
        graphs, case_keepalive, numerical = _capture_case(
            source,
            weight,
            warmup=args.warmup,
        )
        samples = _time_abba(graphs, samples_per_arm=args.samples)
        row = {
            "case": {**asdict(case), "key": case.key},
            "source": {
                "shape": list(source.shape),
                "dtype": str(source.dtype),
                "row_indices_sha256": _array_sha256(
                    np.asarray(case.row_indices, dtype=np.int64)
                ),
                "rms": float(source.float().square().mean().sqrt()),
                "max_abs": float(source.float().abs().max()),
                "down": down_meta,
            },
            "weight": {
                "path": str(weight_path.resolve()),
                "file_sha256": _sha256(weight_path),
                "shape_k_n": [weight.in_features, weight.out_features],
                "bits": weight.trellis_bits,
                "trellis_num_bytes": weight.trellis.numel()
                * weight.trellis.element_size(),
                "pair_kind": weight.trellis_pair_kind,
                "dense_materialized": False,
            },
            "numerical": numerical,
            "timing": {
                "w4a16": {**_summary(samples["w4a16"]), "samples_us": samples["w4a16"]},
                "w4a8": {**_summary(samples["w4a8"]), "samples_us": samples["w4a8"]},
                "ratio": _ratio_bootstrap(
                    samples["w4a16"],
                    samples["w4a8"],
                    replicates=args.bootstrap_replicates,
                    seed=args.seed + index,
                ),
            },
        }
        results.append(row)
        # Graphs and tensor storage are case-local. Retaining all 126 planned
        # cases would consume several GiB and would turn allocator pressure
        # into an unintended benchmark variable.
        del graphs, case_keepalive, hidden, source, weight
    payload["results"] = results
    payload["serial_core_projection"] = _project_serial_core(
        results,
        histograms,
        moe_fraction=args.moe_prefill_fraction,
    )
    payload["status"] = "complete"
    payload["conclusion_scope"] = (
        "Real GLM K3/K4 compact-core evidence. A serving migration decision "
        "still requires a route-packed layer benchmark and end-to-end prefill."
    )
    _write_new(args.output, payload)


if __name__ == "__main__":
    main()
