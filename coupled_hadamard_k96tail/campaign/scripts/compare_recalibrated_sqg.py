#!/usr/bin/env python3
"""Compare MCG, current SQG, and expert-local-H13 SQG from packed bytes.

The primary metric is routed expert-function NMSE on fit, selection, and
holdout rows.  It includes gate, up, SwiGLU, and down together and weights each
expert row by the square of the applied router gate.  Raw weight NMSE and
separate gate/up routed-output errors are also reported.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_hessian_weighted_nmse import (  # noqa: E402
    DEFAULT_CAPTURE_ROOT,
    _load_hidden_chunk,
    _load_reference_functions,
    _read_source_triplet,
)
from scripts.compare_raw_encoded_nmse import (  # noqa: E402
    DEFAULT_BF16_ROOT,
    DEFAULT_MCG_ROOT,
    DEFAULT_SQG_ROOT,
    _load_sqg_manifests,
    _mcg_lut,
)


DEFAULT_CORRECTED_ROOT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-expert-h13-oas-r1"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/recalibrated_sqg_vs_mcg.json"
LAYERS = (6, 28, 52, 77)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ARMS = ("mcg", "current_sqg", "corrected_sqg")
ROLES = {"fit": 0, "selection": 1, "holdout": 2}
HIDDEN = 6144
TOPK = 8


def _route_indices(
    capture_layer: Path,
) -> tuple[np.memmap, np.memmap, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]:
    role_path = capture_layer / "role_ids.u8.bin"
    total = role_path.stat().st_size
    role_ids = np.memmap(role_path, mode="r", dtype="u1", shape=(total,))
    topk_ids = np.memmap(
        capture_layer / "topk_ids.u8.bin",
        mode="r",
        dtype="u1",
        shape=(total, TOPK),
    )
    topk_weights = np.memmap(
        capture_layer / "topk_weights.f32le.bin",
        mode="r",
        dtype="<f4",
        shape=(total, TOPK),
    )
    result: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for role, role_id in ROLES.items():
        selected = np.flatnonzero(np.asarray(role_ids) == role_id).astype(
            np.int64, copy=False
        )
        ids = np.asarray(topk_ids[selected]).reshape(-1)
        rows = np.repeat(selected, TOPK)
        slots = np.tile(np.arange(TOPK, dtype=np.int16), selected.size)
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        bounds = np.searchsorted(sorted_ids, np.arange(257), side="left")
        result[role] = {
            expert: (
                rows[order[bounds[expert] : bounds[expert + 1]]],
                slots[order[bounds[expert] : bounds[expert + 1]]],
            )
            for expert in range(256)
        }
    hidden_words = np.memmap(
        capture_layer / "hidden.bf16.bin",
        mode="r",
        dtype="<u2",
        shape=(total, HIDDEN),
    )
    return hidden_words, topk_weights, result


def _permutation_inverse(
    sqg_reference_root: Path,
    layer: int,
    expert: int,
    device: torch.device,
) -> torch.Tensor:
    path = (
        sqg_reference_root
        / f"layer_{layer:03d}/preparation/permutations/expert-{expert:03d}.json"
    )
    value = json.loads(path.read_text())
    order = torch.tensor(value["new_to_old"], dtype=torch.long, device=device)
    return torch.argsort(order)


def _decode_mcg(
    *,
    layer: int,
    expert: int,
    sidecar: dict[str, Any],
    handle: Any,
    device: torch.device,
    lut: torch.Tensor,
    decode_exl3_weight: Any,
    unpack_trellis_states: Any,
) -> dict[str, torch.Tensor]:
    order = torch.tensor(
        sidecar["permutations"][str(expert)]["new_to_old"],
        dtype=torch.long,
        device=device,
    )
    inverse = torch.argsort(order)
    result: dict[str, torch.Tensor] = {}
    for projection in PROJECTIONS:
        prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        bits = int(sidecar["bit_map"][prefix])
        refs = sidecar["vector_refs"][prefix]
        states = unpack_trellis_states(
            handle.get_tensor(prefix + ".trellis"), bits
        ).to(device)
        decoded = decode_exl3_weight(
            states,
            handle.get_tensor(refs["suh"]).to(device),
            handle.get_tensor(refs["svh"]).to(device),
            codebook_values=lut,
            bits=bits,
        )
        result[projection] = (
            decoded.index_select(1, inverse).contiguous()
            if projection != "down_proj"
            else decoded.index_select(0, inverse).contiguous()
        )
        del states, decoded
    return result


def _decode_sqg(
    *,
    root: Path,
    reference_root: Path,
    layer: int,
    expert: int,
    manifest: dict[str, Any],
    sidecar: dict[str, Any],
    device: torch.device,
    decode_exl3_weight: Any,
    unpack_trellis_states: Any,
    tensor_sha256: Any,
    verify_hashes: bool,
) -> dict[str, torch.Tensor]:
    inverse = _permutation_inverse(reference_root, layer, expert, device)
    shard = root / f"layer_{layer:03d}/experts" / manifest["shard"]
    result: dict[str, torch.Tensor] = {}
    with safe_open(shard, framework="pt", device="cpu") as handle:
        for projection in PROJECTIONS:
            prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
            bits = int(sidecar["bit_map"][prefix])
            tensor_manifest = manifest["tensor_manifests"][prefix]
            if bits != int(tensor_manifest["bits"]):
                raise RuntimeError(f"{prefix}: frozen rate differs")
            states = unpack_trellis_states(
                handle.get_tensor(prefix + ".trellis"), bits
            ).to(device)
            decoded = decode_exl3_weight(
                states,
                handle.get_tensor(prefix + ".suh").to(device),
                handle.get_tensor(prefix + ".svh").to(device),
                bits=bits,
            )
            if verify_hashes and tensor_sha256(decoded.cpu()) != tensor_manifest[
                "decoded_closure"
            ]["decoded_exl_sha256"]:
                raise RuntimeError(f"{prefix}: packed SQG decode hash differs")
            result[projection] = (
                decoded.index_select(1, inverse).contiguous()
                if projection != "down_proj"
                else decoded.index_select(0, inverse).contiguous()
            )
            del states, decoded
    return result


def _new_metric() -> dict[str, float]:
    return {"denominator": 0.0, **{f"{arm}_numerator": 0.0 for arm in ARMS}}


def _add_weighted_sse(
    metric: dict[str, float],
    reference: torch.Tensor,
    candidates: dict[str, torch.Tensor],
    importance: torch.Tensor,
) -> None:
    metric["denominator"] += float(
        torch.sum(reference.square().sum(dim=1) * importance, dtype=torch.float64)
    )
    for arm, candidate in candidates.items():
        difference = candidate - reference
        metric[f"{arm}_numerator"] += float(
            torch.sum(difference.square().sum(dim=1) * importance, dtype=torch.float64)
        )


def _compare_layer(
    layer: int,
    gpu: int,
    mcg_root_raw: str,
    current_root_raw: str,
    corrected_root_raw: str,
    bf16_root_raw: str,
    capture_root_raw: str,
    project_root_raw: str,
    chunk_rows: int,
) -> dict[str, Any]:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(gpu)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(f"cuda:{gpu}")
    mcg_root = Path(mcg_root_raw)
    current_root = Path(current_root_raw)
    corrected_root = Path(corrected_root_raw)
    bf16_root = Path(bf16_root_raw)
    capture_root = Path(capture_root_raw)
    project_root = Path(project_root_raw)
    (
        decode_exl3_weight,
        _,
        unpack_trellis_states,
        _,
        tensor_sha256,
        _,
    ) = _load_reference_functions(project_root)

    sidecar = json.loads(
        (mcg_root / f"r7-experts-layer-{layer:03d}.json").read_text()
    )
    current_manifests = _load_sqg_manifests(current_root, layer)
    corrected_manifests = _load_sqg_manifests(corrected_root, layer)
    hidden_words, topk_weights, route_indices = _route_indices(
        capture_root / f"layer_{layer:03d}"
    )
    lut = _mcg_lut().to(device)
    records: list[dict[str, Any]] = []
    payload = mcg_root / f"r7-experts-layer-{layer:03d}.safetensors"
    with safe_open(payload, framework="pt", device="cpu") as mcg_handle:
        for expert in range(256):
            source_hf = _read_source_triplet(bf16_root, sidecar, layer, expert)
            source = {
                projection: source_hf[projection].T.float().to(device)
                for projection in PROJECTIONS
            }
            weights = {
                "mcg": _decode_mcg(
                    layer=layer,
                    expert=expert,
                    sidecar=sidecar,
                    handle=mcg_handle,
                    device=device,
                    lut=lut,
                    decode_exl3_weight=decode_exl3_weight,
                    unpack_trellis_states=unpack_trellis_states,
                ),
                "current_sqg": _decode_sqg(
                    root=current_root,
                    reference_root=current_root,
                    layer=layer,
                    expert=expert,
                    manifest=current_manifests[expert],
                    sidecar=sidecar,
                    device=device,
                    decode_exl3_weight=decode_exl3_weight,
                    unpack_trellis_states=unpack_trellis_states,
                    tensor_sha256=tensor_sha256,
                    verify_hashes=expert == 0,
                ),
                "corrected_sqg": _decode_sqg(
                    root=corrected_root,
                    reference_root=current_root,
                    layer=layer,
                    expert=expert,
                    manifest=corrected_manifests[expert],
                    sidecar=sidecar,
                    device=device,
                    decode_exl3_weight=decode_exl3_weight,
                    unpack_trellis_states=unpack_trellis_states,
                    tensor_sha256=tensor_sha256,
                    verify_hashes=expert == 0,
                ),
            }

            raw: dict[str, dict[str, float]] = {}
            for projection in PROJECTIONS:
                denominator = float(
                    torch.sum(source[projection].square(), dtype=torch.float64)
                )
                raw[projection] = {
                    "denominator": denominator,
                    **{
                        f"{arm}_numerator": float(
                            torch.sum(
                                (weights[arm][projection] - source[projection]).square(),
                                dtype=torch.float64,
                            )
                        )
                        for arm in ARMS
                    },
                }

            routed: dict[str, dict[str, dict[str, float]]] = {}
            route_mass: dict[str, float] = {}
            route_rows: dict[str, int] = {}
            for role in ROLES:
                rows, slots = route_indices[role][expert]
                gates_cpu = torch.from_numpy(
                    np.array(topk_weights[rows, slots], dtype=np.float32, copy=True)
                )
                importance_cpu = gates_cpu.square()
                route_mass[role] = float(importance_cpu.double().sum())
                route_rows[role] = int(rows.size)
                metrics = {
                    "gate_proj": _new_metric(),
                    "up_proj": _new_metric(),
                    "expert_function": _new_metric(),
                }
                for begin in range(0, rows.size, chunk_rows):
                    end = min(rows.size, begin + chunk_rows)
                    hidden = _load_hidden_chunk(hidden_words, rows[begin:end], device)
                    importance = importance_cpu[begin:end].to(device)
                    reference_gate = hidden @ source["gate_proj"]
                    reference_up = hidden @ source["up_proj"]
                    reference_middle = F.silu(reference_gate) * reference_up
                    reference_output = reference_middle @ source["down_proj"]
                    candidate_gate: dict[str, torch.Tensor] = {}
                    candidate_up: dict[str, torch.Tensor] = {}
                    candidate_output: dict[str, torch.Tensor] = {}
                    for arm in ARMS:
                        gate = hidden @ weights[arm]["gate_proj"]
                        up = hidden @ weights[arm]["up_proj"]
                        candidate_gate[arm] = gate
                        candidate_up[arm] = up
                        candidate_output[arm] = (
                            F.silu(gate) * up
                        ) @ weights[arm]["down_proj"]
                    _add_weighted_sse(
                        metrics["gate_proj"], reference_gate, candidate_gate, importance
                    )
                    _add_weighted_sse(
                        metrics["up_proj"], reference_up, candidate_up, importance
                    )
                    _add_weighted_sse(
                        metrics["expert_function"],
                        reference_output,
                        candidate_output,
                        importance,
                    )
                    del (
                        hidden,
                        importance,
                        reference_gate,
                        reference_up,
                        reference_middle,
                        reference_output,
                        candidate_gate,
                        candidate_up,
                        candidate_output,
                    )
                routed[role] = metrics

            gate_prefix = f"model.layers.{layer}.mlp.experts.{expert}.gate_proj"
            h13_shrinkage = corrected_manifests[expert]["calibration"][
                "profile_cell"
            ]["h13_shrinkage"]
            records.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "bits": {
                        projection: int(
                            sidecar["bit_map"][
                                f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                            ]
                        )
                        for projection in PROJECTIONS
                    },
                    "route_rows": route_rows,
                    "route_gate_square_mass": route_mass,
                    "raw_weight": raw,
                    "routed": routed,
                    "h13_shrinkage": h13_shrinkage,
                    "corrected_h13_matrix_sha256": corrected_manifests[expert][
                        "tensor_manifests"
                    ][gate_prefix]["dense_h"]["matrix_sha256"],
                }
            )
            del source_hf, source, weights
            torch.cuda.empty_cache()
            if (expert + 1) % 16 == 0:
                print(f"layer {layer}: {expert + 1}/256 experts", flush=True)
    return {"layer": layer, "gpu": gpu, "records": records}


def _aggregate_metrics(metrics: Iterable[dict[str, float]]) -> dict[str, Any]:
    values = list(metrics)
    denominator = sum(value["denominator"] for value in values)
    result = {
        arm: sum(value[f"{arm}_numerator"] for value in values) / denominator
        for arm in ARMS
    }
    result["corrected_reduction_vs_mcg_percent"] = (
        1.0 - result["corrected_sqg"] / result["mcg"]
    ) * 100.0
    result["corrected_reduction_vs_current_percent"] = (
        1.0 - result["corrected_sqg"] / result["current_sqg"]
    ) * 100.0
    result["current_reduction_vs_mcg_percent"] = (
        1.0 - result["current_sqg"] / result["mcg"]
    ) * 100.0
    return result


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    summary["raw_weight_all"] = _aggregate_metrics(
        record["raw_weight"][projection]
        for record in records
        for projection in PROJECTIONS
    )
    for projection in PROJECTIONS:
        summary[f"raw_weight_{projection}"] = _aggregate_metrics(
            record["raw_weight"][projection] for record in records
        )
    for role in ROLES:
        summary[f"{role}_gate_up"] = _aggregate_metrics(
            record["routed"][role][projection]
            for record in records
            for projection in ("gate_proj", "up_proj")
        )
        summary[f"{role}_expert_function"] = _aggregate_metrics(
            record["routed"][role]["expert_function"] for record in records
        )
    alphas = [float(record["h13_shrinkage"]["local_alpha"]) for record in records]
    summary["h13_local_alpha"] = {
        "min": min(alphas),
        "median": statistics.median(alphas),
        "mean": statistics.fmean(alphas),
        "max": max(alphas),
    }
    return summary


def _render(result: dict[str, Any]) -> str:
    labels = {
        "raw_weight_all": "Raw weights, all projections",
        "fit_gate_up": "Fit routed gate/up",
        "selection_gate_up": "Selection routed gate/up",
        "holdout_gate_up": "Holdout routed gate/up",
        "fit_expert_function": "Fit complete expert function",
        "selection_expert_function": "Selection complete expert function",
        "holdout_expert_function": "Holdout complete expert function",
    }
    lines = [
        "# Expert-local H13 SQG recalibration",
        "",
        "Lower NMSE is better. Complete-expert metrics include gate, up, SwiGLU, and down and are weighted by applied router-gate squared.",
        "",
        "## Overall",
        "",
        "| Metric | MCG | Current SQG | Corrected SQG | Corrected vs current | Corrected vs MCG |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in labels.items():
        value = result["summary"][key]
        lines.append(
            f"| {label} | {value['mcg']:.9e} | {value['current_sqg']:.9e} | "
            f"{value['corrected_sqg']:.9e} | "
            f"{value['corrected_reduction_vs_current_percent']:+.3f}% | "
            f"{value['corrected_reduction_vs_mcg_percent']:+.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Holdout complete expert function by layer",
            "",
            "| Layer | MCG | Current SQG | Corrected SQG | Corrected vs current | Corrected vs MCG |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for layer in LAYERS:
        value = result["by_layer"][str(layer)]["holdout_expert_function"]
        lines.append(
            f"| {layer} | {value['mcg']:.9e} | {value['current_sqg']:.9e} | "
            f"{value['corrected_sqg']:.9e} | "
            f"{value['corrected_reduction_vs_current_percent']:+.3f}% | "
            f"{value['corrected_reduction_vs_mcg_percent']:+.3f}% |"
        )
    alpha = result["summary"]["h13_local_alpha"]
    lines.extend(
        [
            "",
            "## Calibration contract",
            "",
            f"- Weighted-OAS local alpha: min `{alpha['min']:.6f}`, median `{alpha['median']:.6f}`, mean `{alpha['mean']:.6f}`, max `{alpha['max']:.6f}`.",
            "- Frozen BF16 source, K3/K4 assignment, profiles, permutations, transforms, SQG codebook, and C128 tail-biting.",
            "- Down H2 rebuilt independently from each corrected decoded gate/up candidate.",
            "- Selection and holdout rows were not used by the corrected encoder.",
            "",
            "## Interpretation limit",
            "",
            "This measures isolated routed expert functions. It does not include cross-expert error cancellation, later-layer routing drift, cross-layer compounding, or final-logit KLD.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcg-root", type=Path, default=DEFAULT_MCG_ROOT)
    parser.add_argument("--current-root", type=Path, default=DEFAULT_SQG_ROOT)
    parser.add_argument("--corrected-root", type=Path, default=DEFAULT_CORRECTED_ROOT)
    parser.add_argument("--bf16-root", type=Path, default=DEFAULT_BF16_ROOT)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.chunk_rows <= 0:
        raise ValueError("chunk rows must be positive")

    context = multiprocessing.get_context("spawn")
    layer_results: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        futures = {
            pool.submit(
                _compare_layer,
                layer,
                gpu,
                str(args.mcg_root.resolve()),
                str(args.current_root.resolve()),
                str(args.corrected_root.resolve()),
                str(args.bf16_root.resolve()),
                str(args.capture_root.resolve()),
                str(PROJECT_ROOT),
                args.chunk_rows,
            ): layer
            for gpu, layer in enumerate(LAYERS)
        }
        for future in as_completed(futures):
            layer = futures[future]
            layer_results[layer] = future.result()
            print(f"layer {layer}: scoring complete", flush=True)

    records = [
        record
        for layer in LAYERS
        for record in layer_results[layer]["records"]
    ]
    if len(records) != 1024:
        raise RuntimeError("comparison did not produce exactly 1,024 experts")
    result = {
        "schema": "glm52-expert-local-h13-sqg-comparison-v1",
        "layers": list(LAYERS),
        "experts": len(records),
        "arms": list(ARMS),
        "summary": _summarize(records),
        "by_layer": {
            str(layer): _summarize(layer_results[layer]["records"])
            for layer in LAYERS
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = args.output.with_suffix(".md")
    report.write_text(_render(result), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
