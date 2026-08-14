#!/usr/bin/env python3
"""Exact Hessian/routed distortion comparison for matched GLM MCG and SQG.

One process is assigned to each selected layer/GPU.  Gate and up projections
are scored under both the sealed layer-global H13 used by the encoder and the
exact expert-local routed covariance.  Down projections are scored under
three common candidate-conditioned H2 regimes: BF16, decoded MCG, and decoded
SQG upstream gate/up.  Every arm uses the same BF16 source and frozen K3/K4
assignment.
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

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from scripts.compare_raw_encoded_nmse import (
    DEFAULT_BF16_ROOT,
    DEFAULT_MCG_ROOT,
    DEFAULT_SQG_ROOT,
    _load_sqg_manifests,
    _mcg_lut,
)


DEFAULT_CAPTURE_ROOT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/"
    "fresh-sqg-full2.GPlzPL/fresh-sqg-calibration-r1"
)
DEFAULT_OUTPUT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/"
    "hessian_weighted_nmse_sqg_vs_mcg.json"
)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
HIDDEN = 6144
INTERMEDIATE = 2048
TOPK = 8
FIT_ROLE = 0
SIGMA_REG = 0.025
H2_REGIMES = ("bf16_upstream", "mcg_upstream", "sqg_upstream")


def _load_reference_functions(experiment_root: Path):
    import sys

    for path in (str(experiment_root / "kquant"), str(experiment_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from kquant.exl3_reference import decode_exl3_weight
    from kquant.sqg_e4m3 import sqg_xor_cheb_t12_bytes
    from src.calibration_hessian import apply_frozen_h2_shrinkage
    from src.glm52_fresh_sqg.reference import (
        decode_stored_fp16,
        tensor_sha256,
        unpack_trellis_states,
    )

    return (
        decode_exl3_weight,
        decode_stored_fp16,
        unpack_trellis_states,
        sqg_xor_cheb_t12_bytes,
        tensor_sha256,
        apply_frozen_h2_shrinkage,
    )


def _read_source_triplet(
    bf16_root: Path,
    sidecar: dict[str, Any],
    layer: int,
    expert: int,
) -> dict[str, torch.Tensor]:
    prefixes = {
        projection: f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        for projection in PROJECTIONS
    }
    provenance = {
        projection: sidecar["tensor_provenance"][prefix]
        for projection, prefix in prefixes.items()
    }
    result: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for projection, value in provenance.items():
        by_shard[value["source_shard"]].append((projection, value))
    for shard, entries in by_shard.items():
        with safe_open(bf16_root / shard, framework="pt", device="cpu") as handle:
            for projection, value in entries:
                result[projection] = handle.get_tensor(value["source_name"])
    return result


def _decode_expert(
    *,
    layer: int,
    expert: int,
    device: torch.device,
    sidecar: dict[str, Any],
    mcg_handle: Any,
    sqg_manifest: dict[str, Any],
    sqg_root: Path,
    lut_mcg: torch.Tensor,
    decode_exl3_weight: Any,
    decode_stored_fp16: Any,
    unpack_trellis_states: Any,
    sqg_lut_bytes: Any,
    tensor_sha256: Any,
    verify_decode_hashes: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    mcg_permutation = torch.tensor(
        sidecar["permutations"][str(expert)]["new_to_old"],
        dtype=torch.long,
        device=device,
    )
    sqg_permutation_path = (
        sqg_root
        / f"layer_{layer:03d}"
        / "preparation"
        / "permutations"
        / f"expert-{expert:03d}.json"
    )
    sqg_permutation_json = json.loads(sqg_permutation_path.read_text())
    sqg_permutation = torch.tensor(
        sqg_permutation_json["new_to_old"], dtype=torch.long, device=device
    )
    mcg_inverse = torch.argsort(mcg_permutation)
    sqg_inverse = torch.argsort(sqg_permutation)

    sqg_shard = (
        sqg_root / f"layer_{layer:03d}" / "experts" / sqg_manifest["shard"]
    )
    mcg: dict[str, torch.Tensor] = {}
    sqg: dict[str, torch.Tensor] = {}
    with safe_open(sqg_shard, framework="pt", device="cpu") as sqg_handle:
        for projection in PROJECTIONS:
            prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
            bits = int(sidecar["bit_map"][prefix])
            if bits != int(sqg_manifest["tensor_manifests"][prefix]["bits"]):
                raise RuntimeError(f"{prefix}: K3/K4 assignment differs")

            vector_refs = sidecar["vector_refs"][prefix]
            # Unpack on CPU.  The int64 bitwise intermediates otherwise create
            # a 96--200 MiB transient CUDA spike, which is unnecessary because
            # the packed payload is small and the decoded states are identical.
            mcg_states = unpack_trellis_states(
                mcg_handle.get_tensor(prefix + ".trellis"), bits
            ).to(device)
            mcg_exl = decode_exl3_weight(
                mcg_states,
                mcg_handle.get_tensor(vector_refs["suh"]).to(device),
                mcg_handle.get_tensor(vector_refs["svh"]).to(device),
                codebook_values=lut_mcg,
                bits=bits,
            )

            sqg_states = unpack_trellis_states(
                sqg_handle.get_tensor(prefix + ".trellis"), bits
            ).to(device)
            sqg_exl = decode_exl3_weight(
                sqg_states,
                sqg_handle.get_tensor(prefix + ".suh").to(device),
                sqg_handle.get_tensor(prefix + ".svh").to(device),
                bits=bits,
            )
            if verify_decode_hashes:
                observed_sqg_hash = tensor_sha256(sqg_exl.cpu())
                expected_sqg_hash = sqg_manifest["tensor_manifests"][prefix][
                    "decoded_closure"
                ]["decoded_exl_sha256"]
                if observed_sqg_hash != expected_sqg_hash:
                    raise RuntimeError(f"{prefix}: SQG packed-byte decode hash differs")

            if projection in ("gate_proj", "up_proj"):
                mcg[projection] = mcg_exl.index_select(1, mcg_inverse).contiguous()
                sqg[projection] = sqg_exl.index_select(1, sqg_inverse).contiguous()
            else:
                mcg[projection] = mcg_exl.index_select(0, mcg_inverse).contiguous()
                sqg[projection] = sqg_exl.index_select(0, sqg_inverse).contiguous()
            del mcg_states, sqg_states, mcg_exl, sqg_exl
    return mcg, sqg


def _regularize_hessian(hessian: torch.Tensor, sigma_reg: float) -> torch.Tensor:
    value = hessian.float().contiguous()
    value.diagonal().add_(float(sigma_reg) * value.diagonal().mean())
    return value


def _trace(weight: torch.Tensor, hessian: torch.Tensor) -> float:
    product = torch.matmul(hessian, weight)
    result = float(torch.sum(weight * product, dtype=torch.float64).item())
    del product
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError(f"invalid positive Hessian trace: {result}")
    return result


def _route_index(capture_layer: Path) -> tuple[np.memmap, dict[int, tuple[np.ndarray, np.ndarray]]]:
    role_path = capture_layer / "role_ids.u8.bin"
    rows = role_path.stat().st_size
    role_ids = np.memmap(role_path, mode="r", dtype="u1", shape=(rows,))
    topk_ids = np.memmap(
        capture_layer / "topk_ids.u8.bin",
        mode="r",
        dtype="u1",
        shape=(rows, TOPK),
    )
    fit_rows = np.flatnonzero(np.asarray(role_ids) == FIT_ROLE).astype(
        np.int64, copy=False
    )
    ids = np.asarray(topk_ids[fit_rows]).reshape(-1)
    routed_rows = np.repeat(fit_rows, TOPK)
    routed_slots = np.tile(np.arange(TOPK, dtype=np.int16), fit_rows.size)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    boundaries = np.searchsorted(sorted_ids, np.arange(257), side="left")
    result = {
        expert: (
            routed_rows[order[boundaries[expert] : boundaries[expert + 1]]],
            routed_slots[order[boundaries[expert] : boundaries[expert + 1]]],
        )
        for expert in range(256)
    }
    hidden_words = np.memmap(
        capture_layer / "hidden.bf16.bin",
        mode="r",
        dtype="<u2",
        shape=(rows, HIDDEN),
    )
    return hidden_words, result


def _load_hidden_chunk(
    hidden_words: np.memmap,
    rows: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    words = torch.from_numpy(
        np.array(hidden_words[rows], dtype=np.uint16, copy=True)
    )
    return words.view(torch.bfloat16).to(device=device, dtype=torch.float32)


def _finish_h2(
    accumulator: torch.Tensor,
    denominator: float,
    importance: torch.Tensor,
    apply_frozen_h2_shrinkage: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    covariance = accumulator.div_(denominator)
    covariance = ((covariance + covariance.T) * 0.5).contiguous()
    hessian, evidence = apply_frozen_h2_shrinkage(covariance, importance)
    return _regularize_hessian(hessian, SIGMA_REG), evidence


def _compare_layer(
    layer: int,
    gpu: int,
    mcg_root_raw: str,
    sqg_root_raw: str,
    bf16_root_raw: str,
    capture_root_raw: str,
    experiment_root_raw: str,
    chunk_rows: int,
    limit_experts: int | None,
) -> dict[str, Any]:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    torch.set_float32_matmul_precision("highest")

    mcg_root = Path(mcg_root_raw)
    sqg_root = Path(sqg_root_raw)
    bf16_root = Path(bf16_root_raw)
    capture_root = Path(capture_root_raw)
    experiment_root = Path(experiment_root_raw)
    (
        decode_exl3_weight,
        decode_stored_fp16,
        unpack_trellis_states,
        sqg_lut_bytes,
        tensor_sha256,
        apply_frozen_h2_shrinkage,
    ) = _load_reference_functions(experiment_root)

    sidecar = json.loads(
        (mcg_root / f"r7-experts-layer-{layer:03d}.json").read_text()
    )
    sqg_manifests = _load_sqg_manifests(sqg_root, layer)
    with safe_open(
        sqg_root / f"layer_{layer:03d}/preparation/h13.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        h13 = handle.get_tensor("h13")
    h13_json = json.loads(
        (sqg_root / f"layer_{layer:03d}/preparation/h13.json").read_text()
    )
    if tensor_sha256(h13) != h13_json["evidence"]["matrix_sha256"]:
        raise RuntimeError(f"layer {layer}: sealed H13 tensor hash differs")
    h13 = _regularize_hessian(h13.to(device), SIGMA_REG)

    capture_layer = capture_root / f"layer_{layer:03d}"
    hidden_words, route_index = _route_index(capture_layer)
    capture_rows = hidden_words.shape[0]
    topk_weights = np.memmap(
        capture_layer / "topk_weights.f32le.bin",
        mode="r",
        dtype="<f4",
        shape=(capture_rows, TOPK),
    )
    lut_mcg = _mcg_lut().to(device)
    expert_count = 256 if limit_experts is None else min(256, limit_experts)
    rows_out: list[dict[str, Any]] = []

    mcg_payload_path = mcg_root / f"r7-experts-layer-{layer:03d}.safetensors"
    with safe_open(mcg_payload_path, framework="pt", device="cpu") as mcg_handle:
        for expert in range(expert_count):
            sqg_manifest = sqg_manifests[expert]
            sources_hf = _read_source_triplet(bf16_root, sidecar, layer, expert)
            source = {
                "gate_proj": sources_hf["gate_proj"].T.float().to(device),
                "up_proj": sources_hf["up_proj"].T.float().to(device),
                "down_proj": sources_hf["down_proj"].T.float().to(device),
            }
            mcg, sqg = _decode_expert(
                layer=layer,
                expert=expert,
                device=device,
                sidecar=sidecar,
                mcg_handle=mcg_handle,
                sqg_manifest=sqg_manifest,
                sqg_root=sqg_root,
                lut_mcg=lut_mcg,
                decode_exl3_weight=decode_exl3_weight,
                decode_stored_fp16=decode_stored_fp16,
                unpack_trellis_states=unpack_trellis_states,
                sqg_lut_bytes=sqg_lut_bytes,
                tensor_sha256=tensor_sha256,
                verify_decode_hashes=expert == 0,
            )

            prefixes = {
                projection: f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                for projection in PROJECTIONS
            }
            global_h13: dict[str, dict[str, float]] = {}
            for projection in ("gate_proj", "up_proj"):
                denominator = _trace(source[projection], h13)
                mcg_numerator = _trace(mcg[projection] - source[projection], h13)
                sqg_numerator = _trace(sqg[projection] - source[projection], h13)
                global_h13[projection] = {
                    "denominator": denominator,
                    "mcg_numerator": mcg_numerator,
                    "sqg_numerator": sqg_numerator,
                }

            routed_rows, route_slots = route_index[expert]
            gates = torch.from_numpy(
                np.array(
                    topk_weights[routed_rows, route_slots],
                    dtype=np.float32,
                    copy=True,
                )
            )
            importance_cpu = gates.square().contiguous()
            route_mass = float(importance_cpu.double().sum().item())
            if routed_rows.size != int(
                sqg_manifest["calibration"]["fit_routes"]["rows"]
            ) or not math.isclose(
                route_mass,
                float(sqg_manifest["calibration"]["fit_routes"]["gate_square_sum"]),
                rel_tol=1e-11,
                abs_tol=1e-7,
            ):
                raise RuntimeError(f"layer {layer} expert {expert}: route binding differs")

            local_sums = {
                projection: {
                    "denominator": 0.0,
                    "mcg_numerator": 0.0,
                    "sqg_numerator": 0.0,
                }
                for projection in ("gate_proj", "up_proj")
            }
            h2_accumulators = {
                regime: torch.zeros(
                    (INTERMEDIATE, INTERMEDIATE),
                    dtype=torch.float32,
                    device=device,
                )
                for regime in H2_REGIMES
            }
            input_square_weighted_sum = 0.0
            for begin in range(0, routed_rows.size, chunk_rows):
                end = min(routed_rows.size, begin + chunk_rows)
                hidden = _load_hidden_chunk(
                    hidden_words, routed_rows[begin:end], device
                )
                importance = importance_cpu[begin:end].to(device)
                input_square_weighted_sum += float(
                    torch.sum(
                        hidden.square().sum(dim=1) * importance,
                        dtype=torch.float64,
                    ).item()
                )

                outputs: dict[str, dict[str, torch.Tensor]] = {}
                for arm, weights in (("bf16", source), ("mcg", mcg), ("sqg", sqg)):
                    gate_output = hidden @ weights["gate_proj"]
                    up_output = hidden @ weights["up_proj"]
                    outputs[arm] = {
                        "gate_proj": gate_output,
                        "up_proj": up_output,
                    }
                    intermediate = F.silu(gate_output) * up_output
                    h2_accumulators[f"{arm}_upstream"].addmm_(
                        intermediate.T, intermediate * importance[:, None]
                    )

                for projection in ("gate_proj", "up_proj"):
                    reference_output = outputs["bf16"][projection]
                    local_sums[projection]["denominator"] += float(
                        torch.sum(
                            reference_output.square().sum(dim=1) * importance,
                            dtype=torch.float64,
                        ).item()
                    )
                    for arm in ("mcg", "sqg"):
                        difference = outputs[arm][projection] - reference_output
                        local_sums[projection][f"{arm}_numerator"] += float(
                            torch.sum(
                                difference.square().sum(dim=1) * importance,
                                dtype=torch.float64,
                            ).item()
                        )
                del hidden, importance, outputs

            local_diag_mean = input_square_weighted_sum / (route_mass * HIDDEN)
            expert_h13: dict[str, dict[str, float]] = {}
            for projection in ("gate_proj", "up_proj"):
                values = local_sums[projection]
                denominator = values["denominator"] / route_mass
                mcg_numerator = values["mcg_numerator"] / route_mass
                sqg_numerator = values["sqg_numerator"] / route_mass
                denominator += (
                    SIGMA_REG * local_diag_mean * float(source[projection].square().sum())
                )
                mcg_numerator += (
                    SIGMA_REG
                    * local_diag_mean
                    * float((mcg[projection] - source[projection]).square().sum())
                )
                sqg_numerator += (
                    SIGMA_REG
                    * local_diag_mean
                    * float((sqg[projection] - source[projection]).square().sum())
                )
                expert_h13[projection] = {
                    "denominator": denominator,
                    "mcg_numerator": mcg_numerator,
                    "sqg_numerator": sqg_numerator,
                }

            h2_matrices: dict[str, torch.Tensor] = {}
            h2_evidence: dict[str, dict[str, float]] = {}
            for regime, accumulator in h2_accumulators.items():
                matrix, evidence = _finish_h2(
                    accumulator,
                    route_mass,
                    importance_cpu,
                    apply_frozen_h2_shrinkage,
                )
                h2_matrices[regime] = matrix
                h2_evidence[regime] = {
                    key: float(evidence[key])
                    for key in (
                        "effective_sample_size",
                        "oas_shrinkage",
                        "local_alpha",
                        "max_local_alpha",
                        "identity_scale",
                    )
                }

            down_scores: dict[str, dict[str, float]] = {}
            down_mcg_error = mcg["down_proj"] - source["down_proj"]
            down_sqg_error = sqg["down_proj"] - source["down_proj"]
            for regime, matrix in h2_matrices.items():
                down_scores[regime] = {
                    "denominator": _trace(source["down_proj"], matrix),
                    "mcg_numerator": _trace(down_mcg_error, matrix),
                    "sqg_numerator": _trace(down_sqg_error, matrix),
                }

            expected_h2 = sqg_manifest["h2"]
            sqg_h2_validation = {
                "route_mass_relative_error": abs(
                    route_mass - float(expected_h2["gate_square_sum"])
                )
                / float(expected_h2["gate_square_sum"]),
                "local_alpha_absolute_error": abs(
                    h2_evidence["sqg_upstream"]["local_alpha"]
                    - float(expected_h2["shrinkage"]["local_alpha"])
                ),
                "identity_scale_relative_error": abs(
                    h2_evidence["sqg_upstream"]["identity_scale"]
                    - float(expected_h2["shrinkage"]["identity_scale"])
                )
                / max(abs(float(expected_h2["shrinkage"]["identity_scale"])), 1e-30),
            }

            for projection in ("gate_proj", "up_proj"):
                prefix = prefixes[projection]
                row = {
                    "tensor": prefix,
                    "layer": layer,
                    "expert": expert,
                    "projection": projection,
                    "bits": int(sidecar["bit_map"][prefix]),
                    "route_rows": int(routed_rows.size),
                    "route_gate_square_mass": route_mass,
                    "metrics": {
                        "layer_global_h13": global_h13[projection],
                        "expert_local_h13": expert_h13[projection],
                    },
                    "sqg_encoder_proxy": float(
                        sqg_manifest["tensor_manifests"][prefix]["encoder"][
                            "proxy_error"
                        ]
                    ),
                }
                rows_out.append(row)
            prefix = prefixes["down_proj"]
            rows_out.append(
                {
                    "tensor": prefix,
                    "layer": layer,
                    "expert": expert,
                    "projection": "down_proj",
                    "bits": int(sidecar["bit_map"][prefix]),
                    "route_rows": int(routed_rows.size),
                    "route_gate_square_mass": route_mass,
                    "metrics": {
                        f"candidate_h2_{regime}": values
                        for regime, values in down_scores.items()
                    },
                    "h2_shrinkage": h2_evidence,
                    "sqg_h2_rebuild_validation": sqg_h2_validation,
                    "sqg_encoder_proxy": float(
                        sqg_manifest["tensor_manifests"][prefix]["encoder"][
                            "proxy_error"
                        ]
                    ),
                }
            )

            del sources_hf, source, mcg, sqg, h2_matrices, h2_accumulators
            torch.cuda.empty_cache()
            if (expert + 1) % 16 == 0 or expert + 1 == expert_count:
                print(
                    f"layer {layer} gpu {gpu}: {expert + 1}/{expert_count} experts",
                    flush=True,
                )

    return {"layer": layer, "gpu": gpu, "rows": rows_out}


def _summarize(rows: Iterable[dict[str, Any]], metric: str) -> dict[str, Any]:
    materialized = [row for row in rows if metric in row["metrics"]]
    if not materialized:
        raise ValueError(f"no rows for metric {metric}")
    denominator = sum(row["metrics"][metric]["denominator"] for row in materialized)
    mcg_numerator = sum(
        row["metrics"][metric]["mcg_numerator"] for row in materialized
    )
    sqg_numerator = sum(
        row["metrics"][metric]["sqg_numerator"] for row in materialized
    )
    mcg_hnmse = mcg_numerator / denominator
    sqg_hnmse = sqg_numerator / denominator
    wins = sum(
        row["metrics"][metric]["sqg_numerator"]
        < row["metrics"][metric]["mcg_numerator"]
        for row in materialized
    )
    ratios = [
        row["metrics"][metric]["sqg_numerator"]
        / row["metrics"][metric]["mcg_numerator"]
        for row in materialized
    ]

    routed_denominator = sum(
        row["route_gate_square_mass"] * row["metrics"][metric]["denominator"]
        for row in materialized
    )
    routed_mcg = sum(
        row["route_gate_square_mass"] * row["metrics"][metric]["mcg_numerator"]
        for row in materialized
    ) / routed_denominator
    routed_sqg = sum(
        row["route_gate_square_mass"] * row["metrics"][metric]["sqg_numerator"]
        for row in materialized
    ) / routed_denominator
    return {
        "tensors": len(materialized),
        "energy_weighted_hnmse": {
            "mcg": mcg_hnmse,
            "sqg": sqg_hnmse,
            "sqg_over_mcg": sqg_hnmse / mcg_hnmse,
            "sqg_reduction_percent": (1.0 - sqg_hnmse / mcg_hnmse) * 100.0,
        },
        "routed_mass_weighted_hnmse": {
            "mcg": routed_mcg,
            "sqg": routed_sqg,
            "sqg_over_mcg": routed_sqg / routed_mcg,
            "sqg_reduction_percent": (1.0 - routed_sqg / routed_mcg) * 100.0,
        },
        "sqg_wins": wins,
        "sqg_win_rate": wins / len(materialized),
        "median_tensor_sqg_over_mcg": statistics.median(ratios),
        "geometric_mean_tensor_sqg_over_mcg": math.exp(
            statistics.fmean(math.log(value) for value in ratios)
        ),
    }


def _group(rows: list[dict[str, Any]], metric: str, field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if metric in row["metrics"]:
            groups[str(row[field])].append(row)
    def order(item: tuple[str, Any]) -> tuple[int, int | str]:
        try:
            return (0, int(item[0]))
        except ValueError:
            return (1, item[0])
    return {
        key: _summarize(values, metric)
        for key, values in sorted(groups.items(), key=order)
    }


def _metric_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    return {
        "overall": _summarize(rows, metric),
        "by_bits": _group(rows, metric, "bits"),
        "by_layer": _group(rows, metric, "layer"),
    }


def _render_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Metric | Tensors | MCG HNMSE | SQG HNMSE | SQG reduction | SQG wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "layer_global_h13": "Gate/up: layer-global H13",
        "expert_local_h13": "Gate/up: expert-local H13_e",
        "candidate_h2_bf16_upstream": "Down: H2 from BF16 upstream",
        "candidate_h2_mcg_upstream": "Down: H2 from MCG upstream",
        "candidate_h2_sqg_upstream": "Down: H2 from SQG upstream",
    }
    for metric, value in summary.items():
        overall = value["overall"]
        energy = overall["energy_weighted_hnmse"]
        lines.append(
            f"| {labels[metric]} | {overall['tensors']:,} | "
            f"{energy['mcg']:.9e} | {energy['sqg']:.9e} | "
            f"{energy['sqg_reduction_percent']:+.3f}% | "
            f"{overall['sqg_wins']:,} ({overall['sqg_win_rate']:.1%}) |"
        )
    return lines


def _render_breakdown(
    summary: dict[str, Any], metric: str, group: str, heading: str
) -> list[str]:
    lines = [
        f"## {heading}",
        "",
        "| Group | Tensors | MCG HNMSE | SQG HNMSE | SQG reduction | SQG wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in summary[metric][group].items():
        energy = value["energy_weighted_hnmse"]
        lines.append(
            f"| {key} | {value['tensors']:,} | {energy['mcg']:.9e} | "
            f"{energy['sqg']:.9e} | {energy['sqg_reduction_percent']:+.3f}% | "
            f"{value['sqg_wins']:,} ({value['sqg_win_rate']:.1%}) |"
        )
    lines.append("")
    return lines


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Hessian-weighted encoded distortion: SQG versus MCG",
        "",
        "Lower is better. All reported matrix traces and routed-output reductions are exact FP32 GPU computations; this is not KLD.",
        "",
        "## Overall",
        "",
    ]
    lines.extend(_render_table(result["summary"]))
    lines.append("")
    lines.extend(
        _render_breakdown(
            result["summary"],
            "expert_local_h13",
            "by_layer",
            "Expert-local H13_e by layer",
        )
    )
    lines.extend(
        _render_breakdown(
            result["summary"],
            "expert_local_h13",
            "by_bits",
            "Expert-local H13_e by rate",
        )
    )
    lines.extend(
        _render_breakdown(
            result["summary"],
            "candidate_h2_sqg_upstream",
            "by_layer",
            "SQG-candidate H2 by layer",
        )
    )
    validation = result.get("validation")
    if validation:
        lines.extend(
            [
                "## Validation",
                "",
                "- Maximum SQG-H2 route-mass relative error: "
                f"`{validation['max_h2_route_mass_relative_error']:.3e}`.",
                "- Maximum SQG-H2 local-alpha absolute error: "
                f"`{validation['max_h2_local_alpha_absolute_error']:.3e}`.",
                "- Maximum SQG-H2 identity-scale relative error: "
                f"`{validation['max_h2_identity_scale_relative_error']:.3e}`.",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- Gate/up layer-global scores use each sealed fit-only H13 plus the encoder's 0.025 diagonal damping.",
            "- Expert-local gate/up scores replay the exact fit routes and applied gate-square weights, with the same diagonal damping.",
            "- Down scores rebuild the frozen OAS-to-scaled-identity H2 under BF16, decoded-MCG, and decoded-SQG upstream paths; each common H2 scores both down reconstructions.",
            "- MCG and SQG are decoded from their packed bytes and mapped back to the same official BF16 coordinate order.",
            "- The K3/K4 assignment and BF16 source payload are held fixed.",
            "",
            "## Interpretation limit",
            "",
            "Hessian-weighted local distortion is closer to functional sensitivity than raw NMSE, but it still does not include cross-layer compounding, changed routing, runtime dispatch, or final-logit KLD.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcg-root", type=Path, default=DEFAULT_MCG_ROOT)
    parser.add_argument("--sqg-root", type=Path, default=DEFAULT_SQG_ROOT)
    parser.add_argument("--bf16-root", type=Path, default=DEFAULT_BF16_ROOT)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument(
        "--experiment-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 28, 52, 77])
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--limit-experts", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.gpus) < len(args.layers):
        raise ValueError("one GPU is required per concurrently scored layer")
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    if args.limit_experts is not None and args.limit_experts <= 0:
        raise ValueError("--limit-experts must be positive")

    context = multiprocessing.get_context("spawn")
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=len(args.layers), mp_context=context
    ) as pool:
        futures = {
            pool.submit(
                _compare_layer,
                layer,
                args.gpus[index],
                str(args.mcg_root.resolve()),
                str(args.sqg_root.resolve()),
                str(args.bf16_root.resolve()),
                str(args.capture_root.resolve()),
                str(args.experiment_root.resolve()),
                args.chunk_rows,
                args.limit_experts,
            ): layer
            for index, layer in enumerate(args.layers)
        }
        for future in as_completed(futures):
            value = future.result()
            rows.extend(value["rows"])
            print(
                f"layer {value['layer']}: completed {len(value['rows'])} tensors",
                flush=True,
            )

    rows.sort(key=lambda row: (row["layer"], row["expert"], row["projection"]))
    metrics = (
        "layer_global_h13",
        "expert_local_h13",
        "candidate_h2_bf16_upstream",
        "candidate_h2_mcg_upstream",
        "candidate_h2_sqg_upstream",
    )
    result = {
        "schema": "glm52-hessian-weighted-nmse-sqg-vs-mcg-v1",
        "device": "cuda",
        "layers": sorted(args.layers),
        "gpus": args.gpus[: len(args.layers)],
        "sigma_reg": SIGMA_REG,
        "comparison_contract": {
            "matched_official_bf16_payload_required": True,
            "matched_k3_k4_assignment_required": True,
            "packed_mcg_and_sqg_decode": True,
            "global_h13": "sealed layer-global fit covariance",
            "expert_h13": "exact fit routed gate-square covariance",
            "candidate_h2_regimes": list(H2_REGIMES),
            "h2_shrinkage": "frozen weighted OAS scaled identity cap 0.75",
            "float32_matmul_precision": "highest",
        },
        "summary": {metric: _metric_summary(rows, metric) for metric in metrics},
        "tensors": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report = args.output.with_suffix(".md")
    report.write_text(_render_report(result))
    print(_render_report(result), flush=True)
    print(f"json: {args.output}", flush=True)
    print(f"report: {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
