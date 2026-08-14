#!/usr/bin/env python3
"""Encode a GLM-5.2 expert shard with coupled mixed-rate SQG.

This is a treatment-only experiment.  It preserves the sealed official BF16
source, fit/selection/holdout capture, expert permutation, BMM-law residual
profiles, KQuant transform seeds, and each tensor's independent K3/K4 bit.
The new variable is an exact activation-boundary Hadamard transform shared by
gate, up, and down.  GLM's activation is exactly ``SiLU(gate) * up``.

The script deliberately writes a sibling experiment format rather than
claiming compatibility with the already sealed uncoupled artifacts.  Draws
are encoded independently and may later be selected on the selection split;
the holdout split is never consumed here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


NUM_EXPERTS = 256
HIDDEN = 6144
INTERMEDIATE = 2048
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
SCHEMA = "glm52-coupled-mixed-k3-k4-expert-v4"
H13_CONSTRUCTION = (
    "fit_gate_square_expert_local_fixed_alpha_0p25_"
    "layer_global_prior_coupled_basis_v1"
)
H2_CONSTRUCTION = (
    "fit_applied_gate_square_decoded_coupled_silu_candidate_conditional_"
    "weighted_oas_scaled_identity_cap_0p75_v1"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qsrt-root", type=Path, required=True)
    parser.add_argument("--source-sqg-root", type=Path, required=True)
    parser.add_argument("--profile-selection", type=Path, required=True)
    parser.add_argument("--profile-binding", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument(
        "--draws",
        type=int,
        nargs="+",
        default=(0, 6),
        help="expert-private coupled intermediate draws to encode",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--local-alpha", type=float, default=0.25)
    return parser


def _artifact_paths(
    output_root: Path, layer: int, draw: int, expert: int
) -> tuple[Path, Path]:
    directory = (
        output_root
        / f"layer_{layer:03d}"
        / f"draw_{draw:02d}"
        / "experts"
    )
    stem = f"layer-{layer:03d}-expert-{expert:03d}"
    return directory / f"{stem}.json", directory / f"{stem}.safetensors"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_3p0625_bit_map(path: Path, *, layer: int) -> dict[str, int]:
    """Load an exact measured layer-native coupled allocation without rebasing."""

    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if int(value.get("layer", -1)) != layer:
        raise ValueError(
            f"allocation layer {value.get('layer')} does not match requested layer {layer}"
        )
    if value.get("production_eligible") is False:
        raise ValueError("control-only allocation is not production eligible")
    if value.get("method") == "sealed_layer77_topology_neutral_rate_policy_rebase":
        raise ValueError(
            "layer-77 tensor-identity reuse is a superseded control, not a "
            "measured layer-native allocation"
        )
    policy = value.get("policy_provenance")
    if isinstance(policy, dict) and policy.get("source_layer") not in (None, layer):
        raise ValueError(
            "allocation provenance identifies a different source layer; "
            "production requires layer-native rate scoring"
        )
    assignments = value.get("expert_assignments")
    if not isinstance(assignments, dict) or set(assignments) != {
        str(expert) for expert in range(NUM_EXPERTS)
    }:
        raise ValueError("coupled allocation must cover all 256 experts")
    result: dict[str, int] = {}
    for expert in range(NUM_EXPERTS):
        rates = assignments[str(expert)].get("rates")
        if not isinstance(rates, dict) or set(rates) != set(PROJECTIONS):
            raise ValueError(f"allocation expert {expert} rate map differs")
        for projection in PROJECTIONS:
            bits = int(rates[projection])
            if bits not in (3, 4):
                raise ValueError("coupled allocation permits only K3/K4")
            result[
                f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
            ] = bits
    rates = tuple(result.values())
    k4_count = rates.count(4)
    if k4_count not in (48, 72, 96):
        raise ValueError("coupled allocation K4 count must be 48, 72, or 96")
    k3_count = 768 - k4_count
    bit_units = k3_count * 3 + k4_count * 4
    if (rates.count(3), sum(rates)) != (k3_count, bit_units):
        raise ValueError("coupled allocation does not close at its exact rate budget")
    declared_histogram = value.get("histogram")
    if declared_histogram is not None and declared_histogram != {
        "3": k3_count,
        "4": k4_count,
    }:
        raise ValueError("coupled allocation histogram differs from assignments")
    if value.get("bit_units") not in (None, bit_units):
        raise ValueError("coupled allocation bit units differ from assignments")
    if value.get("bpw") not in (None, bit_units / 768.0):
        raise ValueError("coupled allocation bpw differs from assignments")
    declared_bit_map = value.get("bit_map")
    if declared_bit_map is not None:
        normalized = {str(name): int(bits) for name, bits in declared_bit_map.items()}
        if normalized != result:
            raise ValueError("allocation expert assignments and bit_map differ")
    return result


def _synthetic_config(
    runtime: Any,
    projection: str,
    bits: int,
    profile: Any,
    source_exl: Any,
    *,
    draw: int,
    expert: int,
    derive_seed: Any,
    tensor_sha256: Any,
    SyntheticTensorBinding: Any,
    UniformSQGConfig: Any,
) -> Any:
    digest = tensor_sha256(source_exl)
    binding = SyntheticTensorBinding(
        tensor_name=(
            f"model.layers.{runtime.layer}.mlp.experts.{expert}.{projection}"
        ),
        tensor_sha256=digest,
        fixture_id=(
            "frozen_sqg_function_source_then_saved_physical_permutation_then_"
            f"updated_qsrt_coupled_glm_silu_h512_h128_draw_{draw:02d}"
        ),
    )
    return UniformSQGConfig(
        tensor_id=binding.tensor_name,
        bits=bits,
        matrix_role="synthetic_exl",
        transform_seed=derive_seed(
            runtime.settings.run_id,
            runtime.layer,
            expert,
            projection,
            "kquant-input-transform",
        ),
        output_sign_seed=derive_seed(
            runtime.settings.run_id,
            runtime.layer,
            expert,
            projection,
            "kquant-output-transform",
        ),
        source_binding=binding,
        physical_permutation=None,
        shared_residual_profile=profile,
        sigma_reg=runtime.settings.sigma_reg,
        device=runtime.device,
        production=False,
    )


def _transform_hessian(
    dense_h: Any,
    execution: Any,
    *,
    construction: str,
    canonical_sha256: Any,
    DenseHessian: Any,
    tensor_sha256: Any,
    device: Any,
) -> Any:
    transformed = (
        execution.transform_h13(dense_h.matrix.to(device=device))
        .cpu()
        .contiguous()
    )
    evidence = {
        "source_evidence_id": dense_h.evidence_id,
        "source_hessian_sha256": tensor_sha256(dense_h.matrix),
        "transformed_hessian_sha256": tensor_sha256(transformed),
        "transform": "coupled_residual_input_basis",
        "construction": construction,
        "fit_only": True,
    }
    return DenseHessian(
        matrix=transformed,
        evidence_id=canonical_sha256(evidence),
        construction=construction,
        split_id="fit",
        normalization_count=dense_h.normalization_count,
        routed_sample_count=dense_h.routed_sample_count,
    )


def _build_coupled_h2(
    runtime: Any,
    execution: Any,
    gate_exl: Any,
    up_exl: Any,
    *,
    expert: int,
    chunk_rows: int,
    apply_frozen_h2_shrinkage: Any,
    canonical_sha256: Any,
    DenseHessian: Any,
    tensor_sha256: Any,
    row_mask: Any | None = None,
    fit_scope: str = "fit",
) -> tuple[Any, dict[str, Any]]:
    import torch

    routed = runtime.capture.routed_rows(expert, "fit")
    if routed.rows <= 0:
        raise ValueError(f"layer {runtime.layer} expert {expert}: no fit routes")
    row_indices = routed.row_indices
    gate_square_weights = routed.gate_square_weights
    if row_mask is not None:
        if tuple(row_mask.shape) != (routed.rows,):
            raise ValueError("coupled candidate H2 row mask has the wrong shape")
        torch_mask = torch.from_numpy(row_mask)
        row_indices = row_indices[row_mask]
        gate_square_weights = gate_square_weights[torch_mask]
    selected_rows = int(row_indices.shape[0])
    if selected_rows <= 0:
        raise ValueError(f"layer {runtime.layer} expert {expert}: empty {fit_scope} routes")
    device = torch.device(runtime.device)
    gate = gate_exl.T.to(device=device, dtype=torch.float32).contiguous()
    up = up_exl.T.to(device=device, dtype=torch.float32).contiguous()
    accumulator = torch.zeros(
        (INTERMEDIATE, INTERMEDIATE), dtype=torch.float32, device=device
    )
    importance_chunks: list[torch.Tensor] = []
    denominator = 0.0
    for begin in range(0, selected_rows, chunk_rows):
        end = min(selected_rows, begin + chunk_rows)
        hidden = runtime.capture.load_hidden(
            row_indices[begin:end], device=device, dtype=torch.float32
        )
        transformed_hidden = execution.transform_inputs(hidden)
        middle = execution.decode_middle(transformed_hidden, gate, up)
        importance = gate_square_weights[begin:end].to(
            device=device, dtype=torch.float32
        )
        accumulator.addmm_(middle.T, middle * importance[:, None])
        denominator += float(importance.double().sum())
        importance_chunks.append(importance.cpu())
        del hidden, transformed_hidden, middle, importance
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("coupled candidate H2 has no positive routed mass")
    raw = accumulator.div_(denominator)
    raw = ((raw + raw.T) * 0.5).contiguous()
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("coupled candidate H2 is non-finite")
    importance_cpu = torch.cat(importance_chunks)
    shrunk, shrinkage = apply_frozen_h2_shrinkage(raw, importance_cpu)
    matrix = shrunk.cpu().contiguous()
    evidence: dict[str, Any] = {
        "schema": "glm52-coupled-candidate-h2-v2",
        "layer": runtime.layer,
        "expert": expert,
        "role": fit_scope,
        "scope": "expert-local",
        "construction": H2_CONSTRUCTION,
        "routed": routed.evidence(),
        "routed_parent_row_count": routed.rows,
        "routed_selected_row_count": selected_rows,
        "row_mask_applied": row_mask is not None,
        "weighting": "exact applied route gate squared",
        "upstream_replay": (
            "transform_hidden_then_decode_coupled_gate_up_then_"
            "exact_silu_gate_times_up_then_postactivation_transform"
        ),
        "decoded_gate_exl_sha256": tensor_sha256(gate_exl),
        "decoded_up_exl_sha256": tensor_sha256(up_exl),
        "raw_hessian_sha256": tensor_sha256(raw),
        "hessian_sha256": tensor_sha256(matrix),
        "shrinkage": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in shrinkage.items()
        },
        "fit_only": True,
        "selection_used": False,
        "holdout_used": False,
    }
    evidence_id = canonical_sha256(evidence)
    del gate, up, accumulator, raw, shrunk, importance_cpu
    return (
        DenseHessian(
            matrix=matrix,
            evidence_id=evidence_id,
            construction=H2_CONSTRUCTION,
            split_id=fit_scope,
            normalization_count=1,
            routed_sample_count=selected_rows,
        ),
        {**evidence, "evidence_id": evidence_id},
    )


def _write_artifact(
    *,
    output_root: Path,
    runtime: Any,
    expert: int,
    draw: int,
    spec: Any,
    encoded: dict[str, Any],
    weights: Any,
    permutation: Any,
    h13_evidence: dict[str, Any],
    h2_evidence: dict[str, Any],
    gate_profile: Any,
    down_profile: Any,
    qsrt_revision: str,
    qsrt_diff_sha256: str,
    qsrt_status: str,
    allocation_binding: dict[str, Any],
    profile_selection: dict[str, Any],
    profile_binding: dict[str, Any],
    selected_beta: float,
    chunk_rows: int,
    atomic_json: Any,
    canonical_sha256: Any,
    tensor_prefix: Any,
    torch_tensor_entry: Any,
    write_safetensors_atomic: Any,
) -> dict[str, Any]:
    manifest_path, shard_path = _artifact_paths(
        output_root, runtime.layer, draw, expert
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() or shard_path.exists():
        if not (manifest_path.is_file() and shard_path.is_file()):
            raise ValueError(f"partial coupled artifact exists: {manifest_path}")
        value = json.loads(manifest_path.read_text())
        expected = {
            "schema": SCHEMA,
            "complete": True,
            "layer": runtime.layer,
            "expert": expert,
            "intermediate_draw": draw,
            "allocation": allocation_binding,
            "selected_beta": selected_beta,
            "final_profile_binding": profile_binding,
        }
        if any(value.get(key) != item for key, item in expected.items()):
            raise ValueError(f"coupled resume binding differs: {manifest_path}")
        return value

    entries = []
    tensor_manifests: dict[str, Any] = {}
    rates: dict[str, int] = {}
    for projection in PROJECTIONS:
        prefix = tensor_prefix(runtime.layer, expert, projection)
        item = encoded[projection]
        expected_bits = int(runtime.bit_map[prefix])
        if item.bits != expected_bits:
            raise RuntimeError(f"{prefix}: independent K3/K4 rate changed")
        rates[projection] = expected_bits
        tensor_manifests[prefix] = dict(item.manifest)
        entries.extend(
            torch_tensor_entry(name, tensor)
            for name, tensor in item.named_tensors(prefix).items()
        )
    payload_sha256, shard_sha256 = write_safetensors_atomic(
        shard_path,
        entries,
        metadata={
            "format": "pt",
            "schema": SCHEMA,
            "layer": str(runtime.layer),
            "expert": str(expert),
            "intermediate_draw": str(draw),
            "activation": "silu",
            "rate_contract": "independent_per_tensor_k3_k4",
        },
    )
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "complete": True,
        "layer": runtime.layer,
        "expert": expert,
        "intermediate_draw": draw,
        "activation": "silu_gate_times_up",
        "rate_contract": "independent_per_tensor_k3_k4",
        "bits": rates,
        "bit_map_entries": {
            tensor_prefix(runtime.layer, expert, projection): rates[projection]
            for projection in PROJECTIONS
        },
        "allocation": allocation_binding,
        "selected_beta": selected_beta,
        "profile_selection": profile_selection,
        "final_profile_binding": profile_binding,
        "uniform_k3_candidate": False,
        "shard": shard_path.name,
        "shard_sha256": shard_sha256,
        "payload_sha256": payload_sha256,
        "tensor_manifests": tensor_manifests,
        "coupled_transform": {
            "residual_block_size": spec.residual_block_size,
            "preactivation_block_size": spec.preactivation_block_size,
            "postactivation_block_size": spec.postactivation_block_size,
            "residual_draw": spec.residual_draw,
            "intermediate_draw": spec.intermediate_draw,
            "activation": spec.activation,
            "residual_policy": (
                "identity_to_preserve_sealed_bmmlaw_profiles"
                if spec.residual_block_size == 1
                else "coupled_block_hadamard"
            ),
        },
        "physical_permutation": permutation.manifest(),
        "profiles": {
            "gate_up_input": gate_profile.manifest(),
            "down_output": down_profile.manifest(),
            "no_shortcut_final_recipe": True,
        },
        "calibration": {
            "h13": h13_evidence,
            "h13_evidence_id": h13_evidence["transformed_evidence_id"],
            "h2": h2_evidence,
            "fit_only": True,
            "selection_used": False,
            "holdout_used": False,
            "local_h13_alpha": 0.25,
            "numerical_chunk_rows": chunk_rows,
        },
        "source": {
            **runtime.source.validation.manifest(),
            "tensor_payload_sha256": dict(sorted(weights.tensor_sha256.items())),
            "shards": dict(sorted(weights.shard_names.items())),
            "lossy_second_stage_transcode": True,
            "original_bf16_routed_shards_read": False,
        },
        "qsrt": {
            "revision": qsrt_revision,
            "activation_parametric_patch": True,
            "tracked_diff_sha256": qsrt_diff_sha256,
            "status_porcelain": qsrt_status,
        },
        "legacy_mcg_inputs": False,
    }
    unsigned = dict(manifest)
    manifest["manifest_id"] = canonical_sha256(unsigned)
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    args = _parser().parse_args()
    if not 0 <= args.start < args.end <= NUM_EXPERTS:
        raise ValueError("expert range must satisfy 0 <= start < end <= 256")
    if args.threads <= 0 or args.chunk_rows <= 0:
        raise ValueError("thread and chunk counts must be positive")
    if not 0.0 <= args.local_alpha <= 1.0:
        raise ValueError("--local-alpha must lie in [0,1]")
    if args.local_alpha != 0.25:
        raise ValueError("this isolated coupled arm freezes local H13 alpha at 0.25")
    draws = tuple(dict.fromkeys(args.draws))
    if not draws or any(draw not in range(8) for draw in draws):
        raise ValueError("coupled draws must be unique values in 0..7")
    qsrt_root = args.qsrt_root.resolve()
    if not (qsrt_root / "qsrt/qsrt_coupled.py").is_file():
        raise FileNotFoundError("QSRT coupled implementation is missing")
    project_root = Path(__file__).resolve().parents[1]
    for root in (project_root, qsrt_root):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(args.threads)

    import torch
    from bmmlaw_r7_encoder.safetensors_io import (
        torch_tensor_entry,
        write_safetensors_atomic,
    )
    from scripts.coupled_recipe_core import (
        build_profile_global_h13,
        fit_coupled_down,
        load_final_recipe_profile,
        prepare_coupled_expert,
    )
    from src.fresh_pipeline_common import (
        atomic_json,
        canonical_sha256,
        tensor_prefix,
    )
    from src.sqg_checkpoint_source import open_sqg_transcode_runtime

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("real mixed-rate SQG encoding requires CUDA")

    started = time.monotonic()
    allocation_path = args.allocation.resolve()
    allocation_payload = json.loads(allocation_path.read_text(encoding="utf-8"))
    allocation_bit_map = _load_3p0625_bit_map(allocation_path, layer=args.layer)
    allocation_binding = {
        "file": allocation_path.name,
        "schema": allocation_payload.get("schema"),
        "allocation_id": allocation_payload.get("allocation_id"),
        "sha256": _sha256_file(allocation_path),
        "histogram": allocation_payload.get("histogram"),
        "bpw": allocation_payload.get("bpw"),
    }
    runtime = open_sqg_transcode_runtime(
        args.preflight,
        model_root=args.source_sqg_root,
        layer=args.layer,
        device=args.device,
        bit_map=allocation_bit_map,
        probe_rows=1024,
    )
    output_root = args.output_root.resolve()
    if output_root == runtime.paths.output_root.resolve():
        raise ValueError("coupled output must differ from the sealed preparation")
    gate_profile, down_profile, selection, beta, profile_binding = (
        load_final_recipe_profile(
            runtime,
            selection_path=args.profile_selection,
            binding_path=args.profile_binding,
            layer=args.layer,
        )
    )
    if (
        allocation_payload.get("final_profile_binding") != profile_binding
        or allocation_payload.get("profile_selection") != selection
        or float(allocation_payload.get("selected_beta", -1.0)) != beta
    ):
        raise ValueError("allocation and no-shortcut profile binding differ")
    global_h13, global_h13_evidence = build_profile_global_h13(
        runtime,
        gate_profile,
        cell_id=str(selection["selected_cell_id"]),
        chunk_rows=args.chunk_rows,
    )
    qsrt_revision = subprocess.check_output(
        ["git", "-C", str(qsrt_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if len(qsrt_revision) != 40:
        raise RuntimeError("could not bind the QSRT source revision")
    qsrt_diff = subprocess.check_output(
        ["git", "-C", str(qsrt_root), "diff", "--binary", "HEAD"]
    )
    qsrt_diff_sha256 = hashlib.sha256(qsrt_diff).hexdigest()
    qsrt_status = subprocess.check_output(
        ["git", "-C", str(qsrt_root), "status", "--short"],
        text=True,
    ).strip()

    newly_encoded = 0
    resumed = 0
    for expert in range(args.start, args.end):
        expected_bits = {
            projection: int(
                runtime.bit_map[
                    tensor_prefix(runtime.layer, expert, projection)
                ]
            )
            for projection in PROJECTIONS
        }
        complete_draws = 0
        for draw in draws:
            manifest_path, shard_path = _artifact_paths(
                output_root, runtime.layer, draw, expert
            )
            if manifest_path.is_file() and shard_path.is_file():
                existing = json.loads(manifest_path.read_text())
                if (
                    existing.get("schema") != SCHEMA
                    or existing.get("complete") is not True
                    or existing.get("layer") != runtime.layer
                    or existing.get("expert") != expert
                    or existing.get("intermediate_draw") != draw
                    or existing.get("bits") != expected_bits
                    or existing.get("allocation") != allocation_binding
                    or existing.get("selected_beta") != beta
                    or existing.get("final_profile_binding") != profile_binding
                    or existing.get("calibration", {}).get(
                        "numerical_chunk_rows"
                    )
                    != args.chunk_rows
                    or existing.get("shard") != shard_path.name
                    or existing.get("shard_sha256") != _sha256_file(shard_path)
                ):
                    raise ValueError(
                        f"coupled resume artifact differs: {manifest_path}"
                    )
                complete_draws += 1
                continue
            if manifest_path.exists() or shard_path.exists():
                raise ValueError(f"partial coupled artifact exists: {manifest_path}")
        if complete_draws == len(draws):
            resumed += complete_draws
            progress = expert - args.start + 1
            print(
                f"layer {runtime.layer} experts {args.start}:{args.end}: "
                f"{progress}/{args.end - args.start} draws={draws} (resumed)",
                flush=True,
            )
            continue

        for draw in draws:
            manifest_path, shard_path = _artifact_paths(
                output_root, runtime.layer, draw, expert
            )
            if manifest_path.is_file() and shard_path.is_file():
                existing = json.loads(manifest_path.read_text())
                if (
                    existing.get("schema") != SCHEMA
                    or existing.get("complete") is not True
                    or existing.get("intermediate_draw") != draw
                    or existing.get("bits") != expected_bits
                    or existing.get("allocation") != allocation_binding
                    or existing.get("selected_beta") != beta
                    or existing.get("final_profile_binding") != profile_binding
                    or existing.get("calibration", {}).get(
                        "numerical_chunk_rows"
                    )
                    != args.chunk_rows
                ):
                    raise ValueError(f"coupled resume artifact differs: {manifest_path}")
                resumed += 1
                continue
            if manifest_path.exists() or shard_path.exists():
                raise ValueError(f"partial coupled artifact exists: {manifest_path}")
            rates = tuple(expected_bits[name] for name in PROJECTIONS)
            prepared = prepare_coupled_expert(
                runtime,
                expert=expert,
                gate_profile=gate_profile,
                down_profile=down_profile,
                global_h13=global_h13,
                global_evidence=global_h13_evidence,
                triplets=(rates,),
                qsrt_root=qsrt_root,
                chunk_rows=args.chunk_rows,
                intermediate_draw=draw,
            )
            encoded_down, _native_down, down_evidence = fit_coupled_down(
                prepared, beta=beta
            )
            pair = (expected_bits["gate_proj"], expected_bits["up_proj"])
            down_bits = expected_bits["down_proj"]
            encoded = {
                "gate_proj": prepared.encoded_upstream["gate_proj"][rates[0]],
                "up_proj": prepared.encoded_upstream["up_proj"][rates[1]],
                "down_proj": encoded_down[pair][down_bits],
            }
            _write_artifact(
                output_root=output_root,
                runtime=runtime,
                expert=expert,
                draw=draw,
                spec=prepared.spec,
                encoded=encoded,
                weights=prepared.weights,
                permutation=prepared.permutation,
                h13_evidence=prepared.coupled_h13_evidence,
                h2_evidence={
                    "preliminary_anchor": prepared.preliminary_h2_evidence[
                        f"k{pair[0]}_k{pair[1]}"
                    ],
                    "candidate_conditioned_h_b": down_evidence[pair][down_bits],
                },
                gate_profile=gate_profile,
                down_profile=down_profile,
                qsrt_revision=qsrt_revision,
                qsrt_diff_sha256=qsrt_diff_sha256,
                qsrt_status=qsrt_status,
                allocation_binding=allocation_binding,
                profile_selection=selection,
                profile_binding=profile_binding,
                selected_beta=beta,
                chunk_rows=args.chunk_rows,
                atomic_json=atomic_json,
                canonical_sha256=canonical_sha256,
                tensor_prefix=tensor_prefix,
                torch_tensor_entry=torch_tensor_entry,
                write_safetensors_atomic=write_safetensors_atomic,
            )
            newly_encoded += 1
            del encoded_down, _native_down, down_evidence, encoded
            prepared.release()
            gc.collect()
            torch.cuda.empty_cache()
        progress = {
            "schema": "glm52-coupled-mixed-rate-shard-progress-v4",
            "complete": False,
            "layer": runtime.layer,
            "start": args.start,
            "end": args.end,
            "completed_through": expert,
            "draws": list(draws),
            "uniform_k3_candidate": False,
            "newly_encoded": newly_encoded,
            "resumed": resumed,
            "elapsed_seconds": time.monotonic() - started,
            "allocation": allocation_binding,
            "selected_beta": beta,
            "final_profile_binding_id": profile_binding["binding_id"],
            "numerical_chunk_rows": args.chunk_rows,
        }
        atomic_json(
            output_root
            / f"layer_{runtime.layer:03d}"
            / f"progress_{args.start:03d}_{args.end:03d}.json",
            progress,
            overwrite=True,
        )
        print(
            f"layer {runtime.layer} experts {args.start}:{args.end}: "
            f"{expert + 1 - args.start}/{args.end - args.start} "
            f"draws={draws}",
            flush=True,
        )

    result = {
        "schema": "glm52-coupled-mixed-rate-shard-result-v4",
        "complete": True,
        "layer": runtime.layer,
        "start": args.start,
        "end": args.end,
        "draws": list(draws),
        "activation": "silu_gate_times_up",
        "rate_contract": "independent_per_tensor_k3_k4",
        "uniform_k3_candidate": False,
        "local_h13_alpha": args.local_alpha,
        "selected_profile_id": selection["selection_id"],
        "selected_beta": beta,
        "final_profile_binding_id": profile_binding["binding_id"],
        "allocation": allocation_binding,
        "numerical_chunk_rows": args.chunk_rows,
        "newly_encoded": newly_encoded,
        "resumed": resumed,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(
        output_root
        / f"layer_{runtime.layer:03d}"
        / f"result_{args.start:03d}_{args.end:03d}.json",
        result,
        overwrite=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
