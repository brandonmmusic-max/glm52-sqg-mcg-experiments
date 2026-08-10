#!/usr/bin/env python3
"""Re-encode a GLM expert shard with support-shrunk expert-local H13.

This is a directional experiment over the already sealed four-layer SQG run.
It preserves the source tensors, K3/K4 map, permutations, profiles, transform
seeds, codebook, and encoder.  The only independent change is gate/up H13:
the fit-routed expert covariance is blended toward the sealed layer H13 using
KQuant's weighted-OAS local-reliability estimate.  Down H2 is then rebuilt
from the newly decoded gate/up candidate before down is encoded.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


NUM_EXPERTS = 256
HIDDEN = 6144
LOCAL_ALPHA_CAP = 0.75
H13_CONSTRUCTION = (
    "fit_gate_square_expert_local_weighted_oas_reliability_"
    "layer_global_prior_cap_0p75_v1"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--chunk-rows", type=int, default=256)
    return parser


def _artifact_paths(directory: Path, layer: int, expert: int) -> tuple[Path, ...]:
    stem = f"layer-{layer:03d}-expert-{expert:03d}"
    manifest = directory / f"{stem}.json"
    return manifest, directory / f"{stem}.safetensors", manifest.with_suffix(
        ".json.sha256"
    )


def _build_h13(
    runtime: Any,
    global_h13: Any,
    *,
    expert: int,
    device: Any,
    chunk_rows: int,
    apply_frozen_h2_shrinkage: Any,
    canonical_sha256: Any,
    tensor_sha256: Any,
    DenseHessian: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch

    routed = runtime.capture.routed_rows(expert, "fit")
    if routed.rows <= 0:
        raise ValueError(f"layer {runtime.layer} expert {expert}: no fit routes")
    importance_cpu = routed.gate_square_weights.contiguous()
    denominator = float(importance_cpu.double().sum())
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("expert-local H13 has no positive routed mass")

    accumulator = torch.zeros(
        (HIDDEN, HIDDEN), dtype=torch.float32, device=device
    )
    for begin in range(0, routed.rows, chunk_rows):
        end = min(routed.rows, begin + chunk_rows)
        hidden = runtime.capture.load_hidden(
            routed.row_indices[begin:end], device=device, dtype=torch.float32
        )
        importance = importance_cpu[begin:end].to(device=device)
        accumulator.addmm_(hidden.T, hidden * importance[:, None])
        del hidden, importance
    local = accumulator.div_(denominator)
    local = ((local + local.T) * 0.5).contiguous()
    if not bool(torch.isfinite(local).all()):
        raise ValueError("expert-local H13 is non-finite")

    # Reuse KQuant's frozen weighted-OAS calculation as a reliability
    # estimator.  H13 coordinates are shared across experts, so its valid
    # prior is the sealed layer covariance rather than scaled identity.
    identity_blend, reliability = apply_frozen_h2_shrinkage(
        local, importance_cpu
    )
    del identity_blend
    local_alpha = float(reliability["local_alpha"])
    if not 0.0 <= local_alpha <= LOCAL_ALPHA_CAP:
        raise RuntimeError("weighted-OAS H13 reliability lies outside its cap")
    global_gpu = global_h13.matrix.to(device=device, dtype=torch.float32)
    blended = torch.lerp(global_gpu, local, local_alpha)
    blended = ((blended + blended.T) * 0.5).cpu().contiguous()
    del accumulator, local, global_gpu

    diag_mean = float(blended.diagonal().double().mean())
    if not math.isfinite(diag_mean) or diag_mean <= 1e-20:
        raise ValueError("shrunk expert-local H13 is degenerate")
    blended_sha256 = tensor_sha256(blended)
    evidence: dict[str, Any] = {
        "construction": H13_CONSTRUCTION,
        "role": "fit",
        "layer": runtime.layer,
        "expert": expert,
        "routed": routed.evidence(),
        "global_h13_evidence_id": global_h13.evidence_id,
        "global_h13_matrix_sha256": tensor_sha256(global_h13.matrix),
        "matrix_sha256": blended_sha256,
        "diagonal_mean": diag_mean,
        "shrinkage": {
            "policy": "weighted_oas_reliability_layer_global_prior",
            "effective_sample_size": float(reliability["effective_sample_size"]),
            "oas_shrinkage": float(reliability["oas_shrinkage"]),
            "local_alpha": local_alpha,
            "local_alpha_cap": LOCAL_ALPHA_CAP,
            "global_alpha": 1.0 - local_alpha,
        },
        "fit_only": True,
        "selection_used": False,
        "holdout_used": False,
        "fallback": False,
    }
    evidence_id = canonical_sha256(evidence)
    return (
        DenseHessian(
            matrix=blended,
            evidence_id=evidence_id,
            construction=H13_CONSTRUCTION,
            split_id="fit",
            normalization_count=1,
            routed_sample_count=routed.rows,
        ),
        {**evidence, "evidence_id": evidence_id},
    )


def main() -> int:
    args = _parser().parse_args()
    if not 0 <= args.start < args.end <= NUM_EXPERTS:
        raise ValueError("expert range must satisfy 0 <= start < end <= 256")
    if args.threads <= 0 or args.chunk_rows <= 0:
        raise ValueError("threads and chunk rows must be positive")

    thread_count = str(args.threads)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = thread_count
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the production SQG encoder requires CUDA")

    from scripts.encode_final_shard import _open_fast_sealed_runtime
    from src.calibration_hessian import apply_frozen_h2_shrinkage
    from src.fresh_pipeline_artifacts import validate_expert_artifact
    from src.fresh_pipeline_common import (
        atomic_json,
        canonical_sha256,
        sha256_file,
    )
    from src.fresh_pipeline_runner import (
        _config,
        _encode_one_expert,
        _load_bound_kquant_runtime,
        _load_h13,
        _load_permutation,
        _load_profile_selection,
    )
    from src.glm52_fresh_sqg import DenseHessian, prepare_dense_h_session
    from src.glm52_fresh_sqg.reference import tensor_sha256
    import src.glm52_fresh_sqg.codec as codec

    started = time.monotonic()
    runtime = _open_fast_sealed_runtime(
        args.preflight, layer=args.layer, device=args.device
    )
    output_root = args.output_root.resolve()
    if output_root == runtime.paths.output_root.resolve():
        raise ValueError("directional output must differ from the sealed SQG root")
    output_dir = output_root / f"layer_{args.layer:03d}" / "experts"
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_path = runtime.profile_dir / "selection.json"
    selection, gate_profile, down_profile = _load_profile_selection(runtime)
    selection_sha256 = sha256_file(selection_path)
    global_h13, _ = _load_h13(runtime)
    kquant_runtime = _load_bound_kquant_runtime(runtime)

    # Keep production validation active while declaring the new experimental
    # construction honestly in every tensor manifest.
    codec.PRODUCTION_H13_CONSTRUCTION = H13_CONSTRUCTION

    completed = 0
    skipped = 0
    alphas: list[float] = []
    for expert in range(args.start, args.end):
        manifest_path, shard_path, seal_path = _artifact_paths(
            output_dir, args.layer, expert
        )
        existence = (manifest_path.exists(), shard_path.exists(), seal_path.exists())
        if any(existence):
            if not all(existence):
                raise ValueError(f"partial expert artifact exists: {manifest_path}")
            existing = validate_expert_artifact(manifest_path)
            gate_id = f"model.layers.{args.layer}.mlp.experts.{expert}.gate_proj"
            dense_h = existing["tensor_manifests"][gate_id]["dense_h"]
            if (
                existing.get("purpose") != "final_treatment"
                or dense_h.get("construction") != H13_CONSTRUCTION
            ):
                raise ValueError(f"expert {expert} resume artifact policy differs")
            alpha = float(
                existing["calibration"]["profile_cell"]["h13_shrinkage"][
                    "local_alpha"
                ]
            )
            alphas.append(alpha)
            skipped += 1
            continue

        weights = runtime.source.load_expert_bf16(
            runtime.layer, expert, device="cpu"
        )
        expert_h13, h13_evidence = _build_h13(
            runtime,
            global_h13,
            expert=expert,
            device=device,
            chunk_rows=args.chunk_rows,
            apply_frozen_h2_shrinkage=apply_frozen_h2_shrinkage,
            canonical_sha256=canonical_sha256,
            tensor_sha256=tensor_sha256,
            DenseHessian=DenseHessian,
        )
        permutation = _load_permutation(runtime, expert)
        gate_config = _config(
            runtime,
            weights,
            "gate_proj",
            permutation,
            gate_profile,
            down_profile,
        )
        session = prepare_dense_h_session(expert_h13, gate_config)
        _encode_one_expert(
            runtime,
            kquant_runtime,
            expert_h13,
            session,
            expert=expert,
            output_dir=output_dir,
            purpose="final_treatment",
            selection_evidence_sha256=selection_sha256,
            gate_profile=gate_profile,
            down_profile=down_profile,
            cell_evidence={
                "experiment": "expert_local_h13_oas_global_prior_r1",
                "selection_id": selection["selection_id"],
                "selection_sha256": selection_sha256,
                "selected_cell_id": selection["selected_cell_id"],
                "profiles_frozen": True,
                "permutation_frozen": True,
                "bit_map_frozen": True,
                "h13_shrinkage": h13_evidence["shrinkage"],
                "holdout_unseen": True,
            },
            cached_weights=weights,
        )
        validate_expert_artifact(manifest_path)
        alpha = float(h13_evidence["shrinkage"]["local_alpha"])
        alphas.append(alpha)
        completed += 1
        atomic_json(
            output_root
            / f"layer_{args.layer:03d}"
            / f"progress_{args.start:03d}_{args.end:03d}.json",
            {
                "schema": "glm52-expert-local-h13-shard-progress-v1",
                "complete": False,
                "layer": args.layer,
                "start": args.start,
                "end": args.end,
                "completed_through": expert,
                "newly_encoded": completed,
                "resumed": skipped,
                "local_alpha_min": min(alphas),
                "local_alpha_max": max(alphas),
                "elapsed_seconds": time.monotonic() - started,
            },
            overwrite=True,
        )
        del weights, expert_h13, session
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"layer {args.layer} experts {args.start}:{args.end}: "
            f"{expert + 1 - args.start}/{args.end - args.start} "
            f"alpha={alpha:.6f}",
            flush=True,
        )

    result = {
        "schema": "glm52-expert-local-h13-shard-result-v1",
        "complete": True,
        "layer": args.layer,
        "start": args.start,
        "end": args.end,
        "experts": args.end - args.start,
        "newly_encoded": completed,
        "resumed": skipped,
        "h13_construction": H13_CONSTRUCTION,
        "local_alpha_cap": LOCAL_ALPHA_CAP,
        "local_alpha_min": min(alphas),
        "local_alpha_mean": sum(alphas) / len(alphas),
        "local_alpha_max": max(alphas),
        "selection_sha256": selection_sha256,
        "output_dir": str(output_dir),
        "elapsed_seconds": time.monotonic() - started,
    }
    result_path = (
        output_root
        / f"layer_{args.layer:03d}"
        / f"result_{args.start:03d}_{args.end:03d}.json"
    )
    atomic_json(result_path, result, overwrite=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
