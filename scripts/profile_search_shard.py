#!/usr/bin/env python3
"""Initialize or encode one of four disjoint corrected profile-search shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def assigned_cell_ids(cells: list[dict[str, object]], worker_index: int) -> tuple[str, ...]:
    if worker_index not in range(4):
        raise ValueError("profile worker index must be 0, 1, 2, or 3")
    active = [cell for cell in cells if int(cell["draw"]) < 4]
    selected = active[worker_index::4]
    if len(active) != 16 or len(selected) != 4:
        raise ValueError("corrected search requires 16 cells split four ways")
    return tuple(str(cell["cell_id"]) for cell in selected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=3)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize-only", action="store_true")
    mode.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(args.threads)

    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    from scripts.encode_final_shard import _open_fast_sealed_runtime
    from src.fresh_pipeline_common import (
        atomic_json,
        canonical_sha256,
        load_json_object,
        sha256_file,
    )
    from src.fresh_pipeline_evaluation import score_routed_artifacts
    from src.fresh_pipeline_runner import (
        ACTIVE_PROFILE_DRAWS,
        _encode_expert_sequence,
        _load_bound_kquant_runtime,
        _load_h13,
        _load_scale_evidence,
        _profiles_for_cell,
        _validate_score_file,
        _write_or_validate_preregistration,
    )
    from src.glm52_fresh_sqg.codec import realize_shared_residual_profile

    started = time.monotonic()
    runtime = _open_fast_sealed_runtime(
        args.preflight, layer=args.layer, device=args.device
    )
    smoke_path = runtime.paths.output_root / "absolute_gate_scale_smoke.json"
    smoke = load_json_object(smoke_path)
    if (
        smoke.get("complete") is not True
        or smoke.get("preflight_id") != runtime.preflight["preflight_id"]
    ):
        raise ValueError("corrected profile search is blocked on real K3/K4 smoke")
    h13, _ = _load_h13(runtime)
    scales = _load_scale_evidence(runtime)
    prereg = _write_or_validate_preregistration(runtime, scales, h13)
    prereg_path = runtime.profile_dir / "preregistration.json"
    prereg_sha256 = sha256_file(prereg_path)
    if args.initialize_only:
        result = {
            "complete": True,
            "mode": "initialize_only",
            "layer": args.layer,
            "preregistration_id": prereg["preregistration_id"],
            "preregistration_sha256": prereg_sha256,
            "active_cells": ACTIVE_PROFILE_DRAWS * 4,
            "elapsed_seconds": time.monotonic() - started,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    worker_index = int(args.worker_index)
    cell_ids = assigned_cell_ids(list(prereg["cells"]), worker_index)
    panel = tuple(int(value) for value in prereg["selection_panel"])
    cells = {str(cell["cell_id"]): cell for cell in prereg["cells"]}
    kquant_runtime = _load_bound_kquant_runtime(runtime)
    completed: list[dict[str, object]] = []
    for cell_id in cell_ids:
        cell = cells[cell_id]
        draw = int(cell["draw"])
        family = str(cell["family"])
        gate, down = _profiles_for_cell(runtime, scales, draw=draw, family=family)
        if (
            gate.manifest() != cell["gate_up_input_profile"]
            or down.manifest() != cell["down_output_profile"]
        ):
            raise RuntimeError(f"corrected preregistered profile differs: {cell_id}")
        gate = realize_shared_residual_profile(gate, runtime.device)
        down = realize_shared_residual_profile(down, runtime.device)
        cell_root = runtime.profile_dir / "cells" / cell_id
        artifact_dir = cell_root / "experts"
        sequence = _encode_expert_sequence(
            runtime,
            kquant_runtime,
            h13,
            order=panel,
            output_dir=artifact_dir,
            purpose="profile_search_cell",
            selection_evidence_sha256=prereg_sha256,
            gate_profile=gate,
            down_profile=down,
            cell_evidence={
                "preregistration_id": prereg["preregistration_id"],
                "cell_id": cell_id,
                "draw": draw,
                "family": family,
                "selection_panel": list(panel),
                "profile_choice_not_yet_made": True,
                "absolute_gate_scale_fix": True,
                "profile_worker_index": worker_index,
            },
        )
        score_path = cell_root / "score.json"
        if score_path.exists():
            score = _validate_score_file(
                runtime, score_path, panel=panel, artifact_dir=artifact_dir
            )
        else:
            score = score_routed_artifacts(
                runtime.capture,
                runtime.source,
                kquant_runtime,
                artifact_dir,
                cell_root / "score_scratch",
                role="selection",
                experts=panel,
                device=runtime.device,
                chunk_rows=runtime.settings.chunk_rows,
                cleanup_scratch=True,
            )
            score["factorial_cell"] = dict(cell)
            score.pop("score_id", None)
            score["score_id"] = canonical_sha256(score)
            atomic_json(score_path, score)
            score = _validate_score_file(
                runtime, score_path, panel=panel, artifact_dir=artifact_dir
            )
        completed.append(
            {
                "cell_id": cell_id,
                "score_id": score["score_id"],
                "score_sha256": sha256_file(score_path),
                "aggregate_relative_error": score["aggregate_relative_error"],
                "expert_count": sequence["expert_count"],
            }
        )

    result = {
        "complete": True,
        "mode": "profile_worker",
        "layer": args.layer,
        "worker_index": worker_index,
        "cell_ids": list(cell_ids),
        "cells_completed": completed,
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
