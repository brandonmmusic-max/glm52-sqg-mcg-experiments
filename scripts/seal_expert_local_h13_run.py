#!/usr/bin/env python3
"""Seal the four assembled expert-local-H13 SQG directional layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()

    from src.fresh_pipeline_artifacts import validate_layer_artifact
    from src.fresh_pipeline_common import (
        RUN_SEAL_SCHEMA,
        SELECTED_LAYERS,
        canonical_sha256,
        sha256_file,
    )

    root = args.root.resolve()
    layers: dict[str, object] = {}
    total_k3 = total_k4 = total_sqg = 0
    run_id: str | None = None
    for layer in SELECTED_LAYERS:
        layer_root = root / f"layer_{layer:03d}"
        manifest_path = (
            layer_root / "final" / f"fresh-sqg-layer-{layer:03d}.json"
        )
        manifest = validate_layer_artifact(manifest_path)
        selection_path = layer_root / "profile_search" / "selection.json"
        selection = json.loads(selection_path.read_text())
        if run_id is None:
            run_id = str(manifest["run_id"])
        elif manifest["run_id"] != run_id:
            raise ValueError("assembled layers have different run IDs")
        histogram = manifest["bit_histogram"]
        total_k3 += int(histogram["3"])
        total_k4 += int(histogram["4"])
        total_sqg += int(manifest["lineage"]["sqg_tensor_count"])
        layers[str(layer)] = {
            "layer_artifact": str(manifest_path),
            "layer_artifact_sha256": sha256_file(manifest_path),
            "layer_shard_sha256": manifest["shard_sha256"],
            "selection_sha256": sha256_file(selection_path),
            "selection_id": selection["selection_id"],
            "selected_cell_id": selection["selected_cell_id"],
            "sqg_tensors": manifest["lineage"]["sqg_tensor_count"],
            "mcg_tensors": manifest["lineage"]["mcg_tensor_count"],
            "K3": histogram["3"],
            "K4": histogram["4"],
            "holdout": {
                "status": "reserved_for_post_encode_directional_scoring",
                "used_by_encoder": False,
            },
        }
    if (total_k3, total_k4, total_sqg) != (1536, 1536, 3072):
        raise RuntimeError("four-layer SQG/bit census differs")
    seal: dict[str, object] = {
        "schema": RUN_SEAL_SCHEMA,
        "complete": True,
        "run_id": run_id,
        "preflight_sha256": sha256_file(args.preflight),
        "layers": layers,
        "census": {
            "layers": 4,
            "experts": 1024,
            "sqg_tensors": total_sqg,
            "mcg_tensors": 0,
            "K3": total_k3,
            "K4": total_k4,
            "other_rates": 0,
        },
        "lineage": {
            "official_bf16": True,
            "fresh_capture": True,
            "frozen_bit_map_only": True,
            "fresh_sqg_encoding": True,
            "expert_local_h13_oas_global_prior": True,
            "candidate_conditioned_h2_rebuilt": True,
            "legacy_mcg_artifact_reads": 0,
            "fallback_count": 0,
        },
        "scope": {
            "four_layer_replacement_artifacts": True,
            "runnable_model_materialized": False,
            "existing_model_mutated": False,
            "selection_and_holdout_used_by_encoder": False,
        },
    }
    seal["run_seal_id"] = canonical_sha256(seal)
    destination = root / "run_seal.json"
    destination.write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(seal["census"], indent=2, sort_keys=True))
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
