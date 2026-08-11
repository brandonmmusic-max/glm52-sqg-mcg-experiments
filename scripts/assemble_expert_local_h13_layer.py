#!/usr/bin/env python3
"""Assemble one expert-local-H13 directional layer from sealed expert shards."""

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
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--local-alpha",
        type=float,
        default=None,
        help="Exact fixed alpha for the blend ablation; omit for the original OAS run.",
    )
    args = parser.parse_args()
    if args.local_alpha is not None and not 0.0 <= args.local_alpha <= 1.0:
        raise ValueError("--local-alpha must be in [0,1]")

    from scripts.encode_expert_local_h13_shard import _fixed_alpha_construction

    from scripts.encode_final_shard import _open_fast_sealed_runtime
    from src.fresh_pipeline_artifacts import (
        assemble_layer_artifact,
        validate_layer_artifact,
    )
    from src.fresh_pipeline_common import sha256_file
    from src.fresh_pipeline_runner import _load_profile_selection

    runtime = _open_fast_sealed_runtime(
        args.preflight, layer=args.layer, device=args.device
    )
    root = args.output_root.resolve()
    if root == runtime.paths.output_root.resolve():
        raise ValueError("directional output must differ from baseline")
    layer_root = root / f"layer_{args.layer:03d}"
    expert_dir = layer_root / "experts"
    if len(list(expert_dir.glob(f"layer-{args.layer:03d}-expert-*.json"))) != 256:
        raise RuntimeError("layer does not contain exactly 256 expert manifests")

    selection, gate_profile, down_profile = _load_profile_selection(runtime)
    source_selection = runtime.profile_dir / "selection.json"
    profile_dir = layer_root / "profile_search"
    profile_dir.mkdir(parents=True, exist_ok=True)
    destination_selection = profile_dir / "selection.json"
    source_bytes = source_selection.read_bytes()
    if destination_selection.exists():
        if destination_selection.read_bytes() != source_bytes:
            raise ValueError("copied profile selection differs")
    else:
        destination_selection.write_bytes(source_bytes)

    h13_construction = _fixed_alpha_construction(args.local_alpha)
    import src.glm52_fresh_sqg.codec as codec

    codec.PRODUCTION_H13_CONSTRUCTION = h13_construction
    manifest = assemble_layer_artifact(
        expert_dir,
        layer_root / "final",
        run_id=runtime.settings.run_id,
        layer=args.layer,
        bit_map=runtime.bit_map,
        gate_up_profile=gate_profile,
        down_profile=down_profile,
        layer_evidence={
            "experiment": (
                "expert_local_h13_oas_global_prior_r1"
                if args.local_alpha is None
                else "expert_local_h13_fixed_alpha_ablation_r2"
            ),
            "h13_construction": h13_construction,
            "requested_local_alpha": args.local_alpha,
            "baseline_preflight_id": runtime.preflight["preflight_id"],
            "profile_selection_sha256": sha256_file(source_selection),
            "profile_selection_id": selection["selection_id"],
            "selected_cell_id": selection["selected_cell_id"],
            "source_profiles_frozen": True,
            "source_permutations_frozen": True,
            "source_bit_map_frozen": True,
            "fit_calibration_complete": True,
            "selection_used_by_encoder": False,
            "holdout_used_by_encoder": False,
            "candidate_conditioned_h2_rebuilt": True,
            "mcg_inputs": 0,
        },
    )
    path = layer_root / "final" / f"fresh-sqg-layer-{args.layer:03d}.json"
    validated = validate_layer_artifact(path)
    result = {
        "complete": True,
        "layer": args.layer,
        "layer_manifest": str(path),
        "layer_manifest_sha256": sha256_file(path),
        "layer_shard_sha256": validated["shard_sha256"],
        "selection_sha256": sha256_file(destination_selection),
        "selection_id": selection["selection_id"],
        "bit_histogram": manifest["bit_histogram"],
        "sqg_tensors": manifest["lineage"]["sqg_tensor_count"],
        "mcg_tensors": manifest["lineage"]["mcg_tensor_count"],
    }
    (layer_root / "assembly_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
