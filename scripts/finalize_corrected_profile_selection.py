#!/usr/bin/env python3
"""Validate all corrected cells and freeze one profile selection per layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from scripts.encode_final_shard import _open_fast_sealed_runtime
    from src.fresh_pipeline_common import load_json_object, sha256_file
    from src.fresh_pipeline_runner import search_profiles

    runtime = _open_fast_sealed_runtime(
        args.preflight, layer=args.layer, device=args.device
    )
    smoke = load_json_object(runtime.paths.output_root / "absolute_gate_scale_smoke.json")
    if (
        smoke.get("complete") is not True
        or smoke.get("preflight_id") != runtime.preflight["preflight_id"]
    ):
        raise ValueError("selection finalization is blocked on corrected real smoke")
    selection = search_profiles(runtime)
    selection_path = runtime.profile_dir / "selection.json"
    result = {
        "complete": True,
        "layer": args.layer,
        "selection_id": selection["selection_id"],
        "selection_sha256": sha256_file(selection_path),
        "selected_cell_id": selection["selected_cell_id"],
        "factorial_cell_count": selection["factorial_cell_count"],
        "all_cells_exact_sqg": selection["all_cells_exact_sqg"],
        "all_cells_candidate_conditional_h2": selection[
            "all_cells_candidate_conditional_h2"
        ],
        "all_cells_exact_down_sqg": selection["all_cells_exact_down_sqg"],
        "mcg_inputs": selection["mcg_inputs"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
