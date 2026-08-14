#!/usr/bin/env python3
"""Archive compact per-layer fit/selection/holdout quality and tail evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("tail metrics require nonempty document scores")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _role(record: dict) -> dict:
    documents = record.get("document_scores")
    if not isinstance(documents, list) or not documents:
        raise ValueError("quality role lacks document-disjoint scores")
    values = [float(item["relative_error"]) for item in documents]
    worst_count = max(1, math.ceil(len(values) * 0.01))
    return {
        "role_rows": int(record["role_rows"]),
        "documents": len(values),
        "aggregation": record["aggregation"],
        "aggregate_relative_error": float(record["aggregate_relative_error"]),
        "document_relative_error": {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
            "cvar1": statistics.fmean(sorted(values, reverse=True)[:worst_count]),
            "max": max(values),
            "cvar1_document_count": worst_count,
        },
    }


def main() -> int:
    args = _parser().parse_args()
    from src.fresh_pipeline_common import atomic_json, canonical_sha256, sha256_file

    layer_root = args.candidate_root.resolve() / f"layer_{args.layer:03d}"
    selection_path = layer_root / "coupled_selection_manifest.json"
    score_path = layer_root / "selected_selection_holdout_score.json"
    summary_path = layer_root / "coupled_result_summary.json"
    selection = json.loads(selection_path.read_text())
    score = json.loads(score_path.read_text())
    summary = json.loads(summary_path.read_text())
    profile_binding = selection.get("final_profile_binding")
    if (
        selection.get("schema") != "glm52-coupled-mixed-rate-selection-v4"
        or score.get("schema")
        != "glm52-coupled-mixed-rate-signed-top8-score-v4"
        or summary.get("schema") != "glm52-coupled-mixed-rate-layer-result-v2"
        or selection.get("complete") is not True
        or score.get("complete") is not True
        or summary.get("complete") is not True
        or {selection.get("layer"), score.get("layer"), summary.get("layer")}
        != {args.layer}
        or score.get("selection_id") != selection.get("selection_id")
        or summary.get("selection_id") != selection.get("selection_id")
        or selection.get("allocation_binding") != score.get("allocation_binding")
        or selection.get("allocation_binding") != summary.get("allocation_binding")
        or not isinstance(profile_binding, dict)
        or selection.get("final_profile_binding")
        != score.get("final_profile_binding")
        or selection.get("final_profile_binding")
        != summary.get("final_profile_binding")
    ):
        raise ValueError("coupled layer quality bindings differ")
    archive = {
        "schema": "glm52-coupled-layer-quality-tails-v2",
        "complete": True,
        "layer": args.layer,
        "selection_id": selection["selection_id"],
        "score_id": score["score_id"],
        "draw_histogram": selection["draw_histogram"],
        "selection_policy": selection["selection_policy"],
        "holdout_used_for_selection": False,
        "rate_contract": "independent_per_tensor_k3_k4",
        "allocation_binding": selection["allocation_binding"],
        "selected_beta": selection["selected_beta"],
        "final_profile_binding": profile_binding,
        "roles": {
            role: _role(score["roles"][role]) for role in ("selection", "holdout")
        },
        "source_files": {
            selection_path.name: sha256_file(selection_path),
            score_path.name: sha256_file(score_path),
            summary_path.name: sha256_file(summary_path),
        },
    }
    archive["archive_id"] = canonical_sha256(archive)
    atomic_json(args.output, archive)
    print(json.dumps(archive, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
