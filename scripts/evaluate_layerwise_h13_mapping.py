#!/usr/bin/env python3
"""Evaluate one frozen layerwise H13 mapping on a separate scored role."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.score_signed_top8_blends import tail_comparison  # noqa: E402
from scripts.select_layerwise_h13_blend import (  # noqa: E402
    MINIMUM_RELATIVE_NMSE_GAIN,
    _passes,
)


def evaluate(
    source: dict[str, Any], selection: dict[str, Any], *, baseline_label: str
) -> dict[str, Any]:
    layers = sorted(int(layer) for layer in source["layers"])
    mapping = selection.get(
        "recommended_mapping", selection["winner"]["layer_to_label"]
    )
    if set(mapping) != {str(layer) for layer in layers}:
        raise ValueError("frozen mapping layers differ from score layers")
    labels = set(source["aggregate"]["metrics"])
    if baseline_label not in labels or not set(mapping.values()) <= labels:
        raise ValueError("frozen mapping requires an unscored candidate")

    reference_total: np.ndarray | None = None
    baseline_error: np.ndarray | None = None
    candidate_error: np.ndarray | None = None
    identity: tuple[np.ndarray, np.ndarray] | None = None
    for layer in layers:
        row_output = Path(source["layers"][str(layer)]["row_output"])
        with np.load(row_output) as payload:
            docs = payload["doc_epochs"]
            positions = payload["token_positions"]
            if identity is None:
                identity = (docs.copy(), positions.copy())
            elif not (
                np.array_equal(identity[0], docs)
                and np.array_equal(identity[1], positions)
            ):
                raise RuntimeError("layer row identities do not align")
            reference = payload["reference_energy"].astype(np.float64, copy=False)
            baseline = payload[f"error__{baseline_label}"].astype(
                np.float64, copy=False
            )
            candidate = payload[f"error__{mapping[str(layer)]}"].astype(
                np.float64, copy=False
            )
            reference_total = (
                reference.copy()
                if reference_total is None
                else reference_total + reference
            )
            baseline_error = (
                baseline.copy()
                if baseline_error is None
                else baseline_error + baseline
            )
            candidate_error = (
                candidate.copy()
                if candidate_error is None
                else candidate_error + candidate
            )
    if reference_total is None or baseline_error is None or candidate_error is None:
        raise RuntimeError("no layer vectors were evaluated")

    denominator = float(reference_total.sum(dtype=np.float64))
    baseline_nmse = float(baseline_error.sum(dtype=np.float64)) / denominator
    candidate_nmse = float(candidate_error.sum(dtype=np.float64)) / denominator
    baseline_tail = tail_comparison(
        baseline_error, baseline_error, reference_total
    )
    candidate_tail = tail_comparison(
        candidate_error, baseline_error, reference_total
    )
    metric = {
        "signed_top8_nmse": candidate_nmse,
        "tail_vs_baseline": candidate_tail,
    }
    return {
        "analysis_type": "frozen_layerwise_mapping_evaluation",
        "role": source["aggregate"]["role"],
        "layers": layers,
        "mapping": mapping,
        "baseline_label": baseline_label,
        "baseline_signed_top8_nmse": baseline_nmse,
        "candidate": metric,
        "passes_all_hard_constraints": _passes(
            metric,
            baseline_nmse,
            baseline_tail,
            minimum_relative_nmse_gain=float(
                selection.get(
                    "minimum_relative_nmse_gain", MINIMUM_RELATIVE_NMSE_GAIN
                )
            ),
        ),
        "selection_source": {
            "analysis_type": selection["analysis_type"],
            "source_role": selection["source_role"],
            "selection_status": selection["status"],
        },
        "limitation": (
            "The existing holdout is encoder-unseen but analysis-seen from the "
            "earlier alpha0/alpha075 experiment."
        ),
    }


def _render(result: dict[str, Any]) -> str:
    candidate = result["candidate"]
    tail = candidate["tail_vs_baseline"]
    return "\n".join(
        [
            f"# Frozen layerwise H13 mapping — {result['role']}",
            "",
            f"- Mapping: `{result['mapping']}`",
            f"- Passes all hard constraints: `{result['passes_all_hard_constraints']}`",
            f"- Baseline signed top-8 NMSE: `{result['baseline_signed_top8_nmse']:.9e}`",
            f"- Candidate signed top-8 NMSE: `{candidate['signed_top8_nmse']:.9e}`",
            f"- Positions improved: `{tail['improved_fraction']:.3%}`",
            (
                "- Absolute-error worst-1% CVaR: "
                f"`{tail['candidate_squared_error']['upper_cvar_1pct']:.9e}`"
            ),
            (
                "- Relative-error worst-1% CVaR: "
                f"`{tail['candidate_relative_error']['upper_cvar_1pct']:.9e}`"
            ),
            "",
            result["limitation"],
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline-label", default="alpha0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.scores.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    result = evaluate(source, selection, baseline_label=args.baseline_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(_render(result), encoding="utf-8")
    print(_render(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
