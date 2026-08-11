#!/usr/bin/env python3
"""Select a tail-safe per-layer H13 blend from completed panel outputs.

This is an exploratory refinement of the preregistered single-alpha panel.
It does not re-score expert functions: it combines the exact per-position
signed-top8 squared-error vectors already emitted for each layer and alpha.
The search space is small (five choices across four layers = 625 arms).
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.score_signed_top8_blends import tail_comparison  # noqa: E402


MINIMUM_RELATIVE_NMSE_GAIN = 1e-3


def _passes(
    metric: dict[str, Any],
    baseline_nmse: float,
    baseline_tail: dict[str, Any],
    *,
    minimum_relative_nmse_gain: float = 0.0,
) -> bool:
    tail = metric["tail_vs_baseline"]
    absolute = tail["candidate_squared_error"]
    relative = tail["candidate_relative_error"]
    baseline_absolute = baseline_tail["baseline_squared_error"]
    baseline_relative = baseline_tail["baseline_relative_error"]
    return bool(
        metric["signed_top8_nmse"]
        <= baseline_nmse * (1.0 - minimum_relative_nmse_gain)
        and absolute["upper_cvar_1pct"]
        <= baseline_absolute["upper_cvar_1pct"]
        and absolute["p99"] <= baseline_absolute["p99"]
        and relative["upper_cvar_1pct"]
        <= baseline_relative["upper_cvar_1pct"]
        and relative["p99"] <= baseline_relative["p99"]
    )


def select_layerwise(
    source: dict[str, Any], *, baseline_label: str
) -> dict[str, Any]:
    layers = sorted(int(layer) for layer in source["layers"])
    labels = list(source["aggregate"]["metrics"])
    if baseline_label not in labels:
        raise ValueError("baseline label is absent")

    reference: dict[int, np.ndarray] = {}
    errors: dict[int, dict[str, np.ndarray]] = {}
    identity: tuple[np.ndarray, np.ndarray] | None = None
    for layer in layers:
        path = Path(source["layers"][str(layer)]["row_output"])
        with np.load(path) as payload:
            docs = payload["doc_epochs"].copy()
            positions = payload["token_positions"].copy()
            if identity is None:
                identity = (docs, positions)
            elif not (
                np.array_equal(identity[0], docs)
                and np.array_equal(identity[1], positions)
            ):
                raise RuntimeError("layer row identities do not align")
            reference[layer] = payload["reference_energy"].astype(
                np.float64, copy=True
            )
            errors[layer] = {
                label: payload[f"error__{label}"].astype(np.float64, copy=True)
                for label in labels
            }

    reference_total = sum(reference.values())
    baseline_error = sum(errors[layer][baseline_label] for layer in layers)
    denominator = float(reference_total.sum(dtype=np.float64))
    baseline_nmse = float(baseline_error.sum(dtype=np.float64)) / denominator
    baseline_tail = tail_comparison(
        baseline_error, baseline_error, reference_total
    )

    combinations = []
    for choices in product(labels, repeat=len(layers)):
        error = sum(errors[layer][label] for layer, label in zip(layers, choices))
        tail = tail_comparison(error, baseline_error, reference_total)
        metric = {
            "signed_top8_nmse": float(error.sum(dtype=np.float64)) / denominator,
            "tail_vs_baseline": tail,
        }
        mapping = {str(layer): label for layer, label in zip(layers, choices)}
        nonbaseline_layers = sum(label != baseline_label for label in choices)
        eligible = nonbaseline_layers > 0 and _passes(
            metric,
            baseline_nmse,
            baseline_tail,
            minimum_relative_nmse_gain=MINIMUM_RELATIVE_NMSE_GAIN,
        )
        combinations.append(
            {
                "layer_to_label": mapping,
                "nonbaseline_layers": nonbaseline_layers,
                "eligible": eligible,
                **metric,
            }
        )

    eligible = [item for item in combinations if item["eligible"]]
    if eligible:
        winner = max(
            eligible,
            key=lambda item: (
                item["tail_vs_baseline"]["improved_fraction"],
                -item["tail_vs_baseline"]["candidate_squared_error"][
                    "upper_cvar_1pct"
                ],
                -item["tail_vs_baseline"]["candidate_relative_error"][
                    "upper_cvar_1pct"
                ],
                -item["signed_top8_nmse"],
            ),
        )
        status = "tail_and_mean_constraints_passed"
    else:
        winner = next(
            item for item in combinations if item["nonbaseline_layers"] == 0
        )
        status = "no_nonbaseline_combination_passed_baseline_retained"

    uniform_label = source["aggregate"]["selection_policy"].get(
        "winner", baseline_label
    )
    uniform_metric = next(
        item
        for item in combinations
        if all(
            label == uniform_label
            for label in item["layer_to_label"].values()
        )
    )
    uniform_tail = uniform_metric["tail_vs_baseline"]
    winner_tail = winner["tail_vs_baseline"]
    nmse_gain_vs_uniform = (
        uniform_metric["signed_top8_nmse"] - winner["signed_top8_nmse"]
    ) / uniform_metric["signed_top8_nmse"]
    position_gain_vs_uniform = (
        winner_tail["improved_fraction"] - uniform_tail["improved_fraction"]
    )
    tail_no_worse_than_uniform = bool(
        winner_tail["candidate_squared_error"]["upper_cvar_1pct"]
        <= uniform_tail["candidate_squared_error"]["upper_cvar_1pct"]
        and winner_tail["candidate_squared_error"]["p99"]
        <= uniform_tail["candidate_squared_error"]["p99"]
        and winner_tail["candidate_relative_error"]["upper_cvar_1pct"]
        <= uniform_tail["candidate_relative_error"]["upper_cvar_1pct"]
        and winner_tail["candidate_relative_error"]["p99"]
        <= uniform_tail["candidate_relative_error"]["p99"]
    )
    advances_over_uniform = bool(
        tail_no_worse_than_uniform
        and (
            nmse_gain_vs_uniform >= 1e-3
            or position_gain_vs_uniform >= 1e-2
        )
    )
    recommended_mapping = (
        winner["layer_to_label"]
        if advances_over_uniform
        else {str(layer): uniform_label for layer in layers}
    )

    return {
        "analysis_type": "exploratory_layerwise_refinement",
        "source_role": source["aggregate"]["role"],
        "layers": layers,
        "labels": labels,
        "baseline_label": baseline_label,
        "search_space": len(combinations),
        "eligible_count": len(eligible),
        "minimum_relative_nmse_gain": MINIMUM_RELATIVE_NMSE_GAIN,
        "status": status,
        "winner": winner,
        "uniform_comparator": {
            "label": uniform_label,
            "signed_top8_nmse": uniform_metric["signed_top8_nmse"],
            "improved_fraction": uniform_tail["improved_fraction"],
        },
        "advancement_over_uniform": {
            "tail_no_worse": tail_no_worse_than_uniform,
            "relative_nmse_gain": nmse_gain_vs_uniform,
            "improved_fraction_gain": position_gain_vs_uniform,
            "minimum_relative_nmse_gain": 1e-3,
            "minimum_improved_fraction_gain": 1e-2,
            "passed": advances_over_uniform,
        },
        "recommended_mapping": recommended_mapping,
        "recommended_source": (
            "exploratory_layerwise_winner"
            if advances_over_uniform
            else "preregistered_uniform_winner"
        ),
        "eligible_ranked": sorted(
            eligible,
            key=lambda item: (
                -item["tail_vs_baseline"]["improved_fraction"],
                item["tail_vs_baseline"]["candidate_squared_error"][
                    "upper_cvar_1pct"
                ],
                item["tail_vs_baseline"]["candidate_relative_error"][
                    "upper_cvar_1pct"
                ],
                item["signed_top8_nmse"],
            ),
        )[:25],
        "hard_constraints": [
            *source["aggregate"]["selection_policy"]["hard_constraints"],
            (
                "aggregate signed-top8 NMSE improves by at least "
                f"{MINIMUM_RELATIVE_NMSE_GAIN:.3%}"
            ),
        ],
        "limitation": (
            "The layerwise search and materiality correction are exploratory; "
            "the 0.1% floor was added after the first position-count ranking "
            "selected a numerically trivial 0.000032% NMSE change. The result "
            "requires holdout and final-logit confirmation."
        ),
    }


def _render(result: dict[str, Any]) -> str:
    winner = result["winner"]
    tail = winner["tail_vs_baseline"]
    lines = [
        "# Exploratory layerwise H13 blend selection",
        "",
        f"- Status: `{result['status']}`",
        f"- Search space: `{result['search_space']}` combinations",
        f"- Eligible: `{result['eligible_count']}`",
        f"- Winner: `{winner['layer_to_label']}`",
        f"- Signed top-8 NMSE: `{winner['signed_top8_nmse']:.9e}`",
        f"- Positions improved: `{tail['improved_fraction']:.3%}`",
        (
            "- Absolute-error worst-1% CVaR: "
            f"`{tail['candidate_squared_error']['upper_cvar_1pct']:.9e}`"
        ),
        (
            "- Relative-error worst-1% CVaR: "
            f"`{tail['candidate_relative_error']['upper_cvar_1pct']:.9e}`"
        ),
        (
            "- Advances over uniform winner: "
            f"`{result['advancement_over_uniform']['passed']}`"
        ),
        f"- Recommended mapping: `{result['recommended_mapping']}`",
        "",
        result["limitation"],
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline-label", default="alpha0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = select_layerwise(source, baseline_label=args.baseline_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(
        _render(result), encoding="utf-8"
    )
    print(_render(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
