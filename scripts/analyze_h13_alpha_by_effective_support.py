#!/usr/bin/env python3
"""Relate the winning H13 local-alpha to fit-only effective routed support.

This is a post-grid diagnostic preregistered before the late uniform-alpha
selection result was available.  It uses gate-squared effective sample size
from sealed fit-only expert manifests and per-expert routed SSE from a scored
selection/holdout result.  It never chooses or changes encoded bytes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    begin = 0
    while begin < values.size:
        end = begin + 1
        while end < values.size and sorted_values[end] == sorted_values[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + end - 1)
        begin = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size < 2:
        raise ValueError("Spearman inputs must align and contain at least two rows")
    left_rank = _rank_average(left)
    right_rank = _rank_average(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _load_support(root: Path, layers: list[int]) -> dict[tuple[int, int], float]:
    support: dict[tuple[int, int], float] = {}
    for layer in layers:
        for expert in range(256):
            path = (
                root
                / f"layer_{layer:03d}"
                / "experts"
                / f"layer-{layer:03d}-expert-{expert:03d}.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            calibration = payload["calibration"]
            if not calibration["fit_only"] or calibration["holdout_used"]:
                raise ValueError(f"support manifest is not fit-only: {path}")
            shrinkage = calibration["profile_cell"]["h13_shrinkage"]
            value = float(shrinkage["effective_sample_size"])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"invalid effective sample size: {path}")
            support[(layer, expert)] = value
    return support


def analyze(
    scores: dict[str, Any],
    support: dict[tuple[int, int], float],
    *,
    baseline_label: str,
    alpha_by_label: dict[str, float],
) -> dict[str, Any]:
    layers = sorted(int(layer) for layer in scores["layers"])
    labels = [label for label, _ in sorted(alpha_by_label.items(), key=lambda x: x[1])]
    if baseline_label in labels:
        raise ValueError("baseline label must not be an SQG alpha label")

    records: list[dict[str, Any]] = []
    for layer in layers:
        per_expert = scores["layers"][str(layer)][
            "per_expert_individual_route_sse"
        ]
        required = {baseline_label, *labels}
        if not required <= set(per_expert):
            raise ValueError(f"layer {layer} lacks required per-expert labels")
        for expert in range(256):
            baseline = float(per_expert[baseline_label][expert])
            if not np.isfinite(baseline) or baseline <= 0.0:
                raise ValueError(f"invalid baseline SSE at layer {layer} expert {expert}")
            ratios = {
                label: float(per_expert[label][expert]) / baseline for label in labels
            }
            best_label = min(labels, key=lambda label: (ratios[label], alpha_by_label[label]))
            records.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "effective_sample_size": support[(layer, expert)],
                    "best_label": best_label,
                    "best_local_alpha": alpha_by_label[best_label],
                    "sse_ratio_to_mcg": ratios,
                }
            )

    n_eff = np.asarray([row["effective_sample_size"] for row in records])
    best_alpha = np.asarray([row["best_local_alpha"] for row in records])
    alpha0_label = min(labels, key=lambda label: alpha_by_label[label])
    alpha0_ratio = np.asarray(
        [row["sse_ratio_to_mcg"][alpha0_label] for row in records]
    )

    gain_correlations = {}
    for label in labels:
        ratio = np.asarray([row["sse_ratio_to_mcg"][label] for row in records])
        gain = (alpha0_ratio - ratio) / alpha0_ratio
        gain_correlations[label] = {
            "local_alpha": alpha_by_label[label],
            "spearman_effective_support_vs_gain_over_alpha0": _spearman(n_eff, gain),
            "mean_gain_over_alpha0": float(gain.mean(dtype=np.float64)),
        }

    ordered = np.argsort(n_eff, kind="mergesort")
    bins = []
    for index, indices in enumerate(np.array_split(ordered, 4), start=1):
        means = {
            label: float(
                np.mean(
                    [records[int(i)]["sse_ratio_to_mcg"][label] for i in indices],
                    dtype=np.float64,
                )
            )
            for label in labels
        }
        winner = min(labels, key=lambda label: (means[label], alpha_by_label[label]))
        bins.append(
            {
                "quartile": index,
                "experts": int(indices.size),
                "effective_sample_size_min": float(n_eff[indices].min()),
                "effective_sample_size_max": float(n_eff[indices].max()),
                "mean_sse_ratio_to_mcg": means,
                "winning_label": winner,
                "winning_local_alpha": alpha_by_label[winner],
            }
        )

    layer_summary = {}
    for layer in layers:
        selected = [row for row in records if row["layer"] == layer]
        means = {
            label: float(
                np.mean(
                    [row["sse_ratio_to_mcg"][label] for row in selected],
                    dtype=np.float64,
                )
            )
            for label in labels
        }
        winner = min(labels, key=lambda label: (means[label], alpha_by_label[label]))
        layer_summary[str(layer)] = {
            "effective_sample_size_median": float(
                np.median([row["effective_sample_size"] for row in selected])
            ),
            "mean_sse_ratio_to_mcg": means,
            "winning_label": winner,
            "winning_local_alpha": alpha_by_label[winner],
        }

    counts = {label: 0 for label in labels}
    for row in records:
        counts[row["best_label"]] += 1
    return {
        "schema": "glm52-h13-alpha-effective-support-analysis-v1",
        "analysis_role": scores["aggregate"]["role"],
        "preregistered_before_uniform_panel_result": True,
        "baseline_label": baseline_label,
        "alpha_by_label": alpha_by_label,
        "experts": len(records),
        "best_alpha_spearman_with_effective_support": _spearman(n_eff, best_alpha),
        "per_expert_winner_counts": counts,
        "gain_correlations": gain_correlations,
        "effective_support_quartiles": bins,
        "layers": layer_summary,
        "records": records,
        "interpretation_contract": {
            "positive_gain_correlation": (
                "higher effective support makes that alpha more favorable versus alpha0"
            ),
            "limitation": (
                "diagnostic association only; alpha bytes were encoded under frozen "
                "profiles/permutations selected by the layer-global arm"
            ),
        },
    }


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# H13 local-alpha versus effective routed support",
        "",
        f"- Role: `{result['analysis_role']}`",
        f"- Experts: `{result['experts']}`",
        (
            "- Spearman effective support vs per-expert winning alpha: "
            f"`{result['best_alpha_spearman_with_effective_support']:.6f}`"
        ),
        f"- Per-expert winners: `{result['per_expert_winner_counts']}`",
        "",
        "## Effective-support quartiles",
        "",
        "| Quartile | n_eff range | Winning alpha | Mean SSE ratios to MCG |",
        "|---:|---:|---:|---|",
    ]
    for row in result["effective_support_quartiles"]:
        lines.append(
            f"| {row['quartile']} | {row['effective_sample_size_min']:.1f}–"
            f"{row['effective_sample_size_max']:.1f} | "
            f"{row['winning_local_alpha']:.2f} | "
            f"`{row['mean_sse_ratio_to_mcg']}` |"
        )
    lines.extend(["", result["interpretation_contract"]["limitation"], ""])
    return "\n".join(lines)


def _parse_alpha(raw: str) -> tuple[str, float]:
    label, separator, value = raw.partition("=")
    if not separator or not label:
        raise argparse.ArgumentTypeError("alpha must be LABEL=VALUE")
    alpha = float(value)
    if not 0.0 <= alpha <= 1.0:
        raise argparse.ArgumentTypeError("alpha must be in [0,1]")
    return label, alpha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--baseline-label", default="mcg")
    parser.add_argument("--alpha", action="append", type=_parse_alpha, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    alpha_by_label = dict(args.alpha)
    if len(alpha_by_label) != len(args.alpha):
        raise ValueError("duplicate alpha label")
    layers = sorted(int(layer) for layer in scores["layers"])
    support = _load_support(args.support_root, layers)
    result = analyze(
        scores,
        support,
        baseline_label=args.baseline_label,
        alpha_by_label=alpha_by_label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(_render(result), encoding="utf-8")
    print(_render(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
