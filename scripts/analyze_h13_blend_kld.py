#!/usr/bin/env python3
"""Compare five-boot H13 blend KLD arms with explicit positive-tail gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_h13e_kld_pair import (
    circular_block_bootstrap,
    load_arm,
    load_summary,
)


def _parse_arm(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("arm must be LABEL=/absolute/summary.json")
    label, path_raw = raw.split("=", 1)
    if not label or not label.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError("arm label is invalid")
    path = Path(path_raw)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("arm summary must be absolute")
    return label, path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "p99_5": float(np.quantile(values, 0.995)),
        "p99_9": float(np.quantile(values, 0.999)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _upper_cvar(values: np.ndarray, fraction: float = 0.01) -> float:
    count = max(1, math.ceil(values.size * fraction))
    boundary = values.size - count
    return float(np.partition(values, boundary)[boundary:].mean(dtype=np.float64))


def _render(report: dict[str, Any]) -> str:
    lines = [
        "# H13 blend final-logit KLD tail comparison",
        "",
        f"Calibration-selected arm: **`{report['calibration_winner']}`**; "
        f"confirmation status: **`{report['confirmation']['status']}`**.",
        "",
        "| Arm | Mean KLD | Mean delta | Positions improved | KLD CVaR 1% | KLD p99 | Eligible |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metric in report["metrics"].items():
        lines.append(
            f"| {label} | {metric['mean_kld']:.9e} | "
            f"{metric['mean_delta_vs_baseline']:.9e} | "
            f"{metric['improved_fraction']:.3%} | "
            f"{metric['kld_distribution']['upper_cvar_1pct']:.9e} | "
            f"{metric['kld_distribution']['p99']:.9e} | "
            f"{str(metric['passes_tail_and_mean_constraints']).lower()} |"
        )
    lines.extend(
        [
            "",
            "Eligibility requires mean KLD, worst-1% KLD CVaR, and p99 KLD all to be no worse than the alpha-0 baseline. Position win rate is descriptive and is never allowed to override a failed tail gate.",
            "",
            "## Uncertainty",
            "",
            "| Arm | Boot-noise SE | Welch t | Boot direction | Position-block 95% CI | Position direction |",
            "|---|---:|---:|---|---:|---|",
        ]
    )
    for label, value in report["uncertainty"]["comparisons"].items():
        interval = value["circular_block_bootstrap_position_delta_mean_95pct"]
        welch_t = value["welch_t"]
        lines.append(
            f"| {label} | {value['repeat_noise_standard_error']:.9e} | "
            f"{('undefined' if welch_t is None else f'{welch_t:.6f}')} | "
            f"{value['repeat_noise_direction']} | "
            f"[{interval['lower']:.9e}, {interval['upper']:.9e}] | "
            f"{interval['interpretation']} |"
        )
    lines.extend(
        [
            "",
            report["uncertainty"]["note"],
            "",
            "The exploratory best arm is reported for diagnosis only. The calibration-selected arm is the only preregistered confirmation target; choosing another arm from this fixed prompt would make the prompt selection data.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", type=_parse_arm, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--calibration-winner", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-block-size", type=int, default=32)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    args = parser.parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")
    if not 1 <= args.bootstrap_block_size <= 2047:
        raise ValueError("bootstrap block size is invalid")
    arms = dict(args.arm)
    if len(arms) != len(args.arm):
        raise ValueError("arm labels must be unique")
    if args.baseline_label not in arms or args.calibration_winner not in arms:
        raise ValueError("baseline and calibration winner must both be supplied")

    summaries: dict[str, dict[str, Any]] = {}
    summary_hashes: dict[str, str] = {}
    vectors: dict[str, np.ndarray] = {}
    evidence: dict[str, Any] = {}
    identity: dict[str, Any] | None = None
    for label, path in arms.items():
        summary, digest = load_summary(path.resolve(strict=True))
        tensor, tensor_evidence = load_arm(summary)
        current_identity = {
            "reference_sha256": summary.get("reference_sha256"),
            "reference_token_ids_u32le_sha256": summary.get(
                "reference_token_ids_u32le_sha256"
            ),
            "regime": summary.get("regime"),
            "runtime_image": summary.get("runtime", {}).get("candidate_image_id"),
            "runtime_overlay_manifest": summary.get("runtime", {})
            .get("runtime_overlay", {})
            .get("manifest_sha256"),
        }
        if identity is None:
            identity = current_identity
        elif identity != current_identity:
            raise ValueError(f"arm {label} evaluation identity differs")
        summaries[label] = summary
        summary_hashes[label] = digest
        vectors[label] = tensor.mean(dim=0).numpy().astype(np.float64, copy=False)
        evidence[label] = tensor_evidence

    baseline = vectors[args.baseline_label]
    baseline_tail = {
        "upper_cvar_1pct": _upper_cvar(baseline),
        "p99": float(np.quantile(baseline, 0.99)),
    }
    metrics: dict[str, Any] = {}
    for label, values in vectors.items():
        delta = values - baseline
        distribution = {
            **_quantiles(values),
            "mean": float(values.mean(dtype=np.float64)),
            "upper_cvar_1pct": _upper_cvar(values),
        }
        eligible = (
            float(values.mean(dtype=np.float64))
            <= float(baseline.mean(dtype=np.float64))
            and distribution["upper_cvar_1pct"]
            <= baseline_tail["upper_cvar_1pct"]
            and distribution["p99"] <= baseline_tail["p99"]
        )
        metrics[label] = {
            "mean_kld": float(values.mean(dtype=np.float64)),
            "mean_delta_vs_baseline": float(delta.mean(dtype=np.float64)),
            "improved_positions": int((delta < 0.0).sum()),
            "worsened_positions": int((delta > 0.0).sum()),
            "improved_fraction": float((delta < 0.0).mean()),
            "median_delta_vs_baseline": float(np.median(delta)),
            "kld_distribution": distribution,
            "delta_distribution": {
                **_quantiles(delta),
                "upper_cvar_1pct": _upper_cvar(delta),
                "positive_sum": float(np.maximum(delta, 0.0).sum(dtype=np.float64)),
                "negative_sum": float(np.minimum(delta, 0.0).sum(dtype=np.float64)),
            },
            "passes_tail_and_mean_constraints": eligible,
        }

    nonbaseline = [label for label in arms if label != args.baseline_label]
    repeat_and_position_uncertainty: dict[str, Any] = {}
    baseline_summary = summaries[args.baseline_label]["candidate_result"]
    baseline_runs = int(baseline_summary["runs"])
    baseline_sd = float(baseline_summary["sample_sd_kld"])
    for offset, label in enumerate(nonbaseline):
        candidate_summary = summaries[label]["candidate_result"]
        candidate_runs = int(candidate_summary["runs"])
        candidate_sd = float(candidate_summary["sample_sd_kld"])
        scalar_delta = float(
            candidate_summary["mean_kld"] - baseline_summary["mean_kld"]
        )
        baseline_term = baseline_sd**2 / baseline_runs
        candidate_term = candidate_sd**2 / candidate_runs
        repeat_se = math.sqrt(baseline_term + candidate_term)
        welch_t = scalar_delta / repeat_se if repeat_se else None
        denominator = (
            baseline_term**2 / max(1, baseline_runs - 1)
            + candidate_term**2 / max(1, candidate_runs - 1)
        )
        welch_df = (
            (baseline_term + candidate_term) ** 2 / denominator
            if denominator
            else None
        )
        position_delta = torch.from_numpy(
            vectors[label] - vectors[args.baseline_label]
        ).to(torch.float64)
        lower, upper = circular_block_bootstrap(
            position_delta,
            iterations=args.bootstrap_iterations,
            block_size=args.bootstrap_block_size,
            seed=args.bootstrap_seed + offset,
        )
        repeat_and_position_uncertainty[label] = {
            "baseline_runs": baseline_runs,
            "candidate_runs": candidate_runs,
            "baseline_sample_sd_kld": baseline_sd,
            "candidate_sample_sd_kld": candidate_sd,
            "candidate_minus_baseline_mean_kld": scalar_delta,
            "repeat_noise_standard_error": repeat_se,
            "welch_t": welch_t,
            "welch_degrees_freedom": welch_df,
            "repeat_noise_direction": (
                "undefined"
                if welch_t is None
                else (
                    "inconclusive"
                    if abs(welch_t) < 2.776
                    else ("higher" if scalar_delta > 0 else "lower")
                )
            ),
            "circular_block_bootstrap_position_delta_mean_95pct": {
                "iterations": args.bootstrap_iterations,
                "block_size": args.bootstrap_block_size,
                "seed": args.bootstrap_seed + offset,
                "lower": lower,
                "upper": upper,
                "interpretation": (
                    "lower"
                    if upper < 0.0
                    else ("higher" if lower > 0.0 else "inconclusive")
                ),
            },
        }
    eligible = [
        label for label in nonbaseline if metrics[label]["passes_tail_and_mean_constraints"]
    ]
    if eligible:
        exploratory_best = max(
            eligible,
            key=lambda label: (
                metrics[label]["improved_fraction"],
                -metrics[label]["kld_distribution"]["upper_cvar_1pct"],
                -metrics[label]["mean_kld"],
            ),
        )
    else:
        exploratory_best = min(
            nonbaseline,
            key=lambda label: (
                metrics[label]["kld_distribution"]["upper_cvar_1pct"],
                metrics[label]["mean_kld"],
                -metrics[label]["improved_fraction"],
            ),
        )
    confirmed = metrics[args.calibration_winner]["passes_tail_and_mean_constraints"]
    report: dict[str, Any] = {
        "schema": "glm52-h13-blend-final-kld-tail-analysis-v1",
        "baseline_label": args.baseline_label,
        "calibration_winner": args.calibration_winner,
        "identity": identity,
        "summary_sha256": summary_hashes,
        "per_position_inputs": evidence,
        "selection_policy": {
            "hard_constraints": [
                "mean KLD <= alpha0 baseline",
                "upper-CVaR 1% per-position KLD <= alpha0 baseline",
                "p99 per-position KLD <= alpha0 baseline",
            ],
            "position_win_rate_only_after_hard_constraints": True,
            "exploratory_eligible": eligible,
            "exploratory_best": exploratory_best,
        },
        "uncertainty": {
            "comparisons": repeat_and_position_uncertainty,
            "note": (
                "Welch statistics describe independent fresh-boot variation. "
                "The circular block bootstrap describes correlated positions "
                "within the one fixed prompt and is not a document-generalization interval."
            ),
        },
        "confirmation": {
            "status": (
                "calibration_winner_passed_final_kld_tail_and_mean_constraints"
                if confirmed
                else "calibration_winner_failed_one_or_more_final_kld_constraints"
            ),
            "passed": bool(confirmed),
        },
        "metrics": metrics,
    }

    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = output.with_suffix(".npz")
    markdown = output.with_suffix(".md")
    for path in (arrays, markdown):
        if path.exists() or path.is_symlink():
            raise ValueError(f"output already exists: {path}")
    np.savez_compressed(
        arrays,
        **{f"mean_position_kld__{label}": value.astype(np.float32) for label, value in vectors.items()},
    )
    report["per_position_means"] = {"path": str(arrays), "sha256": _sha256(arrays)}
    partial = Path(f"{output}.partial")
    partial.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(partial, output)
    markdown.write_text(_render(report), encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
