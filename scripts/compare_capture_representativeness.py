#!/usr/bin/env python3
"""Compare reduced/full GLM calibration evidence on one routed layer.

This diagnostic checks matrix geometry and profile-selection stability.  It is
not by itself a causal corpus-size experiment because each search rebuilt its
own H13, residual profiles, signs, permutations, expert panel, and candidate
bytes.  A candidate cross-score on documents excluded from the reduced plan is
still required when the selected profile reverses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from scipy.stats import spearmanr


FULL_ROOT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-encode-absfix-r2/layer_077"
)
REDUCED_PREP_ROOT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-contig-late-a025-r1/layer_077"
)
REDUCED_FINAL_ROOT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/"
    "fresh-sqg-contig-late-final-a025-r1/layer_077"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / (
    "results/capture_representativeness_layer077_r1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_h13(path: Path) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if keys != ["h13"]:
            raise RuntimeError(f"unexpected H13 keys in {path}: {keys}")
        value = handle.get_tensor("h13").float().contiguous()
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise RuntimeError(f"H13 is not square: {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("H13 contains non-finite values")
    return value


def matrix_comparison(
    full: torch.Tensor,
    reduced: torch.Tensor,
    *,
    probes: int,
    seed: int,
) -> dict[str, Any]:
    if full.shape != reduced.shape or full.ndim != 2 or full.shape[0] != full.shape[1]:
        raise ValueError("matrices must have identical square shapes")
    if probes <= 0:
        raise ValueError("probes must be positive")
    full64 = full.double()
    reduced64 = reduced.double()
    difference64 = reduced64 - full64
    full_frob2 = float(full64.square().sum().item())
    reduced_frob2 = float(reduced64.square().sum().item())
    difference_frob2 = float(difference64.square().sum().item())
    inner = float((full64 * reduced64).sum().item())
    full_diag = torch.diagonal(full64)
    reduced_diag = torch.diagonal(reduced64)
    full_trace = float(full_diag.sum().item())
    reduced_trace = float(reduced_diag.sum().item())
    diagonal_correlation = float(
        np.corrcoef(full_diag.numpy(), reduced_diag.numpy())[0, 1]
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(
        0,
        2,
        (full.shape[0], probes),
        generator=generator,
        dtype=torch.int8,
    ).float()
    signs.mul_(2.0).sub_(1.0)
    full_q = (signs * (full @ signs)).sum(dim=0).double()
    reduced_q = (signs * (reduced @ signs)).sum(dim=0).double()
    if not bool((full_q > 0).all() and (reduced_q > 0).all()):
        raise RuntimeError("non-positive H13 random-probe quadratic form")
    ratios = (reduced_q / full_q).numpy()

    full_symmetry = float((full - full.T).abs().max().item())
    reduced_symmetry = float((reduced - reduced.T).abs().max().item())
    return {
        "shape": list(full.shape),
        "full": {
            "trace": full_trace,
            "diagonal_mean": float(full_diag.mean().item()),
            "frobenius_norm": full_frob2**0.5,
            "frobenius_effective_rank": full_trace * full_trace / full_frob2,
            "maximum_symmetry_error": full_symmetry,
        },
        "reduced": {
            "trace": reduced_trace,
            "diagonal_mean": float(reduced_diag.mean().item()),
            "frobenius_norm": reduced_frob2**0.5,
            "frobenius_effective_rank": reduced_trace
            * reduced_trace
            / reduced_frob2,
            "maximum_symmetry_error": reduced_symmetry,
        },
        "relative": {
            "trace_change_percent": (reduced_trace / full_trace - 1.0) * 100.0,
            "diagonal_mean_change_percent": (
                float(reduced_diag.mean().item()) / float(full_diag.mean().item())
                - 1.0
            )
            * 100.0,
            "diagonal_pearson_correlation": diagonal_correlation,
            "frobenius_difference_over_full": (
                difference_frob2 / full_frob2
            )
            ** 0.5,
            "frobenius_cosine": inner
            / (full_frob2 * reduced_frob2) ** 0.5,
            "effective_rank_change_percent": (
                (reduced_trace * reduced_trace / reduced_frob2)
                / (full_trace * full_trace / full_frob2)
                - 1.0
            )
            * 100.0,
        },
        "random_probe_rayleigh_ratio_reduced_over_full": {
            "probes": probes,
            "seed": seed,
            "minimum": float(np.min(ratios)),
            "p05": float(np.quantile(ratios, 0.05)),
            "median": float(np.median(ratios)),
            "mean": float(np.mean(ratios)),
            "p95": float(np.quantile(ratios, 0.95)),
            "maximum": float(np.max(ratios)),
        },
    }


def profile_comparison(full: dict[str, Any], reduced: dict[str, Any]) -> dict[str, Any]:
    full_metrics = full["cell_metrics"]
    reduced_metrics = reduced["cell_metrics"]
    common = sorted(set(full_metrics) & set(reduced_metrics))
    if len(common) != 16:
        raise RuntimeError(f"expected 16 common profile cells, got {len(common)}")
    full_errors = [float(full_metrics[cell]["mean_document_relative_error"]) for cell in common]
    reduced_errors = [
        float(reduced_metrics[cell]["mean_document_relative_error"]) for cell in common
    ]
    correlation = spearmanr(full_errors, reduced_errors)
    full_selected = str(full["selected_cell_id"])
    reduced_selected = str(reduced["selected_cell_id"])
    full_panel = set(map(int, full["selection_panel"]))
    reduced_panel = set(map(int, reduced["selection_panel"]))

    cells: dict[str, Any] = {}
    for cell in (full_selected, reduced_selected):
        cells[cell] = {
            "full_mean_document_relative_error": float(
                full_metrics[cell]["mean_document_relative_error"]
            ),
            "reduced_mean_document_relative_error": float(
                reduced_metrics[cell]["mean_document_relative_error"]
            ),
            "full_rank": int(full["ranked_cell_ids"].index(cell) + 1),
            "reduced_rank": int(reduced["ranked_cell_ids"].index(cell) + 1),
            "full_bootstrap_vs_own_baseline": full_metrics[cell][
                "bootstrap_vs_baseline"
            ],
            "reduced_bootstrap_vs_own_baseline": reduced_metrics[cell][
                "bootstrap_vs_baseline"
            ],
        }
    return {
        "full_selected_cell": full_selected,
        "reduced_selected_cell": reduced_selected,
        "selection_reversed": full_selected != reduced_selected,
        "full_ranked_cell_ids": full["ranked_cell_ids"],
        "reduced_ranked_cell_ids": reduced["ranked_cell_ids"],
        "common_cell_error_spearman": {
            "statistic": float(correlation.statistic),
            "pvalue": float(correlation.pvalue),
            "note": "candidate bytes/profile vectors differ; diagnostic only",
        },
        "selection_panels": {
            "full": sorted(full_panel),
            "reduced": sorted(reduced_panel),
            "intersection": sorted(full_panel & reduced_panel),
            "intersection_count": len(full_panel & reduced_panel),
            "jaccard": len(full_panel & reduced_panel)
            / len(full_panel | reduced_panel),
        },
        "selected_cell_diagnostics": cells,
        "causal_corpus_size_comparison": False,
        "why_not_causal": (
            "the searches rebuilt H13, residual profiles, signs, permutations, "
            "expert panels, and candidate bytes"
        ),
        "required_followup": (
            "cross-score the reduced draw-00 and draw-03 candidate bytes on "
            "full-capture documents excluded from the reduced 217-document plan"
        ),
    }


def _render(result: dict[str, Any]) -> str:
    matrix = result["matrix"]
    relative = matrix["relative"]
    profiles = result["profiles"]
    cells = profiles["selected_cell_diagnostics"]
    lines = [
        "# Layer-77 reduced/full calibration representativeness diagnostic",
        "",
        "This is a geometry and selection-stability diagnostic, not a causal corpus-size result.",
        "",
        "## H13 geometry",
        "",
        f"- Full fit rows: **{result['full']['fit_rows']:,}**",
        f"- Reduced fit rows: **{result['reduced']['fit_rows']:,}**",
        f"- Diagonal-mean change: **`{relative['diagonal_mean_change_percent']:+.4f}%`**",
        f"- Diagonal correlation: **`{relative['diagonal_pearson_correlation']:.8f}`**",
        f"- Relative Frobenius difference: **`{relative['frobenius_difference_over_full']:.6f}`**",
        f"- Frobenius cosine: **`{relative['frobenius_cosine']:.8f}`**",
        "",
        "## Profile stability",
        "",
        f"- Full selected: **`{profiles['full_selected_cell']}`**",
        f"- Reduced selected: **`{profiles['reduced_selected_cell']}`**",
        f"- Selection reversed: **{profiles['selection_reversed']}**",
        f"- Common-cell rank correlation: **`{profiles['common_cell_error_spearman']['statistic']:.6f}`**",
        f"- Expert-panel overlap: **{profiles['selection_panels']['intersection_count']} / 16**",
        "",
    ]
    for cell, value in cells.items():
        lines.extend(
            [
                f"### `{cell}`",
                "",
                f"- Full error/rank: `{value['full_mean_document_relative_error']:.9e}` / {value['full_rank']}",
                f"- Reduced error/rank: `{value['reduced_mean_document_relative_error']:.9e}` / {value['reduced_rank']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            "A reversed selection cannot be dismissed as harmless, but it also cannot be attributed only to corpus size because the candidate constructions differ. The required next gate is an external cross-score of the already encoded reduced draw-00 and draw-03 candidates on full-capture documents excluded from the reduced plan. Middle/early encoding should not reuse the reduced selector until that gate is resolved.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-h13", type=Path, default=FULL_ROOT / "preparation/h13.safetensors"
    )
    parser.add_argument(
        "--full-h13-json", type=Path, default=FULL_ROOT / "preparation/h13.json"
    )
    parser.add_argument(
        "--full-selection", type=Path, default=FULL_ROOT / "profile_search/selection.json"
    )
    parser.add_argument(
        "--reduced-h13",
        type=Path,
        default=REDUCED_PREP_ROOT / "preparation/h13.safetensors",
    )
    parser.add_argument(
        "--reduced-h13-json",
        type=Path,
        default=REDUCED_PREP_ROOT / "preparation/h13.json",
    )
    parser.add_argument(
        "--reduced-selection",
        type=Path,
        default=REDUCED_FINAL_ROOT / "profile_search/selection.json",
    )
    parser.add_argument("--probes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--threads", type=int, default=40)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    full_h13_json = json.loads(args.full_h13_json.read_text())
    reduced_h13_json = json.loads(args.reduced_h13_json.read_text())
    full_selection = json.loads(args.full_selection.read_text())
    reduced_selection = json.loads(args.reduced_selection.read_text())
    full_h13 = _load_h13(args.full_h13)
    reduced_h13 = _load_h13(args.reduced_h13)

    result = {
        "schema": "glm52-capture-representativeness-layer-v1",
        "layer": int(full_h13_json["binding"]["layer"]),
        "full": {
            "h13_path": str(args.full_h13.resolve()),
            "h13_sha256": _sha256(args.full_h13),
            "h13_matrix_sha256": full_h13_json["evidence"]["matrix_sha256"],
            "fit_rows": int(full_h13_json["evidence"]["fit_rows"]),
            "selection_documents": int(
                full_selection["cell_metrics"][full_selection["selected_cell_id"]][
                    "bootstrap_vs_baseline"
                ].get("documents", 892)
            ),
            "capture_fingerprint": full_h13_json["evidence"]["capture"][
                "document_plan_fingerprint"
            ],
        },
        "reduced": {
            "h13_path": str(args.reduced_h13.resolve()),
            "h13_sha256": _sha256(args.reduced_h13),
            "h13_matrix_sha256": reduced_h13_json["evidence"]["matrix_sha256"],
            "fit_rows": int(reduced_h13_json["evidence"]["fit_rows"]),
            "selection_documents": 41,
            "capture_fingerprint": reduced_h13_json["evidence"]["capture"][
                "document_plan_fingerprint"
            ],
        },
        "matrix": matrix_comparison(
            full_h13, reduced_h13, probes=args.probes, seed=args.seed
        ),
        "profiles": profile_comparison(full_selection, reduced_selection),
        "limitations": {
            "matched_candidate_bytes": False,
            "matched_expert_panel": False,
            "external_document_cross_score_complete": False,
            "advancement_allowed": False,
        },
    }
    if result["layer"] != int(reduced_h13_json["binding"]["layer"]):
        raise RuntimeError("full/reduced layer binding differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report_path = args.output.with_suffix(".md")
    report_path.write_text(_render(result))
    print(_render(result))
    print(f"json: {args.output}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
