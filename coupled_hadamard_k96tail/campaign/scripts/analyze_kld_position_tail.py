#!/usr/bin/env python3
"""Archive untrimmed KLD tail concentration and position-aligned comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SCHEMA = "glm52-kld-position-tail-concentration-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], list[float]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    values = [float(item) for item in value.get("position_kld", [])]
    if value.get("complete") is not True or not values:
        raise ValueError(f"KLD result is incomplete: {path}")
    # The sealed float32 reduction can produce a few roundoff-scale negative
    # values even though exact KL divergence is nonnegative. Preserve the raw
    # acceptance values, but reject anything larger than numerical noise.
    if any(not math.isfinite(item) or item < -1.0e-6 for item in values):
        raise ValueError(f"KLD positions must be finite and nonnegative: {path}")
    if int(value.get("total_positions", -1)) != len(values):
        raise ValueError(f"KLD position count differs: {path}")
    return value, values


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _summary(values: list[float], removals: tuple[int, ...]) -> dict[str, Any]:
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    total = math.fsum(values)
    ladder: list[dict[str, Any]] = []
    for count in removals:
        if not 0 < count < len(values):
            raise ValueError("each removal count must lie within the position range")
        removed = order[:count]
        removed_sum = math.fsum(values[index] for index in removed)
        ladder.append(
            {
                "removed_positions": count,
                "removed_fraction": count / len(values),
                "remaining_mean_kld": (total - removed_sum) / (len(values) - count),
                "removed_share_of_total_kld": removed_sum / total,
                "cutoff_kld": values[removed[-1]],
                "position_indices": removed,
            }
        )
    return {
        "positions": len(values),
        "untrimmed_mean_kld": _mean(values),
        "removal_ladder": ladder,
        "ranked_positions": order[: max(removals)],
    }


def _aligned_comparison(
    baseline: list[float], candidate: list[float], counts: tuple[int, ...]
) -> dict[str, Any]:
    if len(candidate) != len(baseline):
        raise ValueError("position-aligned KLD results have different lengths")
    order = sorted(range(len(baseline)), key=lambda index: (-baseline[index], index))
    rows: list[dict[str, Any]] = []
    for count in counts:
        indices = order[:count]
        index_set = set(indices)
        base_tail = math.fsum(baseline[index] for index in indices)
        candidate_tail = math.fsum(candidate[index] for index in indices)
        candidate_remainder = [
            value for index, value in enumerate(candidate) if index not in index_set
        ]
        rows.append(
            {
                "baseline_top_positions": count,
                "baseline_position_indices": indices,
                "baseline_tail_kld_sum": base_tail,
                "candidate_tail_kld_sum": candidate_tail,
                "candidate_tail_sum_delta": candidate_tail - base_tail,
                "candidate_mean_without_baseline_top_positions": _mean(
                    candidate_remainder
                ),
            }
        )
    return {"baseline_rank_aligned": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--remove", type=int, nargs="+", default=(10, 20, 30, 40))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = tuple(sorted(set(args.remove)))
    baseline_path = args.baseline.resolve()
    baseline_record, baseline = _load(baseline_path)
    records: dict[str, Any] = {
        "baseline": {
            "path": str(baseline_path),
            "sha256": _sha256(baseline_path),
            "model": baseline_record.get("model"),
            **_summary(baseline, counts),
        }
    }
    for raw_path in args.candidate:
        path = raw_path.resolve()
        record, values = _load(path)
        records.setdefault("candidates", []).append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "model": record.get("model"),
                **_summary(values, counts),
                **_aligned_comparison(baseline, values, counts),
            }
        )
    result = {
        "schema": SCHEMA,
        "complete": True,
        "interpretation": (
            "diagnostic_only_no_positions_are_removed_from_the_acceptance_kld"
        ),
        "removal_counts": list(counts),
        **records,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(json.dumps(result["baseline"]["removal_ladder"], indent=2))


if __name__ == "__main__":
    main()
