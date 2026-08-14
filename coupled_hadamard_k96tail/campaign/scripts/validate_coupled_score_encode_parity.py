#!/usr/bin/env python3
"""Prove draw-zero production payloads equal the sealed scorer candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "kquant"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _allocation_binding(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    return {
        "file": path.name,
        "schema": value.get("schema"),
        "allocation_id": value.get("allocation_id"),
        "sha256": _sha256(path),
        "histogram": value.get("histogram"),
        "bpw": value.get("bpw"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from safetensors.torch import load_file

    from scripts.score_sqg_w4a8_triplet_candidates import (
        _projection_payload_sha256,
    )
    from src.fresh_pipeline_common import atomic_json, canonical_sha256

    allocation_path = args.allocation.resolve()
    allocation = _json(allocation_path)
    assignments = allocation.get("expert_assignments")
    if (
        allocation.get("complete") is not True
        or allocation.get("layer") != args.layer
        or not isinstance(assignments, dict)
        or set(assignments) != {str(expert) for expert in range(256)}
    ):
        raise ValueError("allocation expert binding differs")
    binding = _allocation_binding(allocation_path, allocation)
    profile_binding = allocation.get("final_profile_binding")
    if (
        not isinstance(profile_binding, dict)
        or profile_binding.get("schema")
        != "glm52-updated-qsrt-coupled-final-profile-binding-v1"
        or profile_binding.get("no_b300_owner_speed_rescue") is not True
    ):
        raise ValueError("allocation no-shortcut profile binding differs")

    score_root = args.score_root.resolve()
    candidate_root = args.candidate_root.resolve()
    checked: list[dict[str, Any]] = []
    for expert in range(256):
        score_path = (
            score_root
            / f"layer_{args.layer:03d}"
            / "experts"
            / f"expert_{expert:03d}.json"
        )
        score = _json(score_path)
        rates = {
            projection: int(assignments[str(expert)]["rates"][projection])
            for projection in PROJECTIONS
        }
        candidates = [
            item for item in score.get("candidates", []) if item.get("rates") == rates
        ]
        if (
            score.get("schema") != "glm52-coupled-tail-triplet-expert-scores-v5"
            or score.get("complete") is not True
            or score.get("layer") != args.layer
            or score.get("expert") != expert
            or len(candidates) != 1
            or candidates[0].get("candidate_id")
            != assignments[str(expert)].get("candidate_id")
            or score.get("final_profile_binding") != profile_binding
        ):
            raise ValueError(f"expert {expert}: score/allocation binding differs")
        candidate = candidates[0]
        stem = f"layer-{args.layer:03d}-expert-{expert:03d}"
        artifact_root = (
            candidate_root / f"layer_{args.layer:03d}" / "draw_00" / "experts"
        )
        manifest_path = artifact_root / f"{stem}.json"
        shard_path = artifact_root / f"{stem}.safetensors"
        manifest = _json(manifest_path)
        if (
            manifest.get("schema") != "glm52-coupled-mixed-k3-k4-expert-v4"
            or manifest.get("complete") is not True
            or manifest.get("layer") != args.layer
            or manifest.get("expert") != expert
            or manifest.get("intermediate_draw") != 0
            or manifest.get("bits") != rates
            or manifest.get("allocation") != binding
            or manifest.get("final_profile_binding") != profile_binding
            or manifest.get("calibration", {}).get("numerical_chunk_rows")
            != args.chunk_rows
            or manifest.get("shard") != shard_path.name
            or manifest.get("shard_sha256") != _sha256(shard_path)
        ):
            raise ValueError(f"expert {expert}: production artifact binding differs")

        tensors = load_file(str(shard_path), device="cpu")
        payloads: dict[str, str] = {}
        for projection in PROJECTIONS:
            prefix = (
                f"model.layers.{args.layer}.mlp.experts.{expert}.{projection}."
            )
            encoded = SimpleNamespace(
                **{
                    suffix: tensors[f"{prefix}{suffix}"]
                    for suffix in ("trellis", "suh", "svh", "sqg")
                }
            )
            payloads[projection] = _projection_payload_sha256(encoded)
        if payloads != candidate.get("payload_sha256"):
            raise ValueError(f"expert {expert}: scorer/encoder payload bytes differ")
        checked.append(
            {
                "expert": expert,
                "candidate_id": candidate["candidate_id"],
                "payload_sha256": payloads,
                "artifact_manifest_id": manifest.get("manifest_id"),
            }
        )

    result: dict[str, Any] = {
        "schema": "glm52-coupled-scorer-encoder-byte-parity-v1",
        "complete": True,
        "layer": args.layer,
        "draw": 0,
        "numerical_chunk_rows": args.chunk_rows,
        "allocation": binding,
        "selected_beta": allocation.get("selected_beta"),
        "final_profile_binding": profile_binding,
        "score_root": str(score_root),
        "candidate_root": str(candidate_root),
        "experts_checked": len(checked),
        "projection_payloads_checked": len(checked) * len(PROJECTIONS),
        "all_exact": True,
        "experts": checked,
    }
    result["receipt_id"] = canonical_sha256(result)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps({key: result[key] for key in (
        "complete", "layer", "experts_checked", "projection_payloads_checked",
        "all_exact", "receipt_id",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
