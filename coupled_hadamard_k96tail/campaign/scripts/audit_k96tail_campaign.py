#!/usr/bin/env python3
"""Audit the static contract and locally available seals for the K96 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_LAYER_ROOT = Path(
    "/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-layers"
)
DEFAULT_ACCEPTANCE_ROOT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/"
    "glm52_sqg_w4a8_sm120_local_acceptance_20260812"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(64 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not a map: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--layer-root", type=Path, default=DEFAULT_LAYER_ROOT)
    parser.add_argument("--acceptance-root", type=Path, default=DEFAULT_ACCEPTANCE_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--strict-complete",
        action="store_true",
        help="also require all 75 target oracles and every final-result field",
    )
    args = parser.parse_args()

    project = args.project_root.resolve()
    layer_root = args.layer_root.resolve()
    acceptance = args.acceptance_root.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else project / "reproduction" / "k96tail-distributed-campaign.json"
    )
    errors: list[str] = []

    try:
        manifest = load(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"pass": False, "errors": [str(exc)]}, indent=2))
        return 1

    if manifest.get("schema") != "glm52-coupled-k96tail-distributed-campaign-v1":
        errors.append("campaign manifest schema differs")
    rate = manifest.get("rate_policy", {})
    if rate.get("layer_3") != {
        "policy": "preserve_sealed_k48_exception",
        "k3": 720,
        "k4": 48,
        "total": 768,
        "bits_per_weight": 3.0625,
    }:
        errors.append("layer-3 K48 contract differs")
    if rate.get("layers_4_through_77") != {
        "policy": "layer_native_k96",
        "k3": 672,
        "k4": 96,
        "total": 768,
        "bits_per_weight": 3.125,
    }:
        errors.append("layers 4..77 K96 contract differs")
    if rate.get("layer_78") != {
        "policy": "preserve_source_mtp_unchanged",
        "k3": 384,
        "k4": 384,
        "total": 768,
        "bits_per_weight": 3.5,
    }:
        errors.append("preserved MTP78 contract differs")

    remote_layers: list[int] = []
    local_layers: list[int] = []
    for assignment in manifest.get("assignments", []):
        try:
            start, end = assignment["layers"]
            layers = list(range(int(start), int(end) + 1))
            if assignment["worker"] == "local":
                local_layers.extend(layers)
            else:
                remote_layers.extend(layers)
        except (KeyError, TypeError, ValueError):
            errors.append("assignment record is malformed")
    if local_layers != list(range(47, 51)):
        errors.append("local assignment is not exactly layers 47..50")
    if sorted(remote_layers) != list(range(51, 78)) or len(remote_layers) != 27:
        errors.append("remote assignments do not cover layers 51..77 exactly once")

    required = (
        project / "docs" / "K96_COUPLED_DISTRIBUTED_REPRODUCTION_20260814.md",
        project / "hub" / "k96tail-staging" / "README.md",
        project / "scripts" / "run_full_coupled_3p0625_campaign.sh",
        project / "scripts" / "run_vast_wave_and_upload.sh",
        project / "scripts" / "archive_vast_k96_evidence.sh",
        project / "scripts" / "wait_merge_finalize_k96tail.sh",
        project / "scripts" / "run_finalize_k96tail_model.sh",
        project / "scripts" / "seal_coupled_k96tail_release.py",
        project / "vast_supervisor" / "node1-k96.conf",
        project / "vast_supervisor" / "node2-k96.conf",
        project / "vast_supervisor" / "node3-k96.conf",
        project / "vast_supervisor" / "node4-k96.conf",
        project / "vast_supervisor" / "k96-runahead-guard.conf",
        project / "vast_supervisor" / "k96-runahead-guard.sh",
    )
    for path in required:
        if not path.is_file() or path.is_symlink():
            errors.append(f"required reproduction file absent or unsafe: {path}")

    frozen = {
        acceptance
        / "RESULTS/coupled-tail-source-control-tp4dcp1-routes-v2/kld/"
        "kld_sm120_tp4dcp1.json": manifest.get("pins", {}).get("source_kld_sha256"),
        acceptance
        / "RESULTS/coupled-tail-source-control-tp4dcp1-routes-v2/kld/"
        "routed_experts_tp4dcp1.npz": manifest.get("pins", {}).get(
            "source_routes_sha256"
        ),
    }
    frozen_hashes: dict[str, str | None] = {}
    for path, expected in frozen.items():
        if not path.is_file() or path.is_symlink():
            errors.append(f"frozen allocation input absent or unsafe: {path}")
            frozen_hashes[str(path)] = None
            continue
        observed = sha256(path)
        frozen_hashes[str(path)] = observed
        if observed != expected:
            errors.append(f"frozen allocation input hash differs: {path}")

    oracle_layers: list[int] = []
    oracle_failures: list[int] = []
    if layer_root.is_dir() and not layer_root.is_symlink():
        for layer in range(3, 78):
            path = layer_root / f"runtime-oracle-layer-{layer:03d}.json"
            if not path.is_file() or path.is_symlink():
                continue
            try:
                oracle = load(path)
            except (OSError, ValueError, json.JSONDecodeError):
                oracle_failures.append(layer)
                continue
            census = (
                {"k3": 720, "k4": 48, "total": 768}
                if layer == 3
                else {"k3": 672, "k4": 96, "total": 768}
            )
            bpw = 3.0625 if layer == 3 else 3.125
            if not (
                oracle.get("schema")
                == "glm52-coupled-selected-layer-b12x-oracle-v2"
                and oracle.get("complete") is True
                and oracle.get("pass") is True
                and oracle.get("finite") is True
                and oracle.get("nonzero") is True
                and oracle.get("layer") == layer
                and oracle.get("bit_census") == census
                and oracle.get("bits_per_weight") == bpw
                and oracle.get("production_endpoint")
                == "route_packed_direct_e4m3_w4a8"
            ):
                oracle_failures.append(layer)
            else:
                oracle_layers.append(layer)
    if oracle_failures:
        errors.append(f"invalid runtime oracles: {oracle_failures}")

    final_results = manifest.get("final_results", {})
    pending_final = sorted(key for key, value in final_results.items() if value is None)
    if args.strict_complete:
        if oracle_layers != list(range(3, 78)):
            errors.append("strict mode requires passing runtime oracles for layers 3..77")
        if manifest.get("complete") is not True:
            errors.append("strict mode requires complete=true")
        if pending_final:
            errors.append(f"strict mode has pending final fields: {pending_final}")

    result = {
        "schema": "glm52-coupled-k96tail-campaign-audit-v1",
        "pass": not errors,
        "strict_complete": args.strict_complete,
        "manifest": str(manifest_path),
        "manifest_complete": manifest.get("complete"),
        "passing_runtime_oracle_count": len(oracle_layers),
        "passing_runtime_oracle_layers": oracle_layers,
        "pending_final_result_fields": pending_final,
        "frozen_input_hashes": frozen_hashes,
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
