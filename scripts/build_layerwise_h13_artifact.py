#!/usr/bin/env python3
"""Build a sealed four-layer artifact from a selected per-layer H13 panel.

Large assembled layer shards are hard-linked, not copied or rewritten. The
result is a distinct sealed artifact root suitable for the fast directional
materializer while preserving the exact packed bytes selected by the scorer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fresh_pipeline_common import (  # noqa: E402
    RUN_SEAL_SCHEMA,
    SELECTED_LAYERS,
    canonical_sha256,
    sha256_file,
)


def _candidate(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("candidate must be LABEL=/absolute/root")
    label, root_raw = raw.split("=", 1)
    root = Path(root_raw)
    if not label or not root.is_absolute():
        raise argparse.ArgumentTypeError("candidate label/path is invalid")
    return label, root


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def build(
    *, selection_path: Path, candidates: dict[str, Path], output_root: Path
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    mapping = selection.get(
        "recommended_mapping", selection["winner"]["layer_to_label"]
    )
    if set(mapping) != {str(layer) for layer in SELECTED_LAYERS}:
        raise ValueError("selection layer mapping differs")
    missing = set(mapping.values()) - set(candidates)
    if missing:
        raise ValueError(f"selection labels are absent: {sorted(missing)}")

    seals = {
        label: json.loads((root / "run_seal.json").read_text(encoding="utf-8"))
        for label, root in candidates.items()
    }
    run_ids = {str(seal["run_id"]) for seal in seals.values()}
    preflight_hashes = {str(seal["preflight_sha256"]) for seal in seals.values()}
    if len(run_ids) != 1 or len(preflight_hashes) != 1:
        raise ValueError("candidate run/preflight identities differ")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        (temporary / "logs").mkdir()
        layers: dict[str, Any] = {}
        source_binding: dict[str, Any] = {}
        for layer in SELECTED_LAYERS:
            label = mapping[str(layer)]
            root = candidates[label]
            source_layer = root / f"layer_{layer:03d}"
            source_manifest = (
                source_layer / "final" / f"fresh-sqg-layer-{layer:03d}.json"
            )
            manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
            source_manifest_seal = source_manifest.with_suffix(".json.sha256")
            source_shard = source_manifest.parent / str(manifest["shard"])
            source_selection = source_layer / "profile_search" / "selection.json"

            destination_layer = temporary / f"layer_{layer:03d}"
            destination_manifest = (
                destination_layer
                / "final"
                / f"fresh-sqg-layer-{layer:03d}.json"
            )
            destination_manifest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_manifest, destination_manifest)
            shutil.copy2(
                source_manifest_seal,
                destination_manifest.with_suffix(".json.sha256"),
            )
            _link(source_shard, destination_manifest.parent / source_shard.name)
            destination_selection = (
                destination_layer / "profile_search" / "selection.json"
            )
            destination_selection.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_selection, destination_selection)

            source_record = seals[label]["layers"][str(layer)]
            if (
                sha256_file(destination_manifest)
                != source_record["layer_artifact_sha256"]
                or sha256_file(destination_selection)
                != source_record["selection_sha256"]
                or manifest["shard_sha256"]
                != source_record["layer_shard_sha256"]
            ):
                raise RuntimeError(f"layer {layer} copied binding differs")
            layers[str(layer)] = {
                **source_record,
                "layer_artifact": (
                    f"/output/layer_{layer:03d}/final/"
                    f"fresh-sqg-layer-{layer:03d}.json"
                ),
                "selected_h13_label": label,
            }
            source_binding[str(layer)] = {
                "label": label,
                "source_root": str(root.resolve()),
                "source_run_seal_id": seals[label]["run_seal_id"],
                "source_layer_shard": str(source_shard.resolve()),
                "hard_linked": True,
            }

        first_seal = next(iter(seals.values()))
        seal: dict[str, Any] = {
            "schema": RUN_SEAL_SCHEMA,
            "complete": True,
            "run_id": next(iter(run_ids)),
            "preflight_sha256": next(iter(preflight_hashes)),
            "layers": layers,
            "census": {
                "layers": 4,
                "experts": 1024,
                "sqg_tensors": 3072,
                "mcg_tensors": 0,
                "K3": 1536,
                "K4": 1536,
                "other_rates": 0,
            },
            "lineage": {
                **first_seal["lineage"],
                "expert_local_h13_fixed_alpha_ablation": True,
                "expert_local_h13_oas_global_prior": False,
                "requested_local_alpha": None,
                "requested_local_alpha_by_layer": mapping,
                "fallback_count": 0,
            },
            "scope": {
                "four_layer_replacement_artifacts": True,
                "runnable_model_materialized": False,
                "existing_model_mutated": False,
                "selection_and_holdout_used_by_encoder": False,
                "exploratory_layerwise_blend": True,
            },
            "source_binding": source_binding,
            "selection": {
                "path": str(selection_path.resolve()),
                "sha256": sha256_file(selection_path),
                "analysis_type": selection["analysis_type"],
                "source_role": selection["source_role"],
            },
        }
        seal["run_seal_id"] = canonical_sha256(seal)
        (temporary / "run_seal.json").write_text(
            json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "layerwise_build.json").write_text(
            json.dumps(
                {
                    "schema": "glm52-sqg-layerwise-h13-build-v1",
                    "selection": seal["selection"],
                    "mapping": mapping,
                    "source_binding": source_binding,
                    "run_seal_id": seal["run_seal_id"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return seal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", type=_candidate, required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate):
        raise ValueError("candidate labels must be unique")
    seal = build(
        selection_path=args.selection.resolve(),
        candidates={label: root.resolve() for label, root in candidates.items()},
        output_root=args.output_root.resolve(),
    )
    print(json.dumps({"run_seal_id": seal["run_seal_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
