#!/usr/bin/env python3
"""Materialize the directional-test candidate without rehashing 343 GB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _fast_run_context(candidate, run_seal: Path, artifacts_root: Path):
    """Trust the just-created seal and defer payload checking to copied shards."""

    from src.fresh_pipeline_common import SELECTED_LAYERS, canonical_sha256, sha256_file

    root = artifacts_root.resolve()
    seal_path = run_seal.resolve()
    seal = candidate._load_json(seal_path)
    seal_id = seal.get("run_seal_id")
    if seal_id != canonical_sha256(
        {key: value for key, value in seal.items() if key != "run_seal_id"}
    ):
        raise ValueError("directional fast materializer run-seal ID differs")
    layers = {}
    for layer in SELECTED_LAYERS:
        record = seal["layers"][str(layer)]
        layer_root = root / f"layer_{layer:03d}"
        final_root = layer_root / "final"
        profile_root = layer_root / "profile_search"
        manifest_path = final_root / f"fresh-sqg-layer-{layer:03d}.json"
        manifest_seal_path = manifest_path.with_suffix(".json.sha256")
        selection_path = profile_root / "selection.json"
        manifest = candidate._load_json(manifest_path)
        manifest_sha256 = sha256_file(manifest_path)
        selection_sha256 = sha256_file(selection_path)
        shard_path = final_root / str(manifest["shard"])
        if (
            record.get("layer_artifact_sha256") != manifest_sha256
            or record.get("layer_shard_sha256") != manifest.get("shard_sha256")
            or record.get("selection_sha256") != selection_sha256
            or not shard_path.is_file()
            or shard_path.is_symlink()
        ):
            raise ValueError(
                f"directional fast materializer layer {layer} seal binding differs"
            )
        layers[layer] = {
            "record": record,
            "manifest": manifest,
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_sha256,
            "manifest_seal_path": manifest_seal_path,
            "manifest_seal_sha256": sha256_file(manifest_seal_path),
            "shard_path": shard_path,
            "selection_path": selection_path,
            "selection_sha256": selection_sha256,
            "holdout_path": None,
            "holdout_sha256": None,
        }
    return {
        "seal": seal,
        "path": seal_path,
        "sha256": sha256_file(seal_path),
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--teacher-receipt", type=Path, required=True)
    parser.add_argument("--run-seal", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--bit-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from src.calibration_capture import (
        TEACHER_IDENTITY_SEAL_SHA256,
        TEACHER_IDENTITY_VALIDATION,
    )
    import src.fresh_candidate_materializer as candidate
    from src.teacher_identity import validate_teacher_identity_receipt

    source = args.source_model.resolve()
    receipt = args.teacher_receipt.resolve()
    metadata_validation = validate_teacher_identity_receipt(
        source,
        receipt,
        expected_seal_sha256=TEACHER_IDENTITY_SEAL_SHA256,
        verify_mode="metadata",
        workers=4,
    )
    source_context = candidate._source_inventory(
        source,
        receipt,
        verify_teacher=False,
        teacher_validation=dict(TEACHER_IDENTITY_VALIDATION),
    )
    run_context = _fast_run_context(
        candidate,
        args.run_seal,
        args.artifacts_root,
    )
    original_source_inventory = candidate._source_inventory
    original_validate_run_seal = candidate.validate_run_seal

    def sealed_fast_inventory(
        requested_source: Path,
        requested_receipt: Path,
        *,
        verify_teacher: bool,
        teacher_validation=None,
    ):
        if requested_source.resolve() != source or requested_receipt.resolve() != receipt:
            raise ValueError("directional fast materializer source binding differs")
        return source_context

    candidate._source_inventory = sealed_fast_inventory
    candidate.validate_run_seal = lambda requested_seal, requested_root: run_context
    try:
        result = candidate.materialize_candidate(
            source_model=source,
            teacher_receipt=receipt,
            run_seal=args.run_seal,
            artifacts_root=args.artifacts_root,
            bit_contract=args.bit_contract,
            output=args.output,
        )
    finally:
        candidate._source_inventory = original_source_inventory
        candidate.validate_run_seal = original_validate_run_seal

    print(
        json.dumps(
            {
                "directional_test_fast": True,
                "full_teacher_rehash_performed": False,
                "pre_copy_layer_payload_rehash_performed": False,
                "copied_sqg_payload_validation_performed": True,
                "current_teacher_check": metadata_validation,
                "candidate": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
