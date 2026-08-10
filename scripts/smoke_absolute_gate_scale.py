#!/usr/bin/env python3
"""Encode one real mixed-K3/K4 expert before corrected profile search."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SMOKE_LAYER = 28
SMOKE_EXPERT = 0
SMOKE_DRAW = 0
SMOKE_FAMILY = "identity"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(args.threads)

    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    from scripts.encode_final_shard import _open_fast_sealed_runtime
    from src.fresh_pipeline_artifacts import expert_stem, validate_expert_artifact
    from src.fresh_pipeline_common import (
        atomic_json,
        canonical_sha256,
        load_json_object,
        sha256_file,
        tensor_prefix,
    )
    from src.fresh_pipeline_runner import (
        _encode_expert_sequence,
        _load_bound_kquant_runtime,
        _load_h13,
        _load_scale_evidence,
        _profiles_for_cell,
    )
    from src.glm52_fresh_sqg.codec import (
        realize_shared_residual_profile,
        validate_gate_up_encode_smoke,
    )

    runtime = _open_fast_sealed_runtime(
        args.preflight, layer=SMOKE_LAYER, device=args.device
    )
    receipt_path = runtime.paths.output_root / "successor_preflight_receipt.json"
    receipt = load_json_object(receipt_path)
    if (
        receipt.get("complete") is not True
        or receipt.get("successor_preflight_id")
        != runtime.preflight["preflight_id"]
    ):
        raise ValueError("absolute-scale smoke lacks successor import receipt")

    h13, _ = _load_h13(runtime)
    scales = _load_scale_evidence(runtime)
    gate_profile, down_profile = _profiles_for_cell(
        runtime, scales, draw=SMOKE_DRAW, family=SMOKE_FAMILY
    )
    gate_profile = realize_shared_residual_profile(gate_profile, runtime.device)
    down_profile = realize_shared_residual_profile(down_profile, runtime.device)
    raw_gate_mean = float(scales.gate_input_base.float().mean())
    profile_gate_mean = float(gate_profile.channel_scales.float().mean())
    if not torch.isclose(
        torch.tensor(profile_gate_mean),
        torch.tensor(raw_gate_mean),
        rtol=2e-6,
        atol=1e-8,
    ):
        raise RuntimeError("identity smoke profile lost the absolute gate RMS")
    if abs(profile_gate_mean - 1.0) < 0.1:
        raise RuntimeError("identity smoke profile unexpectedly resembles mean-one scaling")

    output_dir = runtime.layer_root / "absolute_scale_smoke" / "experts"
    sequence = _encode_expert_sequence(
        runtime,
        _load_bound_kquant_runtime(runtime),
        h13,
        order=(SMOKE_EXPERT,),
        output_dir=output_dir,
        purpose="absolute_gate_scale_smoke",
        selection_evidence_sha256=sha256_file(receipt_path),
        gate_profile=gate_profile,
        down_profile=down_profile,
        cell_evidence={
            "smoke": True,
            "draw": SMOKE_DRAW,
            "family": SMOKE_FAMILY,
            "profile_choice_not_yet_made": True,
            "not_eligible_for_profile_scoring": True,
            "not_eligible_for_final_materialization": True,
        },
    )
    manifest_path = output_dir / f"{expert_stem(SMOKE_LAYER, SMOKE_EXPERT)}.json"
    artifact = validate_expert_artifact(manifest_path)
    records: dict[str, object] = {}
    expected_bits = {"gate_proj": 4, "up_proj": 3}
    for projection, expected_bit in expected_bits.items():
        tensor_id = tensor_prefix(SMOKE_LAYER, SMOKE_EXPERT, projection)
        tensor = artifact["tensor_manifests"][tensor_id]
        if tensor["bits"] != expected_bit:
            raise RuntimeError(f"smoke {projection} is not expected K{expected_bit}")
        global_scale = float(tensor["transform"]["global_scale"])
        source_rmse = float(tensor["decoded_closure"]["source_relative_rmse"])
        validate_gate_up_encode_smoke(
            global_scale=global_scale,
            source_relative_rmse=source_rmse,
        )
        records[projection] = {
            "bits": expected_bit,
            "global_scale": global_scale,
            "source_relative_rmse": source_rmse,
            "decoded_exl_sha256": tensor["decoded_closure"]["decoded_exl_sha256"],
            "independent_stored_fp16_decode_passed": tensor["decoded_closure"]["passed"],
        }
    down_id = tensor_prefix(SMOKE_LAYER, SMOKE_EXPERT, "down_proj")
    down = artifact["tensor_manifests"][down_id]
    if artifact["h2"]["upstream_candidate"]["gate_decoded_exl_sha256"] != records[
        "gate_proj"
    ]["decoded_exl_sha256"] or artifact["h2"]["upstream_candidate"][
        "up_decoded_exl_sha256"
    ] != records["up_proj"]["decoded_exl_sha256"]:
        raise RuntimeError("smoke H2 is not conditional on the decoded gate/up candidate")

    smoke: dict[str, object] = {
        "schema": "glm52-fresh-sqg-absolute-gate-scale-real-smoke-v1",
        "complete": True,
        "preflight_id": runtime.preflight["preflight_id"],
        "successor_receipt_sha256": sha256_file(receipt_path),
        "layer": SMOKE_LAYER,
        "expert": SMOKE_EXPERT,
        "mixed_gate_up_bits": [4, 3],
        "profile": gate_profile.manifest(),
        "raw_gate_input_base_mean": raw_gate_mean,
        "smoke_gate_profile_mean": profile_gate_mean,
        "records": records,
        "candidate_conditional_h2": {
            "evidence_id": artifact["h2"]["evidence_id"],
            "matrix_sha256": artifact["h2"]["matrix_sha256"],
            "expert_local": artifact["h2"]["pooled_expert_basis"] is False,
        },
        "down": {
            "bits": down["bits"],
            "decoded_exl_sha256": down["decoded_closure"]["decoded_exl_sha256"],
            "encoded_after_candidate_h2": True,
        },
        "expert_manifest": str(manifest_path),
        "expert_manifest_sha256": sha256_file(manifest_path),
        "sequence": sequence,
        "eligible_for_profile_selection": False,
        "eligible_for_final_materialization": False,
    }
    smoke["smoke_id"] = canonical_sha256(smoke)
    destination = runtime.paths.output_root / "absolute_gate_scale_smoke.json"
    if destination.exists():
        observed = load_json_object(destination)
        if observed != smoke:
            raise ValueError("existing absolute-scale smoke receipt differs")
    else:
        atomic_json(destination, smoke)
    print(json.dumps(smoke, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
