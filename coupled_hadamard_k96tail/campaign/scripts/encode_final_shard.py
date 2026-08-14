#!/usr/bin/env python3
"""Encode one disjoint final-treatment expert shard under a frozen SQG profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


NUM_EXPERTS = 256
REMOTE_PHYSICAL_GPUS = frozenset(str(index) for index in range(8))


def _resolved_visible_gpu_override(explicit: str | None) -> str | None:
    """Resolve the paid-node placement override without changing sealed inputs."""

    value = (
        explicit
        if explicit is not None
        else os.environ.get("FRESH_SQG_EXPECTED_VISIBLE_GPU_OVERRIDE")
    )
    if value is None:
        return None
    value = str(value)
    if value not in REMOTE_PHYSICAL_GPUS:
        raise ValueError("physical GPU override must be one of 0,1,2,3,4,5,6,7")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode a disjoint final-treatment SQG expert shard"
    )
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument(
        "--full-revalidate",
        action="store_true",
        help="rehash sealed BF16/capture inputs before encoding (slow)",
    )
    return parser


def _validated_sealed_shard_identity(
    path: Path,
    expected: object,
) -> tuple[int, int, int, int, int]:
    """Bind the fast path to the preflight inode without hashing its payload."""

    if not isinstance(expected, dict):
        raise ValueError(f"sealed BF16 shard identity is malformed: {path.name}")
    stat = path.stat()
    observed = {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if observed != expected:
        raise ValueError(
            f"BF16 shard file identity drift without payload hash: {path.name}"
        )
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _open_fast_sealed_runtime(
    preflight_path: str,
    *,
    layer: int,
    device: str,
    expected_visible_gpu_override: str | None = None,
):
    """Open a previously sealed runtime without rehashing 145 GB per shard."""

    import torch

    from src.fresh_pipeline_calibration import LayerCaptureView
    from src.fresh_pipeline_common import (
        SELECTED_LAYERS,
        load_bit_contract,
        load_json_object,
        validate_layer,
    )
    from src.fresh_pipeline_runner import (
        LOCAL_CODE_RECOVERY_NAME,
        LayerRuntime,
        PipelinePaths,
        PipelineSettings,
    )
    from src.glm52_bf16_source import open_sealed_bf16_source

    preflight_file = Path(preflight_path).resolve()
    preflight = load_json_object(preflight_file)
    recovery_path = preflight_file.parent / "recovery" / LOCAL_CODE_RECOVERY_NAME
    if recovery_path.is_file():
        preflight["_local_code_recovery"] = load_json_object(recovery_path)

    paths = PipelinePaths.resolve(**preflight["paths"])
    settings = PipelineSettings(**preflight["settings"])
    layer = validate_layer(layer)
    sealed_visible_gpu = str(SELECTED_LAYERS.index(layer))
    placement_override = _resolved_visible_gpu_override(expected_visible_gpu_override)
    expected_visible_gpu = (
        sealed_visible_gpu if placement_override is None else placement_override
    )
    if (
        str(torch.device(device)) != "cuda:0"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible_gpu
        or torch.cuda.device_count() != 1
        or torch.cuda.current_device() != 0
    ):
        raise ValueError(
            "shard must expose only its declared physical GPU "
            f"{expected_visible_gpu} as cuda:0; layer seal is {sealed_visible_gpu}"
        )

    source_seal = load_json_object(paths.source_seal)
    expected_identities = preflight.get("source_seal", {}).get(
        "shard_file_identity"
    )
    if not isinstance(expected_identities, dict) or set(expected_identities) != set(
        source_seal["shards"]
    ):
        raise ValueError("sealed BF16 shard identity domain differs")
    source = open_sealed_bf16_source(
        source_seal=source_seal,
        shard_file_identity=expected_identities,
        cache_namespace=os.environ.get("B300_WEIGHT_CACHE_NAMESPACE"),
    )

    return LayerRuntime(
        paths=paths,
        settings=settings,
        layer=layer,
        device=device,
        source=source,
        capture=LayerCaptureView(
            paths.capture_dir,
            layer,
            validate=False,
            verify_hashes=False,
        ),
        bit_map=load_bit_contract(paths.bit_contract)[layer].bit_map,
        source_seal=source_seal,
        preflight=preflight,
    )


def main() -> int:
    args = _parser().parse_args()
    if not 0 <= args.start < args.end <= NUM_EXPERTS:
        raise ValueError("expert shard must satisfy 0 <= start < end <= 256")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    thread_count = str(args.threads)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = thread_count

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    from src.fresh_pipeline_common import sha256_file
    from src.fresh_pipeline_runner import (
        _encode_expert_sequence,
        _load_bound_kquant_runtime,
        _load_h13,
        _load_profile_selection,
        open_layer_runtime,
    )

    started = time.monotonic()
    runtime = (
        open_layer_runtime(
            args.preflight,
            layer=args.layer,
            device=args.device,
        )
        if args.full_revalidate
        else _open_fast_sealed_runtime(
            args.preflight,
            layer=args.layer,
            device=args.device,
        )
    )
    selection_path = runtime.profile_dir / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"frozen profile selection is required before shard encoding: {selection_path}"
        )
    selection, gate_profile, down_profile = _load_profile_selection(runtime)
    if (
        selection.get("selection_frozen_before_holdout") is not True
        or selection.get("selection_used_once_for_choice") is not True
    ):
        raise ValueError("profile selection is not recorded as frozen")

    selection_sha256 = sha256_file(selection_path)
    h13, _ = _load_h13(runtime)
    kquant_runtime = _load_bound_kquant_runtime(runtime)
    output_dir = (
        runtime.layer_root
        / "expert_shards"
        / f"experts_{args.start:03d}_{args.end:03d}"
    )
    sequence = _encode_expert_sequence(
        runtime,
        kquant_runtime,
        h13,
        order=tuple(range(args.start, args.end)),
        output_dir=output_dir,
        purpose="final_treatment",
        selection_evidence_sha256=selection_sha256,
        gate_profile=gate_profile,
        down_profile=down_profile,
        cell_evidence={
            "selection_id": selection["selection_id"],
            "selection_sha256": selection_sha256,
            "selected_cell_id": selection["selected_cell_id"],
            "selected_profile_shard_sha256": selection[
                "selected_profile_shard"
            ]["shard_sha256"],
            "profile_frozen": True,
            "holdout_unseen": True,
            "parallel_shard": {
                "start": args.start,
                "end": args.end,
            },
        },
    )
    print(
        json.dumps(
            {
                "complete": True,
                "layer": args.layer,
                "start": args.start,
                "end": args.end,
                "expert_count": args.end - args.start,
                "device": args.device,
                "threads": args.threads,
                "full_revalidate": args.full_revalidate,
                "sealed_fast_resume": not args.full_revalidate,
                "output_dir": str(output_dir),
                "selection_id": selection["selection_id"],
                "selection_sha256": selection_sha256,
                "elapsed_seconds": time.monotonic() - started,
                "sequence": sequence,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
