#!/usr/bin/env python3
"""Encode one disjoint final-treatment expert shard under a frozen SQG profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import MappingProxyType


NUM_EXPERTS = 256


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
    from src.glm52_bf16_source import (
        BF16ExpertSource,
        SourceValidation,
        ValidatedShard,
    )

    preflight_file = Path(preflight_path).resolve()
    preflight = load_json_object(preflight_file)
    recovery_path = preflight_file.parent / "recovery" / LOCAL_CODE_RECOVERY_NAME
    if recovery_path.is_file():
        preflight["_local_code_recovery"] = load_json_object(recovery_path)

    paths = PipelinePaths.resolve(**preflight["paths"])
    settings = PipelineSettings(**preflight["settings"])
    layer = validate_layer(layer)
    expected_visible_gpu = str(SELECTED_LAYERS.index(layer))
    if (
        str(torch.device(device)) != "cuda:0"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible_gpu
        or torch.cuda.device_count() != 1
        or torch.cuda.current_device() != 0
    ):
        raise ValueError(
            "shard must expose only the layer's sealed physical GPU "
            f"{expected_visible_gpu} as cuda:0"
        )

    source_seal = load_json_object(paths.source_seal)
    index_path = Path(source_seal["index"]["path"]).resolve()
    config_path = Path(source_seal["config"]["path"]).resolve()
    shard_root = Path(source_seal["shard_root"]).resolve()
    with index_path.open("rb") as handle:
        index = json.load(handle)
    weight_map = MappingProxyType(dict(index["weight_map"]))
    expected_identities = preflight.get("source_seal", {}).get(
        "shard_file_identity"
    )
    if not isinstance(expected_identities, dict) or set(expected_identities) != set(
        source_seal["shards"]
    ):
        raise ValueError("sealed BF16 shard identity domain differs")
    shards = {}
    for name, record in source_seal["shards"].items():
        if Path(name).name != name:
            raise ValueError(f"unsafe BF16 shard name in source seal: {name!r}")
        shard_path = shard_root / name
        shards[name] = ValidatedShard(
            bytes=int(record["bytes"]),
            sha256=str(record["sha256"]),
            header_sha256=str(record["header_sha256"]),
            selected_tensor_count=int(record["selected_tensor_count"]),
            total_tensor_count=int(record["total_tensor_count"]),
            file_identity=_validated_sealed_shard_identity(
                shard_path,
                expected_identities[name],
            ),
        )
    validation = SourceValidation(
        index_bytes=int(source_seal["index"]["bytes"]),
        index_sha256=str(source_seal["index"]["sha256"]),
        config_bytes=int(source_seal["config"]["bytes"]),
        config_sha256=str(source_seal["config"]["sha256"]),
        weight_map=weight_map,
        tensor_counts=MappingProxyType(dict(source_seal["tensor_counts"])),
        shards=MappingProxyType(shards),
    )
    source = BF16ExpertSource.__new__(BF16ExpertSource)
    source.index_path = index_path
    source.shard_root = shard_root
    source.config_path = config_path
    source.validation = validation
    source.weight_map = weight_map

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
