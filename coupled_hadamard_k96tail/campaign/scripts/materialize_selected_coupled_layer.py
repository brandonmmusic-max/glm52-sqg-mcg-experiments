#!/usr/bin/env python3
"""Materialize one selected coupled layer in the existing GLM runtime layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NUM_EXPERTS = 256
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ARTIFACT_SCHEMA = "glm52-coupled-mixed-k3-k4-expert-v4"
SELECTION_SCHEMA = "glm52-coupled-mixed-rate-selection-v4"
OUTPUT_SCHEMA = "glm52-coupled-selected-layer-runtime-v3"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not a map: {path}")
    return value


def _entry(reader: Any, name: str, TensorEntry: Any) -> Any:
    info = reader.tensors[name]
    return TensorEntry(info.name, info.dtype, info.shape, info.payload)


def main() -> int:
    args = _parser().parse_args()
    from bmmlaw_r7_encoder.safetensors_io import (
        SafeTensorReader,
        TensorEntry,
        write_safetensors_atomic,
    )
    from src.fresh_pipeline_common import atomic_json, canonical_sha256, sha256_file

    root = args.candidate_root.resolve() / f"layer_{args.layer:03d}"
    selection_path = root / "coupled_selection_manifest.json"
    selection = _json(selection_path)
    draws_raw = selection.get("selected_draws")
    selection_allocation_binding = selection.get("allocation_binding")
    profile_binding = selection.get("final_profile_binding")
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("complete") is not True
        or selection.get("layer") != args.layer
        or selection.get("holdout_used") is not False
        or selection.get("rate_contract") != "independent_per_tensor_k3_k4"
        or not isinstance(draws_raw, dict)
        or not isinstance(selection_allocation_binding, dict)
        or not isinstance(selection_allocation_binding.get("allocation_id"), str)
        or not isinstance(selection_allocation_binding.get("sha256"), str)
        or selection_allocation_binding.get("histogram")
        not in ({"3": 720, "4": 48}, {"3": 696, "4": 72}, {"3": 672, "4": 96})
        or not isinstance(profile_binding, dict)
        or profile_binding.get("schema")
        != "glm52-updated-qsrt-coupled-final-profile-binding-v1"
        or profile_binding.get("no_b300_owner_speed_rescue") is not True
        or set(draws_raw) != {str(expert) for expert in range(NUM_EXPERTS)}
        or any(type(draw) is not int or draw not in (0, 6) for draw in draws_raw.values())
    ):
        raise ValueError(f"selection contract differs: {selection_path}")

    entries: list[Any] = []
    artifacts: dict[str, str] = {}
    bit_map: dict[str, int] = {}
    gate_up_suh_hash: str | None = None
    down_svh_hash: str | None = None
    shared_gate_up_entry: Any | None = None
    shared_down_entry: Any | None = None
    allocation_binding: dict[str, Any] | None = None
    for expert in range(NUM_EXPERTS):
        draw = int(draws_raw[str(expert)])
        manifest_path = (
            root
            / f"draw_{draw:02d}"
            / "experts"
            / f"layer-{args.layer:03d}-expert-{expert:03d}.json"
        )
        manifest = _json(manifest_path)
        if (
            manifest.get("schema") != ARTIFACT_SCHEMA
            or manifest.get("complete") is not True
            or manifest.get("layer") != args.layer
            or manifest.get("expert") != expert
            or manifest.get("intermediate_draw") != draw
            or manifest.get("activation") != "silu_gate_times_up"
            or manifest.get("final_profile_binding") != profile_binding
            or manifest.get("coupled_transform")
            != {
                "activation": "silu",
                "intermediate_draw": draw,
                "postactivation_block_size": 128,
                "preactivation_block_size": 128,
                "residual_block_size": 512,
                "residual_draw": 0,
                "residual_policy": "coupled_block_hadamard",
            }
        ):
            raise ValueError(f"selected artifact binding differs: {manifest_path}")
        current_allocation = manifest.get("allocation")
        if not isinstance(current_allocation, dict):
            raise ValueError(f"selected artifact allocation is absent: {manifest_path}")
        if allocation_binding is None:
            allocation_binding = current_allocation
        elif current_allocation != allocation_binding:
            raise ValueError(f"selected artifact allocations differ: {manifest_path}")
        shard = manifest_path.parent / str(manifest.get("shard", ""))
        if sha256_file(shard) != manifest.get("shard_sha256"):
            raise ValueError(f"selected artifact shard hash differs: {shard}")
        reader = SafeTensorReader(shard)
        expected_names: set[str] = set()
        for projection in PROJECTIONS:
            prefix = f"model.layers.{args.layer}.mlp.experts.{expert}.{projection}"
            expected_names.update(f"{prefix}.{suffix}" for suffix in ("sqg", "suh", "svh", "trellis"))
        if set(reader.tensors) != expected_names:
            raise ValueError(f"selected artifact tensor domain differs: {shard}")
        artifacts[str(expert)] = str(manifest["manifest_id"])
        for projection in PROJECTIONS:
            bits = int(manifest["bits"][projection])
            prefix = f"model.layers.{args.layer}.mlp.experts.{expert}.{projection}"
            bit_map[prefix] = bits
            kept = (
                ("sqg", "trellis", "svh")
                if projection in ("gate_proj", "up_proj")
                else ("sqg", "trellis", "suh")
            )
            entries.extend(
                _entry(reader, f"{prefix}.{suffix}", TensorEntry) for suffix in kept
            )

        gate_name = f"model.layers.{args.layer}.mlp.experts.{expert}.gate_proj.suh"
        up_name = f"model.layers.{args.layer}.mlp.experts.{expert}.up_proj.suh"
        down_name = f"model.layers.{args.layer}.mlp.experts.{expert}.down_proj.svh"
        gate_hash = reader.tensors[gate_name].payload.sha256()
        up_hash = reader.tensors[up_name].payload.sha256()
        current_down_hash = reader.tensors[down_name].payload.sha256()
        if gate_hash != up_hash:
            raise ValueError(f"expert {expert} coupled gate/up input bases differ")
        if gate_up_suh_hash is None:
            gate_up_suh_hash = gate_hash
            down_svh_hash = current_down_hash
            shared_gate_up_entry = TensorEntry(
                f"model.layers.{args.layer}.mlp.experts.r7_shared.gate_up_suh",
                reader.tensors[gate_name].dtype,
                reader.tensors[gate_name].shape,
                reader.tensors[gate_name].payload,
            )
            shared_down_entry = TensorEntry(
                f"model.layers.{args.layer}.mlp.experts.r7_shared.down_svh",
                reader.tensors[down_name].dtype,
                reader.tensors[down_name].shape,
                reader.tensors[down_name].payload,
            )
        elif gate_hash != gate_up_suh_hash or current_down_hash != down_svh_hash:
            raise ValueError(f"expert {expert} shared coupled residual bases differ")

    census = {
        "k3": sum(bits == 3 for bits in bit_map.values()),
        "k4": sum(bits == 4 for bits in bit_map.values()),
        "total": len(bit_map),
    }
    if (
        census["total"] != 768
        or census["k3"] + census["k4"] != 768
        or census["k4"] not in (48, 72, 96)
    ):
        raise ValueError(f"selected coupled rate census differs: {census}")
    bits_per_weight = (3 * census["k3"] + 4 * census["k4"]) / 768.0
    if (
        allocation_binding["histogram"]
        != {"3": census["k3"], "4": census["k4"]}
        or allocation_binding.get("bpw") != bits_per_weight
    ):
        raise ValueError("selected artifacts differ from the sealed allocation")
    if allocation_binding != selection_allocation_binding:
        raise ValueError("selected artifacts and draw selection allocation differ")
    if shared_gate_up_entry is None or shared_down_entry is None:
        raise RuntimeError("shared coupled layer entries were not resolved")
    if allocation_binding is None:
        raise RuntimeError("selected coupled allocation binding was not resolved")
    entries.extend((shared_gate_up_entry, shared_down_entry))
    if len(entries) != 2306 or len({entry.name for entry in entries}) != 2306:
        raise RuntimeError("runtime layer layout did not close at 2,306 unique tensors")

    output = args.output.resolve()
    manifest_path = output.with_suffix(".json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite coupled layer output: {output}")
    payload_hashes, output_hash = write_safetensors_atomic(
        output,
        entries,
        metadata={"format": "pt", "sqg_materialization": OUTPUT_SCHEMA},
    )
    record: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "complete": True,
        "layer": args.layer,
        "shard": output.name,
        "shard_sha256": output_hash,
        "shard_bytes": output.stat().st_size,
        "tensor_count": len(entries),
        "tensor_domain_sha256": canonical_sha256(sorted(payload_hashes)),
        "selection_manifest": str(selection_path),
        "selection_id": selection["selection_id"],
        "allocation_binding": allocation_binding,
        "selected_beta": selection["selected_beta"],
        "final_profile_binding": profile_binding,
        "selected_draws": draws_raw,
        "draw_histogram": selection["draw_histogram"],
        "selected_artifact_ids": artifacts,
        "allocation": allocation_binding,
        "bit_census": census,
        "bits_per_weight": bits_per_weight,
        "coupled_transform": {
            "residual_policy": "coupled_block_hadamard",
            "residual_block_size": 512,
            "residual_draw": 0,
            "preactivation_block_size": 128,
            "postactivation_block_size": 128,
            "activation": "silu",
        },
        "shared_h": {
            "gate_up_suh_sha256": gate_up_suh_hash,
            "down_svh_sha256": down_svh_hash,
        },
    }
    record["manifest_id"] = canonical_sha256(record)
    atomic_json(manifest_path, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
