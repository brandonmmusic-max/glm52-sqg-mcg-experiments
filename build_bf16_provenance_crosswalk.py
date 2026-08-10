#!/usr/bin/env python3
"""Build an external BF16-source crosswalk for the four-layer SQG pilot.

This is evidence only.  It is not imported by capture or encoding.  The MCG
sidecars are streamed, and only their five BF16 source-provenance scalars are
retained; no transform, scale, permutation, seed, trellis, or payload value is
used.  Every selected tensor is then hashed directly from the pinned official
BF16 safetensors files and compared with both the historical sidecar receipt
and the authoritative R10 source inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, BinaryIO

import ijson


LAYERS = (6, 28, 52, 77)
EXPERTS = 256
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPECTED_TENSOR_BYTES = 25_165_824
MCG_BF16_FIELDS = frozenset(
    {
        "bf16_sha256",
        "source_name",
        "source_shard",
        "source_payload_start",
        "source_payload_end",
    }
)
SCALAR_EVENTS = frozenset({"string", "number", "boolean", "null"})


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    model = Path(
        "/home/brandonmusic/models/"
        "GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bf16-root", type=Path, default=project / "bf16_layers"
    )
    parser.add_argument(
        "--bf16-seal",
        type=Path,
        default=project / "evidence" / "bf16_source_seal.json",
    )
    parser.add_argument(
        "--source-inventory",
        type=Path,
        default=model / "reproducibility/r10/inventories/source_inventory.json",
    )
    parser.add_argument("--mcg-model-root", type=Path, default=model)
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "evidence" / "bf16_mcg_source_crosswalk.json",
    )
    parser.add_argument("--chunk-bytes", type=int, default=32 << 20)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 32 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def load_json_object(path: Path, *, role: str) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{role} is not a JSON object: {path}")
    return value


def target_name(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"


def expected_targets() -> list[tuple[int, int, str, str]]:
    return [
        (layer, expert, projection, target_name(layer, expert, projection))
        for layer in LAYERS
        for expert in range(EXPERTS)
        for projection in PROJECTIONS
    ]


def stream_mcg_bf16_receipts(
    path: Path, *, expected_bases: set[str]
) -> dict[str, dict[str, Any]]:
    """Retain only the explicit BF16 source-provenance fields from a sidecar."""

    results: dict[str, dict[str, Any]] = {}
    current_name: str | None = None
    current_prefix: str | None = None
    current_field: str | None = None
    retained: dict[str, Any] | None = None

    with path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle):
            if prefix == "tensor_provenance" and event == "map_key":
                name = str(value)
                if name in expected_bases:
                    current_name = name
                    current_prefix = f"tensor_provenance.{name}"
                    retained = {}
                else:
                    current_name = None
                    current_prefix = None
                    retained = None
                current_field = None
                continue

            if current_name is None or current_prefix is None or retained is None:
                continue

            if prefix == current_prefix and event == "map_key":
                current_field = str(value)
                continue

            if (
                current_field in MCG_BF16_FIELDS
                and prefix == f"{current_prefix}.{current_field}"
                and event in SCALAR_EVENTS
            ):
                if current_field in retained:
                    raise ValueError(
                        f"duplicate retained field {current_field!r} for "
                        f"{current_name!r} in {path}"
                    )
                retained[current_field] = value
                current_field = None
                continue

            if prefix == current_prefix and event == "end_map":
                missing = MCG_BF16_FIELDS.difference(retained)
                if missing:
                    raise ValueError(
                        f"missing BF16 fields for {current_name!r} in {path}: "
                        f"{sorted(missing)}"
                    )
                if current_name in results:
                    raise ValueError(
                        f"duplicate tensor provenance {current_name!r} in {path}"
                    )
                results[current_name] = retained
                current_name = None
                current_prefix = None
                current_field = None
                retained = None

    missing_names = expected_bases.difference(results)
    extra_names = set(results).difference(expected_bases)
    if missing_names or extra_names:
        raise ValueError(
            f"sidecar tensor set mismatch in {path}: "
            f"missing={len(missing_names)} extra={len(extra_names)}"
        )
    return results


def read_header(path: Path) -> tuple[int, str, dict[str, Any], os.stat_result]:
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > opened.st_size - 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(raw_header)
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return 8 + header_length, sha256_bytes(raw_header), header, opened


def hash_range(
    handle: BinaryIO, *, start: int, end: int, chunk_bytes: int
) -> str:
    if start < 0 or end < start:
        raise ValueError(f"invalid payload range [{start}, {end})")
    handle.seek(start)
    remaining = end - start
    digest = hashlib.sha256()
    while remaining:
        block = handle.read(min(chunk_bytes, remaining))
        if not block:
            raise EOFError(f"short payload read with {remaining} bytes remaining")
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def same_file_identity(before: os.stat_result, after: os.stat_result) -> bool:
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    return all(getattr(before, field) == getattr(after, field) for field in fields)


def validate_sha256(value: object, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"invalid SHA256 for {role}: {value!r}")
    return value


def build_crosswalk(args: argparse.Namespace) -> dict[str, Any]:
    if args.chunk_bytes <= 0:
        raise ValueError("--chunk-bytes must be positive")

    project = Path(__file__).resolve().parent
    bf16_root = args.bf16_root.resolve()
    bf16_seal_path = args.bf16_seal.resolve()
    source_inventory_path = args.source_inventory.resolve()
    mcg_root = args.mcg_model_root.resolve()
    index_path = bf16_root / "model.safetensors.index.json"

    targets = expected_targets()
    if len(targets) != 3072 or len({item[3] for item in targets}) != 3072:
        raise AssertionError("frozen target enumeration is not exactly 3,072 unique tensors")

    seal = load_json_object(bf16_seal_path, role="BF16 source seal")
    source_inventory = load_json_object(
        source_inventory_path, role="R10 source inventory"
    )
    index = load_json_object(index_path, role="official safetensors index")
    weight_map = index.get("weight_map")
    inventory_entries = source_inventory.get("entries")
    if not isinstance(weight_map, dict) or not isinstance(inventory_entries, dict):
        raise ValueError("index/source inventory is missing its tensor mapping")

    index_sha256 = sha256_file(index_path)
    seal_index = seal.get("index")
    if not isinstance(seal_index, dict) or index_sha256 != seal_index.get("sha256"):
        raise ValueError("current official index does not match BF16 source seal")

    source_shards = seal.get("shards")
    if not isinstance(source_shards, dict):
        raise ValueError("BF16 source seal has no shard mapping")

    receipts: dict[str, dict[str, Any]] = {}
    sidecar_paths: list[Path] = []
    sidecar_sha256: dict[str, str] = {}
    for layer in LAYERS:
        sidecar = mcg_root / f"r7-experts-layer-{layer:03d}.json"
        sidecar_paths.append(sidecar)
        layer_bases = {
            name.removesuffix(".weight")
            for item_layer, _, _, name in targets
            if item_layer == layer
        }
        layer_receipts = stream_mcg_bf16_receipts(
            sidecar, expected_bases=layer_bases
        )
        overlap = set(receipts).intersection(layer_receipts)
        if overlap:
            raise ValueError(f"duplicate cross-layer sidecar entries: {sorted(overlap)[:3]}")
        receipts.update(layer_receipts)
        sidecar_sha256[sidecar.name] = sha256_file(sidecar)
    if len(receipts) != 3072:
        raise ValueError(f"expected 3,072 sidecar receipts, found {len(receipts)}")

    grouped: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
    for item in targets:
        shard = weight_map.get(item[3])
        if not isinstance(shard, str):
            raise ValueError(f"official index is missing target {item[3]}")
        grouped[shard].append(item)
    if set(grouped) != set(source_shards):
        raise ValueError(
            "selected shard set differs from BF16 source seal: "
            f"selected={sorted(grouped)} sealed={sorted(source_shards)}"
        )

    records: list[dict[str, Any]] = []
    mismatches: list[dict[str, str]] = []
    file_receipts: dict[str, dict[str, Any]] = {}
    print(f"Hashing 3,072 exact BF16 tensor payloads across {len(grouped)} shards...")

    for shard_number, shard in enumerate(sorted(grouped), start=1):
        shard_path = bf16_root / shard
        data_start, header_sha256, header, before = read_header(shard_path)
        sealed_shard = source_shards.get(shard)
        if not isinstance(sealed_shard, dict):
            raise ValueError(f"missing sealed shard record: {shard}")
        if header_sha256 != sealed_shard.get("header_sha256"):
            raise ValueError(f"header hash differs from BF16 source seal: {shard}")
        if before.st_size != sealed_shard.get("bytes"):
            raise ValueError(f"shard size differs from BF16 source seal: {shard}")

        planned: list[tuple[int, int, tuple[int, int, str, str], dict[str, Any]]] = []
        for item in grouped[shard]:
            tensor = item[3]
            raw = header.get(tensor)
            if not isinstance(raw, dict):
                raise ValueError(f"shard header is missing {tensor}")
            offsets = raw.get("data_offsets")
            shape = raw.get("shape")
            dtype = raw.get("dtype")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
            ):
                raise ValueError(f"invalid data offsets for {tensor}")
            relative_start, relative_end = offsets
            absolute_start = data_start + relative_start
            absolute_end = data_start + relative_end
            if dtype != "BF16" or absolute_end - absolute_start != EXPECTED_TENSOR_BYTES:
                raise ValueError(
                    f"unexpected dtype/byte count for {tensor}: "
                    f"dtype={dtype!r} bytes={absolute_end - absolute_start}"
                )
            if shape not in ([2048, 6144], [6144, 2048]):
                raise ValueError(f"unexpected shape for {tensor}: {shape!r}")
            planned.append((absolute_start, absolute_end, item, raw))

        with shard_path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not same_file_identity(before, opened):
                raise RuntimeError(f"shard changed between header and payload open: {shard}")
            for absolute_start, absolute_end, item, raw in sorted(planned):
                layer, expert, projection, tensor = item
                base = tensor.removesuffix(".weight")
                mcg = receipts[base]
                inventory = inventory_entries.get(tensor)
                if not isinstance(inventory, dict):
                    raise ValueError(f"R10 source inventory is missing {tensor}")

                actual_hash = hash_range(
                    handle,
                    start=absolute_start,
                    end=absolute_end,
                    chunk_bytes=args.chunk_bytes,
                )
                mcg_hash = validate_sha256(
                    mcg["bf16_sha256"], role=f"MCG receipt {tensor}"
                )
                inventory_hash = validate_sha256(
                    inventory.get("payload_sha256"),
                    role=f"R10 source inventory {tensor}",
                )
                checks = {
                    "actual_equals_mcg_bf16_sha256": actual_hash == mcg_hash,
                    "actual_equals_r10_source_inventory_sha256": (
                        actual_hash == inventory_hash
                    ),
                    "mcg_source_name_equals_target": mcg["source_name"] == tensor,
                    "mcg_source_shard_equals_actual": mcg["source_shard"] == shard,
                    "mcg_payload_range_equals_actual": (
                        mcg["source_payload_start"] == absolute_start
                        and mcg["source_payload_end"] == absolute_end
                    ),
                    "r10_source_shard_equals_actual": inventory.get("shard") == shard,
                    "r10_payload_range_equals_actual": (
                        inventory.get("payload_start") == absolute_start
                        and inventory.get("payload_end") == absolute_end
                    ),
                    "r10_dtype_equals_actual": inventory.get("dtype") == raw.get("dtype"),
                    "r10_shape_equals_actual": inventory.get("shape") == raw.get("shape"),
                    "r10_nbytes_equals_actual": (
                        inventory.get("nbytes") == absolute_end - absolute_start
                    ),
                }
                failed = sorted(name for name, passed in checks.items() if not passed)
                if failed:
                    mismatches.append({"tensor_name": tensor, "checks": ",".join(failed)})
                records.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "projection": projection,
                        "tensor_name": tensor,
                        "source_shard": shard,
                        "payload_start": absolute_start,
                        "payload_end": absolute_end,
                        "payload_bytes": absolute_end - absolute_start,
                        "actual_fresh_bf16_sha256": actual_hash,
                        "mcg_recorded_bf16_sha256": mcg_hash,
                        "r10_source_inventory_bf16_sha256": inventory_hash,
                        "checks": checks,
                        "all_match": not failed,
                    }
                )

        after = shard_path.stat()
        if not same_file_identity(before, after):
            raise RuntimeError(f"shard changed while payloads were hashed: {shard}")
        file_receipts[shard] = {
            "bytes": before.st_size,
            "device": before.st_dev,
            "inode": before.st_ino,
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
            "header_sha256": header_sha256,
            "sealed_full_file_sha256": sealed_shard.get("sha256"),
            "selected_tensor_count": len(planned),
            "stable_during_hashing": True,
        }
        print(
            f"[{shard_number:02d}/{len(grouped):02d}] {shard}: "
            f"{len(planned)} payloads"
        )

    records.sort(key=lambda record: (record["layer"], record["expert"], record["projection"]))
    if len(records) != 3072 or len({record["tensor_name"] for record in records}) != 3072:
        raise AssertionError("crosswalk did not produce exactly 3,072 unique records")

    layer_counts = Counter(str(record["layer"]) for record in records)
    projection_counts = Counter(record["projection"] for record in records)
    script_path = Path(__file__).resolve()
    body: dict[str, Any] = {
        "schema": "glm52-fresh-sqg-bf16-mcg-source-crosswalk-v1",
        "purpose": "external_source_identity_evidence_only",
        "capture_input": False,
        "encoder_input": False,
        "scientific_claim": (
            "The 3,072 official BF16 tensor payloads selected for the fresh SQG "
            "pilot are byte-identical to the BF16 sources recorded by the prior "
            "MCG quant and by the authoritative R10 source inventory."
        ),
        "scope": {
            "layers": list(LAYERS),
            "experts_per_layer": EXPERTS,
            "projections": list(PROJECTIONS),
            "target_tensor_count": 3072,
        },
        "mcg_sidecar_access_policy": {
            "method": "streaming_event_parser_allowlist",
            "retained_semantic_fields": sorted(MCG_BF16_FIELDS),
            "transform_scale_permutation_seed_trellis_payload_values_retained": [],
            "mcg_quantization_values_used_as_capture_or_encoder_inputs": False,
        },
        "inputs": {
            "generator": {
                "path": str(script_path.relative_to(project)),
                "sha256": sha256_file(script_path),
            },
            "bf16_source_seal": {
                "path": str(bf16_seal_path),
                "sha256": sha256_file(bf16_seal_path),
                "repo": seal.get("repo"),
                "revision": seal.get("revision"),
            },
            "official_index": {
                "path": str(index_path),
                "sha256": index_sha256,
            },
            "r10_source_inventory": {
                "path": str(source_inventory_path),
                "sha256": sha256_file(source_inventory_path),
                "declared_inventory_sha256": source_inventory.get("inventory_sha256"),
            },
            "mcg_bf16_receipt_sidecars": {
                path.name: {
                    "path": str(path),
                    "sha256": sidecar_sha256[path.name],
                }
                for path in sidecar_paths
            },
        },
        "bf16_shards": file_receipts,
        "results": {
            "passed": not mismatches,
            "tensor_count": len(records),
            "matching_tensor_count": sum(record["all_match"] for record in records),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "layer_counts": dict(sorted(layer_counts.items())),
            "projection_counts": dict(sorted(projection_counts.items())),
            "records_sha256": sha256_bytes(canonical_bytes(records)),
        },
        "records": records,
    }
    body["seal_sha256"] = sha256_bytes(canonical_bytes(body))
    return body


def write_json_atomic(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    result = build_crosswalk(args)
    output = args.output.resolve()
    write_json_atomic(output, result)
    print(
        f"wrote {output}: passed={result['results']['passed']} "
        f"matching={result['results']['matching_tensor_count']}/"
        f"{result['results']['tensor_count']} seal={result['seal_sha256']}"
    )


if __name__ == "__main__":
    main()
