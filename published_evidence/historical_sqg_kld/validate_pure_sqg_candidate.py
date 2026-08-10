#!/usr/bin/env python3
"""Fail closed unless the four-layer candidate is the intended pure-SQG arm."""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SQG_CODEBOOK = "sqg_xor_cheb_t12"
SQG_MARKER = 0x53514731
TENSORS_PER_LAYER = 256 * 3
KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>trellis|sqg|mcg)$"
)


def fail(message: str) -> "NoReturn":
    raise RuntimeError(message)


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read valid JSON {path}: {exc}")


def parse_layers(raw: str) -> list[int]:
    try:
        layers = [int(item) for item in raw.split(",")]
    except ValueError as exc:
        fail(f"invalid --layers value {raw!r}: {exc}")
    if not layers or len(layers) != len(set(layers)) or any(x < 0 for x in layers):
        fail(f"layers must be unique nonnegative integers: {raw!r}")
    return sorted(layers)


def normalize_layer_overrides(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        fail("r7_routed_experts.codebook_overrides must be an object")
    normalized: dict[int, str] = {}
    for key, codebook in value.items():
        try:
            layer = int(key)
        except (TypeError, ValueError) as exc:
            fail(f"invalid layer override key {key!r}: {exc}")
        if layer in normalized:
            fail(f"duplicate normalized layer override: {layer}")
        normalized[layer] = codebook
    return normalized


def safe_shard(candidate: Path, shard_name: str) -> Path:
    if not isinstance(shard_name, str) or not shard_name:
        fail(f"invalid shard name in index: {shard_name!r}")
    shard = (candidate / shard_name).resolve()
    try:
        shard.relative_to(candidate)
    except ValueError:
        fail(f"index shard escapes candidate directory: {shard_name!r}")
    if not shard.is_file():
        fail(f"indexed shard is missing: {shard}")
    return shard


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            fail(f"truncated safetensors length prefix: {path}")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len <= 1 or header_len > 512 * 1024 * 1024:
            fail(f"implausible safetensors header length {header_len}: {path}")
        raw_header = handle.read(header_len)
        if len(raw_header) != header_len:
            fail(f"truncated safetensors header: {path}")
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        fail(f"invalid safetensors header JSON {path}: {exc}")
    if not isinstance(header, dict):
        fail(f"safetensors header is not an object: {path}")
    return 8 + header_len, header


def validate_marker_payloads(
    marker_keys: list[str], weight_map: dict[str, str], candidate: Path
) -> Counter[str]:
    grouped: dict[Path, list[str]] = defaultdict(list)
    for key in marker_keys:
        grouped[safe_shard(candidate, weight_map[key])].append(key)

    counts: Counter[str] = Counter()
    for shard, keys in grouped.items():
        data_start, header = read_safetensors_header(shard)
        with shard.open("rb") as handle:
            for key in keys:
                descriptor = header.get(key)
                if not isinstance(descriptor, dict):
                    fail(f"indexed marker absent from shard header: {key} in {shard}")
                if descriptor.get("dtype") != "I32":
                    fail(f"SQG marker is not I32: {key}")
                if descriptor.get("shape") not in ([], [1]):
                    fail(f"SQG marker is not scalar: {key}")
                offsets = descriptor.get("data_offsets")
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or not all(isinstance(x, int) for x in offsets)
                    or offsets[1] - offsets[0] != 4
                ):
                    fail(f"SQG marker has invalid data offsets: {key}")
                handle.seek(data_start + offsets[0])
                raw_value = handle.read(4)
                if len(raw_value) != 4:
                    fail(f"SQG marker payload is truncated: {key}")
                (value,) = struct.unpack("<i", raw_value)
                if value != SQG_MARKER:
                    fail(
                        f"SQG marker value mismatch for {key}: "
                        f"0x{value & 0xFFFFFFFF:08x}"
                    )
                counts[shard.name] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--layers", default="6,28,52,77")
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    source = args.source.resolve()
    layers = parse_layers(args.layers)
    expected_layer_set = set(layers)
    if not candidate.is_dir() or not source.is_dir():
        fail("candidate and source must both be existing directories")

    config = load_json(candidate / "config.json")
    if not isinstance(config, dict):
        fail("config.json must be an object")
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        fail("config.json has no quantization_config object")
    r7 = quant.get("r7_routed_experts")
    if not isinstance(r7, dict):
        fail("config.json has no r7_routed_experts object")
    if r7.get("codebook") != "mcg":
        fail("global untouched-layer codebook must remain MCG for this four-layer arm")
    overrides = normalize_layer_overrides(r7.get("codebook_overrides"))
    expected_overrides = {layer: SQG_CODEBOOK for layer in layers}
    if overrides != expected_overrides:
        fail(
            f"codebook override mismatch: expected {expected_overrides}, got {overrides}"
        )
    tensor_overrides = r7.get("codebook_tensor_overrides", {})
    if tensor_overrides not in ({}, None):
        fail(f"pure-SQG arm forbids tensor overrides: {tensor_overrides!r}")

    index = load_json(candidate / "model.safetensors.index.json")
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        fail("model.safetensors.index.json has no weight_map object")
    weight_map: dict[str, str] = index["weight_map"]
    recognized: dict[int, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    parsed_fields: dict[str, tuple[int, int, str, str]] = {}
    sqg_outside: list[str] = []
    for key in weight_map:
        match = KEY_RE.match(key)
        if match is None:
            continue
        layer = int(match.group("layer"))
        expert = int(match.group("expert"))
        projection = match.group("projection")
        kind = match.group("kind")
        base = key[: -(len(kind) + 1)]
        recognized[layer][kind].add(base)
        parsed_fields[base] = (layer, expert, projection, kind)
        if kind == "sqg" and layer not in expected_layer_set:
            sqg_outside.append(key)
    if sqg_outside:
        fail(f"SQG markers exist outside selected layers: {sqg_outside[:3]}")

    marker_keys: list[str] = []
    layer_reports: list[dict[str, Any]] = []
    for layer in layers:
        kinds = recognized.get(layer, {})
        trellis = kinds.get("trellis", set())
        sqg = kinds.get("sqg", set())
        mcg = kinds.get("mcg", set())
        if len(trellis) != TENSORS_PER_LAYER:
            fail(f"layer {layer} has {len(trellis)} trellis tensors, expected 768")
        if sqg != trellis:
            fail(
                f"layer {layer} does not have one SQG marker per trellis tensor: "
                f"trellis={len(trellis)} sqg={len(sqg)}"
            )
        if mcg:
            fail(f"layer {layer} still has {len(mcg)} MCG markers")

        observed_pairs = {
            (parsed_fields[base][1], parsed_fields[base][2]) for base in trellis
        }
        expected_pairs = {
            (expert, projection)
            for expert in range(256)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        if observed_pairs != expected_pairs:
            fail(f"layer {layer} expert/projection topology is incomplete")

        candidate_sidecar = load_json(
            candidate / f"r7-experts-layer-{layer:03d}.json"
        )
        source_sidecar = load_json(source / f"r7-experts-layer-{layer:03d}.json")
        candidate_bits = candidate_sidecar.get("bit_map")
        source_bits = source_sidecar.get("bit_map")
        if not isinstance(candidate_bits, dict) or candidate_bits != source_bits:
            fail(f"layer {layer} per-tensor bit map differs from protected source")
        bit_histogram = Counter(candidate_bits.values())
        if bit_histogram != Counter({3: 384, 4: 384}):
            fail(
                f"layer {layer} is not pure K3/K4 SQG: {dict(bit_histogram)}"
            )

        for base in sorted(sqg):
            marker_key = f"{base}.sqg"
            trellis_key = f"{base}.trellis"
            if weight_map[marker_key] != weight_map[trellis_key]:
                fail(f"marker and trellis are split across shards: {base}")
            marker_keys.append(marker_key)
        layer_reports.append(
            {
                "layer": layer,
                "trellis_tensors": len(trellis),
                "sqg_markers": len(sqg),
                "mcg_markers": len(mcg),
                "bit_histogram": {"3": 384, "4": 384, "5": 0},
                "bit_map_matches_source": True,
            }
        )

    marker_shards = validate_marker_payloads(marker_keys, weight_map, candidate)
    report = {
        "schema": "glm52-pure-sqg-four-layer-preflight-v1",
        "candidate": str(candidate),
        "protected_source": str(source),
        "selected_layers": layers,
        "global_unselected_layer_codebook": "mcg",
        "selected_layer_codebook": SQG_CODEBOOK,
        "tensor_overrides": {},
        "sqg_marker": {"suffix": ".sqg", "int32": SQG_MARKER},
        "selected_trellis_tensors": len(marker_keys),
        "selected_sqg_markers": len(marker_keys),
        "selected_mcg_markers": 0,
        "sqg_markers_outside_selected_layers": 0,
        "marker_payloads_verified": True,
        "marker_shards": dict(sorted(marker_shards.items())),
        "layers": layer_reports,
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
