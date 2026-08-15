#!/usr/bin/env python3
"""Export only the frozen per-tensor K3/K4 assignment for the SQG pilot.

This is the deliberate boundary between the existing BMMLaw checkpoint and
the fresh encoder.  The output contains tensor names and bit widths only.  It
does not copy transform vectors, scales, seeds, permutations, packed states,
reconstructions, or codec markers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .pilot_config import selected_layers


LAYERS = selected_layers()
EXPERTS = 256
PROJECTIONS = ("down_proj", "gate_proj", "up_proj")
EXPECTED_PER_LAYER = EXPERTS * len(PROJECTIONS)
EXPECTED_HISTOGRAM = {"3": 384, "4": 384}
TENSOR_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>down_proj|gate_proj|up_proj)$"
)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_layer_map(layer: int, raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise TypeError(f"layer {layer}: bit_map must be an object")
    result = {str(name): int(bits) for name, bits in raw.items()}
    if len(result) != EXPECTED_PER_LAYER:
        raise ValueError(
            f"layer {layer}: expected {EXPECTED_PER_LAYER} tensors, got {len(result)}"
        )

    observed: set[tuple[int, str]] = set()
    histogram = {"3": 0, "4": 0}
    for name, bits in result.items():
        match = TENSOR_RE.fullmatch(name)
        if match is None:
            raise ValueError(f"layer {layer}: unexpected tensor name {name!r}")
        if int(match.group("layer")) != layer:
            raise ValueError(f"layer {layer}: cross-layer tensor {name!r}")
        expert = int(match.group("expert"))
        projection = match.group("projection")
        if not 0 <= expert < EXPERTS:
            raise ValueError(f"layer {layer}: expert out of range in {name!r}")
        if bits not in (3, 4):
            raise ValueError(f"layer {layer}: only K3/K4 are permitted, got K{bits}")
        observed.add((expert, projection))
        histogram[str(bits)] += 1

    expected = {
        (expert, projection)
        for expert in range(EXPERTS)
        for projection in PROJECTIONS
    }
    if observed != expected:
        missing = sorted(expected - observed)[:8]
        extra = sorted(observed - expected)[:8]
        raise ValueError(f"layer {layer}: tensor grid mismatch; missing={missing}, extra={extra}")
    if histogram != EXPECTED_HISTOGRAM:
        raise ValueError(
            f"layer {layer}: expected {EXPECTED_HISTOGRAM}, got {histogram}"
        )
    return dict(sorted(result.items()))


def build_contract(source_model: Path) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for layer in LAYERS:
        sidecar = source_model / f"r7-experts-layer-{layer:03d}.json"
        if not sidecar.is_file():
            raise FileNotFoundError(sidecar)
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        if int(raw.get("layer", -1)) != layer:
            raise ValueError(f"{sidecar}: layer declaration mismatch")
        bit_map = validate_layer_map(layer, raw.get("bit_map"))
        layers[str(layer)] = {
            "bit_map": bit_map,
            "histogram": EXPECTED_HISTOGRAM,
        }

    return {
        "schema": "glm52-fresh-sqg-frozen-per-tensor-bit-allocation-v1",
        "purpose": "topology-neutral per-tensor K3/K4 experimental control",
        "layers": layers,
        "totals": {
            "layers": len(LAYERS),
            "tensors": len(LAYERS) * EXPECTED_PER_LAYER,
            "K3": len(LAYERS) * EXPECTED_HISTOGRAM["3"],
            "K4": len(LAYERS) * EXPECTED_HISTOGRAM["4"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = build_contract(args.source_model.resolve())
    payload = canonical_json_bytes(contract)
    write_atomic(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "layers": list(LAYERS),
                "tensors": contract["totals"]["tensors"],
                "K3": contract["totals"]["K3"],
                "K4": contract["totals"]["K4"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
