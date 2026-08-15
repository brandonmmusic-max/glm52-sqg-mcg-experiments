#!/usr/bin/env python3
"""Validate the downloaded GLM-5.2 SQG W4A8 checkpoint without rewriting it.

Gates covered (host-side, CPU, zero third-party deps):
  * quantization_config.json contract fields (full_w4a8 or exact updated-QSRT
    coupled H512/H128/H128, per-layer census, down targets for layers 3..78,
    r7 routed contract, K6 non-routed contract, no A16).
  * model.safetensors.index.json closure: every mapped shard exists on disk,
    every mapped tensor is present in its shard header.
  * Routed census from shard headers: for every routed layer 3..78 and every
    expert, gate/up/down trellis+rotations+SQG sentinel present, K3/K4 split
    matches the declared per-layer 48/72/96-K4 coupled budget (or the shipped
    384/384 uncoupled census), no MCG/mul1 marker anywhere, all trellis
    int16, all rotation vectors fp16 with matching extents, and the
    intermediate axis is H128-divisible.
  * Non-routed census: every K5/K6 dense entry has trellis+suh+svh+sqg; K6
    count matches the sealed contract.
  * MTP layer 78 accounted (routed + shared rotations).
  * Revision proof: local SHA-256 of the config/index/manifest files and a
    sample of payload shards equals the lfs/oid hashes served by the Hugging
    Face tree API for the pinned revision (skipped with --offline).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO = "brandonmusic/GLM-5.2-SQG-W4A8"


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_remote_tree(revision: str) -> dict[str, dict]:
    files: dict[str, dict] = {}
    url = (
        f"https://huggingface.co/api/models/{REPO}/tree/{revision}"
        "?recursive=true"
    )
    cursor = None
    while True:
        target = url + (f"&cursor={cursor}" if cursor else "")
        request = urllib.request.Request(
            target, headers={"User-Agent": "sm120-codec-validate"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            batch = json.load(response)
            link = response.headers.get("Link", "")
        for item in batch:
            if item.get("type") == "file":
                files[item["path"]] = item
        if "next" in link:
            import re

            match = re.search(r"cursor=([^&>]+)", link)
            cursor = match.group(1) if match else None
            if cursor is None:
                break
        else:
            break
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="skip payload-shard hash sampling (config/index hashes only)")
    parser.add_argument(
        "--require-all-coupled",
        action="store_true",
        help="require every routed target/draft layer 3..78 to use coupled coordinates",
    )
    parser.add_argument(
        "--require-target-coupled-preserved-mtp78",
        action="store_true",
        help=(
            "require target routed layers 3..77 to use coupled coordinates and "
            "require source MTP layer 78 to remain uncoupled at its sealed census"
        ),
    )
    parser.add_argument("--sample-shards", type=int, default=6)
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()
    root = Path(args.model)
    problems: list[str] = []

    config = json.loads((root / "config.json").read_text())
    if config.get("architectures") != ["GlmMoeDsaForCausalLM"]:
        problems.append(f"unexpected architectures: {config.get('architectures')}")
    experts = int(config["n_routed_experts"])
    moe_intermediate = int(config["moe_intermediate_size"])
    if moe_intermediate % 128:
        problems.append(
            f"moe_intermediate_size {moe_intermediate} is not H128-aligned"
        )

    quant = json.loads((root / "quantization_config.json").read_text())
    contract = quant.get("glm_sqg_w4a8", {})
    coupled = contract.get("schema") == (
        "glm52_sqg_atoms_v2_coupled_h512_h128_w4a8_v1"
    )
    expected_contract = {
        "schema": (
            "glm52_sqg_atoms_v2_coupled_h512_h128_w4a8_v1"
            if coupled
            else "glm52_sqg_atoms_v2_w4a8_v1"
        ),
        "execution": "full_coupled_w4a8" if coupled else "full_w4a8",
        "codebook": "sqg_xor_cheb_t12",
        "rates": "independent_per_tensor_k3_k4",
        "direct_e4m3_weights": True,
        "allow_a16_fallback": False,
        "activation": "silu_gate_times_up",
        "topology": "topology_neutral",
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            problems.append(f"contract {key}={contract.get(key)!r} != {value!r}")
    routed_layers = set(range(3, 79))
    target_routed_layers = set(range(3, 78))
    coupled_layers: set[int] = set()
    if coupled:
        expected_transform = {
            "residual_policy": "coupled_block_hadamard",
            "residual_block_size": 512,
            "residual_draw": 0,
            "preactivation_block_size": 128,
            "postactivation_block_size": 128,
            "activation": "silu",
        }
        observed_transform = contract.get("coupled_transform")
        if (
            not isinstance(observed_transform, dict)
            or any(
                observed_transform.get(key) != value
                for key, value in expected_transform.items()
            )
            or observed_transform.get(
                "activation_equation", "silu(gate)*up"
            ) != "silu(gate)*up"
        ):
            problems.append(
                "coupled_transform does not match exact updated-QSRT "
                "H512/H128/H128 SiLU semantics"
            )
        raw_layers = contract.get("coupled_layers")
        if (
            not isinstance(raw_layers, list)
            or any(type(layer) is not int for layer in raw_layers)
            or raw_layers != sorted(set(raw_layers))
            or set(raw_layers) - routed_layers
        ):
            problems.append(f"invalid coupled_layers: {raw_layers!r}")
        else:
            coupled_layers = set(raw_layers)
        draws = contract.get("coupled_rotation_draws")
        if not isinstance(draws, dict) or set(draws) != {
            str(layer) for layer in coupled_layers
        }:
            problems.append("coupled draw maps do not exactly cover coupled_layers")
        else:
            for layer, values in draws.items():
                if (
                    not isinstance(values, list)
                    or len(values) != 256
                    or any(type(value) is not int or value not in (0, 6) for value in values)
                ):
                    problems.append(
                        f"coupled layer {layer} draw map is not 256 draw-0/6 choices"
                    )
        if args.require_all_coupled and coupled_layers != routed_layers:
            problems.append(
                "full coupled release requires layers 3..78; "
                f"observed {len(coupled_layers)} layers"
            )
        if (
            args.require_target_coupled_preserved_mtp78
            and coupled_layers != target_routed_layers
        ):
            problems.append(
                "target-coupled release requires layers 3..77 and preserves "
                f"MTP78; observed {sorted(coupled_layers)}"
            )
    elif args.require_all_coupled:
        problems.append("checkpoint is not the coupled updated-QSRT schema")
    elif args.require_target_coupled_preserved_mtp78:
        problems.append("checkpoint is not the coupled updated-QSRT schema")

    census = contract.get("per_layer_bit_census")
    if coupled:
        if not isinstance(census, dict) or set(census) != {
            str(layer) for layer in routed_layers
        }:
            problems.append("coupled census does not cover every layer 3..78")
        else:
            for layer in routed_layers:
                declared = census[str(layer)]
                expected = (
                    declared
                    if layer in coupled_layers
                    else {"k3": 384, "k4": 384, "total": 768}
                )
                valid_coupled = (
                    isinstance(declared, dict)
                    and declared.get("total") == 768
                    and declared.get("k3", -1) + declared.get("k4", -1) == 768
                    and declared.get("k4") in (48, 72, 96)
                )
                if layer in coupled_layers and not valid_coupled:
                    problems.append(
                        f"coupled layer {layer} declared census is not an exact "
                        f"48/72/96-K4 budget: {declared!r}"
                    )
                elif layer not in coupled_layers and declared != expected:
                    problems.append(
                        f"uncoupled layer {layer} declared census {declared!r} "
                        f"!= {expected!r}"
                    )
    elif census != {"k3": 384, "k4": 384, "total": 768}:
        problems.append(f"sealed census mismatch: {census}")
    if args.require_target_coupled_preserved_mtp78:
        assembly_path = root / "COUPLED_REENCODE_MANIFEST.json"
        if not assembly_path.is_file():
            problems.append("coupled assembly manifest is absent")
        else:
            assembly = json.loads(assembly_path.read_text())
            preserved = assembly.get("preserved_mtp_layer_78", {})
            mtp_shard = root / "r7-experts-layer-078.safetensors"
            mtp_sidecar = root / "r7-experts-layer-078.json"
            if (
                assembly.get("all_target_routed_layers_coupled") is not True
                or assembly.get("mtp_layer_78_policy")
                != "preserve_source_unchanged"
                or preserved.get("bit_census")
                != {"k3": 384, "k4": 384, "total": 768}
                or not mtp_shard.is_file()
                or not mtp_sidecar.is_file()
                or sha256_file(mtp_shard) != preserved.get("shard_sha256")
                or sha256_file(mtp_sidecar) != preserved.get("sidecar_sha256")
            ):
                problems.append(
                    "MTP layer 78 does not match its preserved source hashes/census"
                )
    sealed_parallel = contract.get("parallelism", {})
    sealed_regime = (
        sealed_parallel.get("tensor_parallel_size"),
        sealed_parallel.get("pipeline_parallel_size", 1),
        sealed_parallel.get("decode_context_parallel_size"),
    )
    if sealed_regime not in {(1, 8, 1), (4, 1, 4)}:
        problems.append(f"sealed parallelism {sealed_regime} not contract-legal")
    targets = contract.get("down_targets", {})
    expected_layers = {str(layer) for layer in routed_layers}
    if set(targets) != expected_layers:
        problems.append(
            "down_targets do not cover layers 3..78: "
            f"missing={sorted(expected_layers - set(targets))[:5]} "
            f"extra={sorted(set(targets) - expected_layers)[:5]}"
        )
    r7 = quant.get("r7_routed_experts", {})
    if r7.get("schema") != "glm52-sqg-atoms-v2-routed-v1" or r7.get(
        "moe_layers"
    ) != [3, 78]:
        problems.append(f"r7 routed contract unexpected: {r7.get('schema')}, {r7.get('moe_layers')}")
    k6 = quant.get("sqg_k6_nonrouted", {})
    if k6.get("execution") != "native_sqg_k6_w6a16" or k6.get("bits") != 6:
        problems.append(f"K6 non-routed contract unexpected: {k6}")
    k6_expected_count = int(k6.get("matrix_count", 0))

    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    missing_shards = [shard for shard in shards if not (root / shard).is_file()]
    if missing_shards:
        problems.append(f"missing shard files: {missing_shards[:5]} (+{len(missing_shards)-5 if len(missing_shards)>5 else 0})")
        emit(problems, {}, args)
        return

    headers = {shard: read_safetensors_header(root / shard) for shard in shards}
    tensors_on_disk: dict[str, tuple[str, dict]] = {}
    for shard, header in headers.items():
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            tensors_on_disk[name] = (shard, meta)
    unmapped = [n for n in weight_map if n not in tensors_on_disk]
    if unmapped:
        problems.append(f"{len(unmapped)} indexed tensors absent from shards, e.g. {unmapped[:4]}")
    stray = [n for n in tensors_on_disk if n not in weight_map]
    if stray:
        problems.append(f"{len(stray)} shard tensors not in index, e.g. {stray[:4]}")

    # Routed census.
    per_layer = defaultdict(Counter)
    per_layer_proj = defaultdict(lambda: defaultdict(Counter))
    marker_issues = 0
    routed_missing: list[str] = []
    shared_missing: list[str] = []
    for layer in range(3, 79):
        base = f"model.layers.{layer}.mlp.experts"
        for shared_name in (f"{base}.r7_shared.gate_up_suh", f"{base}.r7_shared.down_svh"):
            if shared_name not in tensors_on_disk:
                shared_missing.append(shared_name)
        for expert in range(experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"{base}.{expert}.{projection}"
                trellis = tensors_on_disk.get(f"{prefix}.trellis")
                if trellis is None:
                    routed_missing.append(f"{prefix}.trellis")
                    continue
                meta = trellis[1]
                if meta["dtype"] != "I16" or len(meta["shape"]) != 3:
                    problems.append(f"{prefix}.trellis bad dtype/shape {meta}")
                    continue
                bits = meta["shape"][2] // 16
                per_layer[layer][bits] += 1
                per_layer_proj[layer][projection][bits] += 1
                vector = "suh" if projection == "down_proj" else "svh"
                vmeta = tensors_on_disk.get(f"{prefix}.{vector}")
                if vmeta is None:
                    routed_missing.append(f"{prefix}.{vector}")
                elif vmeta[1]["dtype"] != "F16" or vmeta[1]["shape"] != [moe_intermediate]:
                    problems.append(f"{prefix}.{vector} bad meta {vmeta[1]}")
                if f"{prefix}.sqg" not in tensors_on_disk:
                    marker_issues += 1
                for forbidden in ("mcg", "mul1"):
                    if f"{prefix}.{forbidden}" in tensors_on_disk:
                        problems.append(f"{prefix} carries forbidden {forbidden} marker")
        counts = per_layer[layer]
        expected_census = (
            census.get(str(layer), {})
            if coupled and layer in coupled_layers and isinstance(census, dict)
            else {"k3": 384, "k4": 384, "total": 768}
        )
        if coupled and isinstance(census, dict):
            if census.get(str(layer)) != expected_census:
                problems.append(
                    f"layer {layer} declared census {census.get(str(layer))} "
                    f"!= {expected_census}"
                )
        if (
            counts[3] != expected_census["k3"]
            or counts[4] != expected_census["k4"]
            or sum(counts.values()) != expected_census["total"]
        ):
            problems.append(
                f"layer {layer} on-disk census {dict(counts)} != {expected_census}"
            )
    if routed_missing:
        problems.append(f"{len(routed_missing)} routed tensors missing, e.g. {routed_missing[:4]}")
    if shared_missing:
        problems.append(f"shared r7 rotations missing: {shared_missing[:4]}")
    if marker_issues:
        problems.append(f"{marker_issues} routed tensors lack the SQG sentinel")

    # Non-routed dense census (K5/K6 SQG + any forbidden markers).
    dense_bits = Counter()
    dense_missing_fields: list[str] = []
    mcg_markers_global = [n for n in tensors_on_disk if n.endswith(".mcg")]
    mul1_markers_global = [n for n in tensors_on_disk if n.endswith(".mul1")]
    if mcg_markers_global:
        problems.append(f"{len(mcg_markers_global)} MCG markers present, e.g. {mcg_markers_global[:3]}")
    if mul1_markers_global:
        problems.append(f"{len(mul1_markers_global)} mul1 markers present, e.g. {mul1_markers_global[:3]}")
    for name, (shard, meta) in tensors_on_disk.items():
        if not name.endswith(".trellis") or ".mlp.experts." in name:
            continue
        bits = meta["shape"][2] // 16 if len(meta["shape"]) == 3 else -1
        dense_bits[bits] += 1
        prefix = name.removesuffix(".trellis")
        for field in ("suh", "svh", "sqg"):
            if f"{prefix}.{field}" not in tensors_on_disk:
                dense_missing_fields.append(f"{prefix}.{field}")
    if dense_missing_fields:
        problems.append(f"{len(dense_missing_fields)} dense sidecar fields missing, e.g. {dense_missing_fields[:4]}")
    if k6_expected_count and dense_bits.get(6, 0) != k6_expected_count:
        problems.append(f"K6 dense count {dense_bits.get(6,0)} != sealed {k6_expected_count}")
    bad_rates = {b: c for b, c in dense_bits.items() if b not in (5, 6)}
    if bad_rates:
        problems.append(f"unexpected dense trellis rates: {bad_rates}")

    # Revision proof against the HF tree API.
    revision_checked = []
    if not args.offline:
        try:
            remote = fetch_remote_tree(args.revision)
            check_files = [
                "config.json",
                "quantization_config.json",
                "model.safetensors.index.json",
                "SHA256SUMS",
                "FULL_SQG_NATIVE_MANIFEST.json",
            ]
            if not args.quick:
                payload = [p for p in remote if p.endswith(".safetensors")]
                random.seed(20260812)
                check_files += random.sample(payload, min(args.sample_shards, len(payload)))
            for rel in check_files:
                item = remote.get(rel)
                local = root / rel
                if item is None or not local.is_file():
                    problems.append(f"revision check: {rel} missing (remote={item is not None}, local={local.is_file()})")
                    continue
                lfs = item.get("lfs") or {}
                expected = lfs.get("oid")
                if expected:
                    actual = sha256_file(local)
                    ok = actual == expected
                else:
                    # Non-LFS blob: oid is a git blob sha1 over
                    # "blob <size>\0" + content.
                    data = local.read_bytes()
                    actual = hashlib.sha1(
                        b"blob %d\x00" % len(data) + data
                    ).hexdigest()
                    ok = actual == item.get("oid")
                    expected = item.get("oid")
                revision_checked.append({"file": rel, "ok": ok})
                if not ok:
                    problems.append(f"revision hash mismatch for {rel}: local {actual[:16]} != remote {expected[:16]}")
        except Exception as error:  # noqa: BLE001
            problems.append(f"revision check unavailable: {error}")

    receipt = {
        "schema": "glm52-sqg-w4a8-model-codec-validation-v1",
        "model": str(root),
        "revision": args.revision,
        "tensors_indexed": len(weight_map),
        "tensors_on_disk": len(tensors_on_disk),
        "shards": len(shards),
        "routed_layers": sorted(per_layer),
        "routed_layer_count": len(per_layer),
        "coupled_layers": sorted(coupled_layers),
        "per_layer_census_ok": all(
            per_layer[layer][3]
            == (
                census[str(layer)]["k3"]
                if coupled and layer in coupled_layers and isinstance(census, dict)
                else 384
            )
            and per_layer[layer][4]
            == (
                census[str(layer)]["k4"]
                if coupled and layer in coupled_layers and isinstance(census, dict)
                else 384
            )
            for layer in per_layer
        ),
        "projection_split_example_layer3": {
            proj: dict(counts) for proj, counts in per_layer_proj[3].items()
        },
        "dense_rate_census": dict(dense_bits),
        "mtp_layer78_routed": per_layer.get(78, Counter())[3] + per_layer.get(78, Counter())[4],
        "mtp_layer78_preserved": (
            coupled_layers == target_routed_layers
            and per_layer.get(78, Counter())[3] == 384
            and per_layer.get(78, Counter())[4] == 384
        ),
        "revision_checked": revision_checked,
        "problems": problems,
        "pass": not problems,
    }
    emit(problems, receipt, args)


def emit(problems: list[str], receipt: dict, args) -> None:
    if args.result_json and receipt:
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2) + "\n")
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({k: v for k, v in receipt.items() if k != "problems"}, indent=2))


if __name__ == "__main__":
    main()
