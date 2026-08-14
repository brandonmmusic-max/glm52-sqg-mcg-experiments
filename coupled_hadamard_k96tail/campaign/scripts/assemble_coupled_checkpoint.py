#!/usr/bin/env python3
"""Assemble a runnable coupled checkpoint from immutable selected layer shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROUTED_LAYERS = tuple(range(3, 79))
NUM_EXPERTS = 256
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ASSEMBLY_SCHEMA = "glm52-sqg-coupled-h512-h128-mixed-rate-checkpoint-v3"
HESSIAN_DATASET = "brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians"
HESSIAN_REVISION = "a05b3b92d749f6a641af5cfd52de2b4720380dfd"
SOURCE_CHECKPOINT = "/home/brandonmusic/models/GLM-5.2-SQG-W4A8"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--layer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    return parser


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not a map: {path}")
    return value


def _link(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    os.link(source, destination, follow_symlinks=False)


def main() -> int:
    args = _parser().parse_args()
    from bmmlaw_r7_encoder.safetensors_io import SafeTensorReader
    from src.fresh_pipeline_common import atomic_json, canonical_sha256, sha256_file

    source = args.source.resolve()
    layer_root = args.layer_root.resolve()
    destination = args.output.resolve()
    layers = tuple(sorted(set(args.layers)))
    if (
        not layers
        or len(layers) != len(args.layers)
        or any(layer not in ROUTED_LAYERS for layer in layers)
    ):
        raise ValueError("coupled layers must be unique members of 3..78")
    if destination.exists():
        raise FileExistsError(
            f"refusing existing checkpoint output: {destination}"
        )
    output = destination.with_name(
        f".{destination.name}.staging-{os.getpid()}"
    )
    if output.exists():
        raise FileExistsError(f"refusing existing assembly staging path: {output}")
    for root in (source, layer_root):
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"unsafe checkpoint root: {root}")

    manifests: dict[int, dict[str, Any]] = {}
    qualities: dict[int, dict[str, Any]] = {}
    readers: dict[int, Any] = {}
    replacement_names: set[str] = set()
    for layer in layers:
        stem = f"r7-experts-layer-{layer:03d}"
        shard = layer_root / f"{stem}.safetensors"
        manifest_path = layer_root / f"{stem}.json"
        quality_path = layer_root / f"{stem}.quality.json"
        manifest = _load(manifest_path)
        quality = _load(quality_path)
        census = manifest.get("bit_census")
        valid_census = isinstance(census, dict) and census in (
            {"k3": 720, "k4": 48, "total": 768},
            {"k3": 672, "k4": 96, "total": 768},
        )
        expected_bpw = (
            (3 * census["k3"] + 4 * census["k4"]) / 768.0
            if valid_census
            else None
        )
        manifest_schema = manifest.get("schema")
        quality_schema = quality.get("schema")
        legacy_layer3 = (
            layer == 3
            and manifest_schema == "glm52-coupled-selected-layer-runtime-v1"
            and quality_schema == "glm52-coupled-layer-quality-tails-v1"
        )
        no_shortcut_layer = (
            manifest_schema == "glm52-coupled-selected-layer-runtime-v3"
            and quality_schema == "glm52-coupled-layer-quality-tails-v2"
        )
        valid_manifest_schema = legacy_layer3 or no_shortcut_layer
        if (
            not valid_manifest_schema
            or manifest.get("complete") is not True
            or manifest.get("layer") != layer
            or not valid_census
            or manifest.get("bits_per_weight") != expected_bpw
            or sha256_file(shard) != manifest.get("shard_sha256")
            or quality.get("complete") is not True
            or quality.get("layer") != layer
            or quality.get("selection_id") != manifest.get("selection_id")
            or (
                no_shortcut_layer
                and (
                    quality.get("allocation_binding") != manifest.get("allocation")
                    or quality.get("final_profile_binding")
                    != manifest.get("final_profile_binding")
                    or not isinstance(manifest.get("final_profile_binding"), dict)
                    or manifest["final_profile_binding"].get(
                        "no_b300_owner_speed_rescue"
                    )
                    is not True
                )
            )
        ):
            raise ValueError(f"coupled layer seal differs: {layer}")
        reader = SafeTensorReader(shard)
        if len(reader.tensors) != 2306:
            raise ValueError(f"coupled layer {layer} tensor count differs")
        manifests[layer] = manifest
        qualities[layer] = quality
        readers[layer] = reader
        replacement_names.update(
            (f"{stem}.safetensors", f"{stem}.json", f"{stem}.quality.json")
        )

    output.mkdir(parents=False)
    try:
        generated = {
            "config.json",
            "quantization_config.json",
            "model.safetensors.index.json",
            "COUPLED_REENCODE_MANIFEST.json",
            "README.md",
            "RELEASE_PROVENANCE.json",
            "SHA256SUMS",
            "FULL_SQG_NATIVE_MANIFEST.json",
            "FULL_SQG_NATIVE_ACCEPTANCE.json",
            "SQG_REPRODUCIBILITY_CAMPAIGN.json",
            "HESSIAN_DATASET_LAYOUT.md",
            "compose.yaml",
            "SQG_REPRODUCIBILITY_SOURCE.tar.gz",
            "SQG_REPRODUCIBILITY_BUILD_BINDING.json",
            "SQG_REPRODUCIBILITY_SHA256SUMS",
            "SQG_RUNTIME_OVERLAY.tar.gz",
            "SQG_RUNTIME_OVERLAY_SHA256SUMS",
            "REPRODUCTION_SOURCE_SHA256.json",
            "REPRODUCE_SQG_W4A8.md",
            "REPRO_hessian_dataset_upload.py",
            "REPRO_materialize_full_sqg_checkpoint.py",
            "REPRO_public_release.py",
            "REPRO_validate_full_sqg_checkpoint.py",
        }
        for path in source.iterdir():
            if (
                path.name in generated
                or path.name in replacement_names
                or not path.is_file()
                or path.is_symlink()
            ):
                continue
            _link(path, output / path.name)
        source_records = {
            "SHA256SUMS": "SOURCE_SHA256SUMS",
            "RELEASE_PROVENANCE.json": "SOURCE_RELEASE_PROVENANCE.json",
            "FULL_SQG_NATIVE_MANIFEST.json": "SOURCE_FULL_SQG_NATIVE_MANIFEST.json",
            "FULL_SQG_NATIVE_ACCEPTANCE.json": "SOURCE_FULL_SQG_NATIVE_ACCEPTANCE.json",
            "SQG_REPRODUCIBILITY_CAMPAIGN.json": "SOURCE_SQG_REPRODUCIBILITY_CAMPAIGN.json",
            "HESSIAN_DATASET_LAYOUT.md": "SOURCE_HESSIAN_DATASET_LAYOUT.md",
            "compose.yaml": "SOURCE_compose.yaml",
            "SQG_REPRODUCIBILITY_SOURCE.tar.gz": "SOURCE_SQG_REPRODUCIBILITY_SOURCE.tar.gz",
            "SQG_REPRODUCIBILITY_BUILD_BINDING.json": "SOURCE_SQG_REPRODUCIBILITY_BUILD_BINDING.json",
            "SQG_REPRODUCIBILITY_SHA256SUMS": "SOURCE_SQG_REPRODUCIBILITY_SHA256SUMS",
            "SQG_RUNTIME_OVERLAY.tar.gz": "SOURCE_SQG_RUNTIME_OVERLAY.tar.gz",
            "SQG_RUNTIME_OVERLAY_SHA256SUMS": "SOURCE_SQG_RUNTIME_OVERLAY_SHA256SUMS",
            "REPRODUCTION_SOURCE_SHA256.json": "SOURCE_REPRODUCTION_SOURCE_SHA256.json",
            "REPRODUCE_SQG_W4A8.md": "SOURCE_REPRODUCE_SQG_W4A8.md",
            "REPRO_hessian_dataset_upload.py": "SOURCE_REPRO_hessian_dataset_upload.py",
            "REPRO_materialize_full_sqg_checkpoint.py": "SOURCE_REPRO_materialize_full_sqg_checkpoint.py",
            "REPRO_public_release.py": "SOURCE_REPRO_public_release.py",
            "REPRO_validate_full_sqg_checkpoint.py": "SOURCE_REPRO_validate_full_sqg_checkpoint.py",
        }
        for source_name, destination_name in source_records.items():
            source_path = source / source_name
            if source_path.is_file() and not source_path.is_symlink():
                _link(source_path, output / destination_name)
        for layer in layers:
            stem = f"r7-experts-layer-{layer:03d}"
            _link(layer_root / f"{stem}.safetensors", output / f"{stem}.safetensors")
            _link(layer_root / f"{stem}.json", output / f"{stem}.json")
            _link(layer_root / f"{stem}.quality.json", output / f"{stem}.quality.json")

        quant = _load(source / "quantization_config.json")
        contract = quant["glm_sqg_w4a8"]
        layer_censuses = {layer: manifests[layer]["bit_census"] for layer in layers}
        all_k48 = all(
            census == {"k3": 720, "k4": 48, "total": 768}
            for census in layer_censuses.values()
        )
        hybrid_k96tail = (
            layers == ROUTED_LAYERS
            and layer_censuses[3] == {"k3": 720, "k4": 48, "total": 768}
            and all(
                layer_censuses[layer] == {"k3": 672, "k4": 96, "total": 768}
                for layer in range(4, 79)
            )
        )
        if all_k48:
            rate_policy = {
                "name": "no_shortcut_layer_native_k48_tail_v1",
                "uniform_rate": False,
                "all_routed_layers": {
                    "k3": 720,
                    "k4": 48,
                    "bits_per_weight": 3.0625,
                },
                "k4_assignment": "independent_per_tensor_gate_up_down",
                "tail_signal": "fit_allocation_worst_2pct_routed_positions",
                "tail_signal_role": "fit_allocation_only_not_model_acceptance",
                "calibration_total_regression_limit": 0.01,
                "calibration_body_regression_limit": 0.01,
                "per_layer_beta_profile_recipe": True,
                "owner_fixed_beta_bypassed": True,
            }
        elif hybrid_k96tail:
            rate_policy = {
                "name": "no_shortcut_layer_native_k96_source_worst40_guarded_v1",
                "uniform_rate": False,
                "layer_3": {"k3": 720, "k4": 48, "bits_per_weight": 3.0625},
                "layers_4_through_78": {
                    "k3": 672,
                    "k4": 96,
                    "bits_per_weight": 3.125,
                },
                "k4_assignment": "independent_per_tensor_gate_up_down",
                "tail_signal": "sealed_source_tp4dcp1_worst40_exact_routes",
                "tail_signal_role": "in_sample_allocation_fit_only_not_acceptance",
                "layer_78_route_signal": "unavailable_use_layer_native_k96",
                "calibration_total_regression_limit": 0.01,
                "calibration_body_regression_limit": 0.01,
                "per_layer_beta_profile_recipe_layers_4_through_78": True,
                "owner_fixed_beta_bypassed": True,
                "sealed_layer_3_preserved": True,
            }
        else:
            rate_policy = {
                "name": "per_layer_mixed_coupled_rate_v1",
                "uniform_rate": False,
                "per_layer_bit_census": {
                    str(layer): layer_censuses[layer] for layer in layers
                },
                "k4_assignment": "independent_per_tensor_gate_up_down",
            }
        contract.update(
            {
                "schema": "glm52_sqg_atoms_v2_coupled_h512_h128_w4a8_v1",
                "execution": "full_coupled_w4a8",
                "coupled_layers": list(layers),
                "coupled_rotation_draws": {
                    str(layer): [
                        int(manifests[layer]["selected_draws"][str(expert)])
                        for expert in range(NUM_EXPERTS)
                    ]
                    for layer in layers
                },
                "coupled_transform": {
                    "residual_policy": "coupled_block_hadamard",
                    "residual_block_size": 512,
                    "residual_draw": 0,
                    "preactivation_block_size": 128,
                    "postactivation_block_size": 128,
                    "activation": "silu",
                    "activation_equation": "silu(gate)*up",
                    "activation_boundary": "native_direct_e4m3_w4a8",
                    "h13_local_alpha": 0.25,
                    "candidate_conditioned_down_hessian": True,
                    "draw_candidates": [0, 6],
                    "selection_rule": "fit_may_propose_draw_6_disjoint_selection_must_confirm",
                    "holdout_used_for_selection": False,
                },
                "rate_policy": rate_policy,
                "codec_execution": {
                    "weight_path": "route_packed_direct_e4m3_w4a8",
                    "activation_dtype": "e4m3",
                    "a16_fallback_allowed": False,
                    "topology_neutral_construction": True,
                    "mtp_layer_78_included": 78 in layers,
                },
                "calibration_provenance": {
                    "dataset": HESSIAN_DATASET,
                    "revision": HESSIAN_REVISION,
                    "source_checkpoint": SOURCE_CHECKPOINT,
                    "original_bf16_downloaded": False,
                },
                "per_layer_bit_census": {
                    str(layer): manifests[layer]["bit_census"]
                    for layer in layers
                },
                "per_layer_selected_beta": {
                    str(layer): manifests[layer].get("selected_beta")
                    for layer in layers
                },
                "per_layer_final_profile_binding": {
                    str(layer): manifests[layer].get("final_profile_binding")
                    for layer in layers
                },
            }
        )
        routed = quant["r7_routed_experts"]
        routed.pop("per_layer_k3", None)
        routed.pop("per_layer_k4", None)
        routed["per_layer_bit_census"] = contract["per_layer_bit_census"]
        routed["coupled_layers"] = list(layers)
        routed["coupled_rotation_layout"] = "updated_qsrt_h512_h128_h128_silu_v1"

        dtype_names = {
            "I16": "torch.int16",
            "I32": "torch.int32",
            "F16": "torch.float16",
        }
        storage = quant["tensor_storage"]
        for layer in layers:
            reader = readers[layer]
            for expert in range(NUM_EXPERTS):
                for projection in PROJECTIONS:
                    prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                    entry = storage[prefix]
                    tensor_names = sorted(
                        name for name in reader.tensors if name.startswith(f"{prefix}.")
                    )
                    expected_suffixes = (
                        {"sqg", "svh", "trellis"}
                        if projection != "down_proj"
                        else {"sqg", "suh", "trellis"}
                    )
                    if {name.rsplit(".", 1)[-1] for name in tensor_names} != expected_suffixes:
                        raise ValueError(f"coupled runtime tensor domain differs: {prefix}")
                    bits = int(reader.tensors[f"{prefix}.trellis"].shape[2]) // 16
                    entry["bits_per_weight"] = bits
                    entry["stored_tensors"] = {
                        name: {
                            "dtype": dtype_names[reader.tensors[name].dtype],
                            "shape": list(reader.tensors[name].shape),
                            "n_bytes": reader.tensors[name].nbytes,
                        }
                        for name in tensor_names
                    }

        atomic_json(output / "quantization_config.json", quant)
        config = _load(source / "config.json")
        config["quantization_config"] = quant
        atomic_json(output / "config.json", config)

        index = _load(source / "model.safetensors.index.json")
        old_total = int(index["metadata"]["total_size"])
        delta = sum(
            int(manifests[layer]["shard_bytes"])
            - (source / f"r7-experts-layer-{layer:03d}.safetensors").stat().st_size
            for layer in layers
        )
        index["metadata"]["total_size"] = old_total + delta
        atomic_json(output / "model.safetensors.index.json", index)

        assembly: dict[str, Any] = {
            "schema": ASSEMBLY_SCHEMA,
            "complete": True,
            "source_model": str(source),
            "source_index_sha256": sha256_file(source / "model.safetensors.index.json"),
            "coupled_layers": list(layers),
            "all_routed_layers_coupled": layers == ROUTED_LAYERS,
            "routed_bits_per_weight": (
                sum(manifests[layer]["bits_per_weight"] for layer in layers)
                / len(layers)
            ),
            "routed_layer_average_rate_is_uniform": len(
                {manifests[layer]["bits_per_weight"] for layer in layers}
            )
            == 1,
            "routed_tensor_rates_are_mixed": True,
            "rate_policy": contract["rate_policy"],
            "codec_execution": contract["codec_execution"],
            "calibration_provenance": contract["calibration_provenance"],
            "per_layer_bits_per_weight": {
                str(layer): manifests[layer]["bits_per_weight"] for layer in layers
            },
            "per_layer_bit_census": {
                str(layer): manifests[layer]["bit_census"] for layer in layers
            },
            "coupled_transform": contract["coupled_transform"],
            "layers": {
                str(layer): {
                    "manifest_id": manifests[layer]["manifest_id"],
                    "shard_sha256": manifests[layer]["shard_sha256"],
                    "quality_archive_id": qualities[layer]["archive_id"],
                    "selection_id": manifests[layer]["selection_id"],
                    "draw_histogram": manifests[layer]["draw_histogram"],
                    "allocation": manifests[layer].get("allocation"),
                    "selected_beta": manifests[layer].get("selected_beta"),
                    "final_profile_binding": manifests[layer].get(
                        "final_profile_binding"
                    ),
                }
                for layer in layers
            },
            "index_sha256": sha256_file(output / "model.safetensors.index.json"),
            "quantization_config_sha256": sha256_file(
                output / "quantization_config.json"
            ),
        }
        assembly["manifest_id"] = canonical_sha256(assembly)
        atomic_json(output / "COUPLED_REENCODE_MANIFEST.json", assembly)
        atomic_json(
            output / "RELEASE_PROVENANCE.json",
            {
                "schema": (
                    "glm52-coupled-no-shortcut-k96tail-provenance-v1"
                    if hybrid_k96tail
                    else "glm52-coupled-no-shortcut-3p0625-provenance-v1"
                ),
                "assembly_complete": True,
                "end_to_end_kld_complete": False,
                "model_repo": None,
                "local_model": str(destination),
                "assembly_manifest_id": assembly["manifest_id"],
                "source_checkpoint": SOURCE_CHECKPOINT,
                "source_index_sha256": assembly["source_index_sha256"],
                "original_bf16_downloaded_for_reencode": False,
                "hessian_dataset": HESSIAN_DATASET,
                "hessian_revision": HESSIAN_REVISION,
            },
        )
        rate_summary = (
            "- Layer 3 is the sealed K48 exception: 720 K3 + 48 K4 tensors "
            "(3.0625 bpw).\n"
            "- Layers 4-78 use K96: 672 K3 + 96 K4 tensors (3.125 bpw)."
            if hybrid_k96tail
            else "- Every routed layer has 720 K3 + 48 K4 tensors "
            "(3.0625 bpw exactly)."
        )
        readme = f"""---
license: other
base_model: zai-org/GLM-5.2
library_name: vllm
tags:
- glm
- sqg
- w4a8
- coupled-hadamard
- blackwell
---

# GLM-5.2 SQG coupled H512/H128 mixed-rate re-encode

This is a distinct, non-overwriting re-encode of `{SOURCE_CHECKPOINT}` using
the frozen checkpoint as its weight source. The original BF16 model was not
downloaded for this re-encode.

- Routed layers 3-78 are coupled, including MTP layer 78.
{rate_summary}
- Actual routed-layer average: {assembly['routed_bits_per_weight']:.12f} bpw.
- Coordinates: residual H512, preactivation H128, postactivation H128.
- Activation: exact `silu(gate)*up`; H13 local alpha is 0.25.
- Execution: route-packed direct-E4M3 full W4A8 with no A16 fallback.
- Layers 4-78 use their own full profile search and seven-beta selection; the
  B300 owner-fixed-beta/identity-only rescue is explicitly bypassed. Layer 3
  is the preserved sealed exception.
- Calibration: `{HESSIAN_DATASET}` at `{HESSIAN_REVISION}`.

Rate allocation uses only fit/calibration and fit/allocation evidence. Full-model
TP4/PP1/DCP1 KLD against sealed BF16 logits is required before release and is
pending in this preliminary assembly record.
"""
        readme_path = output / "README.md"
        temporary_readme = output / f".README.md.tmp-{os.getpid()}"
        with temporary_readme.open("x", encoding="utf-8") as handle:
            handle.write(readme)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_readme, readme_path)
        hessian_layout = (
            "# Calibration and Hessian provenance\n\n"
            f"Dataset: `{HESSIAN_DATASET}`\n\n"
            f"Pinned revision: `{HESSIAN_REVISION}`\n\n"
            "The re-encode uses the canonical capture/profile/Hessian view and "
            "does not download the original BF16 checkpoint.\n"
        )
        hessian_path = output / "HESSIAN_DATASET_LAYOUT.md"
        temporary_hessian = output / f".HESSIAN_DATASET_LAYOUT.md.tmp-{os.getpid()}"
        with temporary_hessian.open("x", encoding="utf-8") as handle:
            handle.write(hessian_layout)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_hessian, hessian_path)
    except BaseException:
        # Keep a failed staging construction visible for inspection; the final
        # destination remains absent and retryable, and source is never mutated.
        raise

    os.replace(output, destination)
    print(json.dumps(assembly, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
