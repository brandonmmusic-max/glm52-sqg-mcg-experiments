"""Sealed TP-independent GLM-5.2 SQG atoms-v2 profile.

The checkpoint stays tensor oriented: every logical gate/up/down tensor keeps
its independently selected K3 or K4 rate and the topology-neutral shared sides
are stored once per layer.  Device/rank-specific atom packing is deliberately
left to the runtime's disposable prepared cache, so neither TP nor DCP
ownership can leak into the durable checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .fresh_pipeline_artifacts import (
    _check_json_seal,
    _seal_json,
    shared_down_name,
    shared_gate_up_name,
    validate_layer_artifact,
)
from .fresh_pipeline_common import (
    EXPECTED_BIT_UNITS_PER_LAYER,
    EXPECTED_K3_PER_LAYER,
    EXPECTED_K4_PER_LAYER,
    NUM_EXPERTS,
    PROJECTIONS,
    atomic_json,
    canonical_sha256,
    load_json_object,
    sha256_file,
    tensor_prefix,
    validate_layer,
)


SCHEMA = "glm52-sqg-atoms-v2-profile-v1"
VERSION = 2
STORAGE_FORMAT = "glm52_sqg_atoms_v2"
PROFILE = "mixed_tensor_k3_k4_topology_neutral"
ENCODING = "sqg_e4m3"
CODEBOOK = "sqg_xor_cheb_t12"


def profile_filename(layer: int) -> str:
    return f"glm52-sqg-atoms-v2-layer-{validate_layer(layer):03d}.json"


def _rate_map(layer: int, artifact: dict[str, Any]) -> dict[str, object]:
    assignments = dict(sorted(artifact["bit_map"].items()))
    expected = {
        tensor_prefix(layer, expert, projection)
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    }
    values = tuple(assignments.values())
    if set(assignments) != expected:
        raise ValueError("GLM atoms-v2 rate-map domain differs")
    if (
        len(values),
        values.count(3),
        values.count(4),
        sum(values),
    ) != (
        NUM_EXPERTS * len(PROJECTIONS),
        EXPECTED_K3_PER_LAYER,
        EXPECTED_K4_PER_LAYER,
        EXPECTED_BIT_UNITS_PER_LAYER,
    ):
        raise ValueError("GLM atoms-v2 exact mixed-rate budget differs")
    return {
        "scope": "logical_tensor",
        "allowed_rates": [3, 4],
        "uniform_rate": False,
        "assignments_preserved_verbatim": True,
        "assignments": assignments,
        "assignments_sha256": canonical_sha256(assignments),
        "histogram": {"3": values.count(3), "4": values.count(4)},
        "bit_units": sum(values),
    }


def _topology_contract(layer: int) -> dict[str, object]:
    return {
        "tensor_parallel_independent": True,
        "rank_ownership_serialized": False,
        "prepared_cache_owned_by_runtime": True,
        "stored_labels_reach_e4m3_mma_unchanged": True,
        "transforms_folded_into_weight_labels": False,
        "gate_up": {
            "input_suh": {
                "scope": "layer_shared",
                "tensor": shared_gate_up_name(layer),
                "application": "activation_before_a8_quantization",
            },
            "output_svh": {"scope": "logical_tensor", "application": "fp32_epilogue"},
        },
        "down": {
            "input_suh": {
                "scope": "logical_tensor",
                "application": "activation_before_a8_quantization",
            },
            "output_svh": {
                "scope": "layer_shared",
                "tensor": shared_down_name(layer),
                "application": "fp32_epilogue",
            },
        },
    }


def materialize_glm52_atoms_v2_profile(
    layer_manifest_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind a validated GLM layer artifact to the atoms-v2 runtime contract."""

    source_path = Path(layer_manifest_path).resolve()
    artifact = validate_layer_artifact(source_path)
    layer = validate_layer(int(artifact["layer"]))
    destination = (
        source_path.parent / profile_filename(layer)
        if output_path is None
        else Path(output_path).resolve()
    )
    if destination.parent != source_path.parent or destination.name != profile_filename(layer):
        raise ValueError("GLM atoms-v2 profile must be a canonical layer-artifact sibling")
    shard_path = source_path.parent / str(artifact["shard"])
    rate_map = _rate_map(layer, artifact)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "complete": True,
        "storage_format": STORAGE_FORMAT,
        "profile": PROFILE,
        "encoding": ENCODING,
        "codebook": CODEBOOK,
        "layer": layer,
        "source": {
            "layer_manifest": source_path.name,
            "layer_manifest_sha256": sha256_file(source_path),
            "layer_shard": shard_path.name,
            "layer_shard_sha256": artifact["shard_sha256"],
            "layer_shard_bytes": shard_path.stat().st_size,
        },
        "rate_map": rate_map,
        "topology": _topology_contract(layer),
        "shared_profiles": artifact["shared_profiles"],
        "runtime_contract": {
            "checkpoint_layout": "logical_tensor",
            "prepared_weight_cache": "device_specific_atom_major",
            "prepared_weight_cache_is_checkpoint": False,
            "supported_checkpoint_rates": [3, 4],
            "activation_equation": "silu(gate)*up",
            "weight_endpoint": "native_e4m3_labels",
            "a16_supported": True,
            "w4a8_requires_runtime_qualification": True,
        },
        "lineage": {
            "run_id": artifact["run_id"],
            "sqg_tensor_count": NUM_EXPERTS * len(PROJECTIONS),
            "mcg_tensor_count": 0,
            "independent_per_tensor_rates": True,
            "topology_neutral_scales_and_transforms": True,
        },
    }
    manifest["manifest_id"] = canonical_sha256(manifest)
    atomic_json(destination, manifest)
    _seal_json(destination)
    return validate_glm52_atoms_v2_profile(destination)


def validate_glm52_atoms_v2_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path).resolve()
    _check_json_seal(profile_path)
    manifest = load_json_object(profile_path)
    manifest_id = manifest.get("manifest_id")
    without_id = dict(manifest)
    without_id.pop("manifest_id", None)
    if manifest_id != canonical_sha256(without_id):
        raise ValueError("GLM atoms-v2 manifest identity differs")
    expected_header = {
        "schema": SCHEMA,
        "version": VERSION,
        "complete": True,
        "storage_format": STORAGE_FORMAT,
        "profile": PROFILE,
        "encoding": ENCODING,
        "codebook": CODEBOOK,
    }
    for name, expected in expected_header.items():
        if manifest.get(name) != expected:
            raise ValueError(f"GLM atoms-v2 {name} differs")
    layer = validate_layer(int(manifest.get("layer", -1)))
    if profile_path.name != profile_filename(layer):
        raise ValueError("GLM atoms-v2 profile filename differs")
    source = manifest.get("source", {})
    expected_source_name = f"fresh-sqg-layer-{layer:03d}.json"
    if source.get("layer_manifest") != expected_source_name:
        raise ValueError("GLM atoms-v2 source layer-manifest name differs")
    layer_path = profile_path.parent / expected_source_name
    if sha256_file(layer_path) != source.get("layer_manifest_sha256"):
        raise ValueError("GLM atoms-v2 source layer-manifest hash differs")
    artifact = validate_layer_artifact(layer_path)
    shard_path = profile_path.parent / str(artifact["shard"])
    if (
        source.get("layer_shard") != shard_path.name
        or source.get("layer_shard_sha256") != artifact["shard_sha256"]
        or source.get("layer_shard_bytes") != shard_path.stat().st_size
    ):
        raise ValueError("GLM atoms-v2 source layer-shard binding differs")
    if manifest.get("rate_map") != _rate_map(layer, artifact):
        raise ValueError("GLM atoms-v2 mixed per-tensor rate map differs")
    if manifest.get("topology") != _topology_contract(layer):
        raise ValueError("GLM atoms-v2 topology-neutral transform contract differs")
    if manifest.get("shared_profiles") != artifact["shared_profiles"]:
        raise ValueError("GLM atoms-v2 shared profile binding differs")
    expected_runtime = {
        "checkpoint_layout": "logical_tensor",
        "prepared_weight_cache": "device_specific_atom_major",
        "prepared_weight_cache_is_checkpoint": False,
        "supported_checkpoint_rates": [3, 4],
        "activation_equation": "silu(gate)*up",
        "weight_endpoint": "native_e4m3_labels",
        "a16_supported": True,
        "w4a8_requires_runtime_qualification": True,
    }
    if manifest.get("runtime_contract") != expected_runtime:
        raise ValueError("GLM atoms-v2 runtime contract differs")
    if manifest.get("lineage") != {
        "run_id": artifact["run_id"],
        "sqg_tensor_count": NUM_EXPERTS * len(PROJECTIONS),
        "mcg_tensor_count": 0,
        "independent_per_tensor_rates": True,
        "topology_neutral_scales_and_transforms": True,
    }:
        raise ValueError("GLM atoms-v2 lineage differs")
    return manifest


__all__ = [
    "CODEBOOK",
    "ENCODING",
    "PROFILE",
    "SCHEMA",
    "STORAGE_FORMAT",
    "VERSION",
    "materialize_glm52_atoms_v2_profile",
    "profile_filename",
    "validate_glm52_atoms_v2_profile",
]
