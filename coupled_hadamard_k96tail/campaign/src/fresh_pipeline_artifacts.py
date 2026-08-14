"""Atomic expert/layer payloads and independent zero-MCG validators."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch
from safetensors import safe_open

from bmmlaw_r7_encoder.safetensors_io import (
    SafeTensorReader,
    TensorEntry,
    torch_tensor_entry,
    write_safetensors_atomic,
)

from .fresh_pipeline_common import (
    EXPERT_ARTIFACT_SCHEMA,
    EXPECTED_BIT_UNITS_PER_LAYER,
    EXPECTED_K3_PER_LAYER,
    EXPECTED_K4_PER_LAYER,
    LAYER_ARTIFACT_SCHEMA,
    NUM_EXPERTS,
    PROJECTIONS,
    SQG_MARKER,
    assert_no_forbidden_tensor_names,
    atomic_json,
    canonical_sha256,
    load_json_object,
    permutation_scope,
    sha256_file,
    tensor_prefix,
    validate_layer,
)
from .glm52_fresh_sqg import (
    EncodedSQGMatrix,
    FreshExpertPermutation,
    FrozenBitBudget,
    SharedResidualProfile,
    build_run_manifest,
    validate_tensor_manifest,
)
from .glm52_fresh_sqg.reference import (
    decode_stored_fp16,
    tensor_sha256,
)


def atomic_bytes(path: str | Path, payload: bytes, *, overwrite: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        if destination.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace bound artifact: {destination}")
        return
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _seal_json(path: Path) -> str:
    digest = sha256_file(path)
    atomic_bytes(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n".encode())
    return digest


def _check_json_seal(path: Path) -> str:
    if not path.is_file() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError(f"artifact JSON is absent or symlinked: {path}")
    seal = path.with_suffix(path.suffix + ".sha256")
    if not seal.is_file() or seal.is_symlink():
        raise ValueError(f"artifact JSON seal is absent: {seal}")
    fields = seal.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise ValueError(f"artifact JSON seal has invalid syntax: {seal}")
    observed = sha256_file(path)
    if fields[0] != observed:
        raise ValueError(f"artifact JSON seal mismatch: {path}")
    return observed


def _artifact_shard(
    manifest_path: Path,
    declared: object,
    expected_name: str,
    *,
    label: str,
) -> Path:
    """Resolve only the exact regular sibling shard emitted by this pipeline."""

    if declared != expected_name or Path(expected_name).name != expected_name:
        raise ValueError(f"{label} shard path differs")
    shard = manifest_path.parent / expected_name
    if (
        not shard.is_file()
        or shard.is_symlink()
        or shard.parent.is_symlink()
        or shard.resolve() != shard
    ):
        raise ValueError(f"{label} shard is absent, symlinked, or escapes: {shard}")
    return shard


def shared_gate_up_name(layer: int) -> str:
    validate_layer(layer)
    return f"model.layers.{layer}.mlp.experts.r7_shared.gate_up_suh"


def shared_down_name(layer: int) -> str:
    validate_layer(layer)
    return f"model.layers.{layer}.mlp.experts.r7_shared.down_svh"


def expert_stem(layer: int, expert: int) -> str:
    validate_layer(layer)
    if not 0 <= expert < NUM_EXPERTS:
        raise ValueError("expert must lie in [0,255]")
    return f"layer-{layer:03d}-expert-{expert:03d}"


def _projection_role(projection: str) -> str:
    if projection == "gate_proj":
        return "gate"
    if projection == "up_proj":
        return "up"
    if projection == "down_proj":
        return "down"
    raise ValueError(f"unsupported projection {projection!r}")


def _require_final_full_closure(manifest: Mapping[str, object], *, label: str) -> None:
    """Reject deferred sweep candidates from every persisted model artifact."""

    closure = manifest.get("decoded_closure", {})
    encoder = manifest.get("encoder", {})
    transform = manifest.get("transform", {})
    if (
        not isinstance(closure, Mapping)
        or not isinstance(encoder, Mapping)
        or not isinstance(transform, Mapping)
        or encoder.get("production") is not True
        or transform.get("candidate_prepared_hash_deferred") is not False
        or encoder.get("candidate_sweep_fast_requested") is not False
        or encoder.get("full_decode_closure_performed") is not True
        or closure.get("mode") != "selected_full"
        or closure.get("implementation")
        != "independent_pytorch_packed_fp16_v1"
        or closure.get("full_decode_deferred") is not False
        or closure.get("selected_model_eligible") is not True
        or closure.get("packed_states_exact") is not True
        or closure.get("repacked_words_exact") is not True
        or closure.get("direct_e4m3_labels_exact") is not True
        or not isinstance(closure.get("decoded_exl_sha256"), str)
        or not isinstance(closure.get("source_relative_rmse"), (int, float))
        or not isinstance(closure.get("encoder_relative_rmse"), (int, float))
    ):
        raise ValueError(f"{label}: final selected full closure is absent")


def write_expert_artifact(
    output_dir: str | Path,
    *,
    run_id: str,
    layer: int,
    expert: int,
    encoded: Mapping[str, EncodedSQGMatrix],
    permutation: FreshExpertPermutation,
    gate_up_profile: SharedResidualProfile,
    down_profile: SharedResidualProfile,
    bit_map: Mapping[str, int],
    h2_evidence: Mapping[str, object],
    calibration_evidence: Mapping[str, object],
    source_evidence: Mapping[str, object],
    selection_evidence_sha256: str,
    purpose: str = "final_treatment",
) -> dict[str, object]:
    """Publish one resumable all-three-projection SQG mini-shard."""

    layer = validate_layer(layer)
    if set(encoded) != set(PROJECTIONS):
        raise ValueError("expert artifact requires gate/up/down encodings")
    if permutation.scope != permutation_scope(layer, expert):
        raise ValueError("expert artifact permutation scope differs")
    if not permutation.production_qualified:
        raise ValueError("expert artifact requires production h2_reverse permutation")
    if gate_up_profile.side != "input" or down_profile.side != "output":
        raise ValueError("shared residual profile sides differ")
    h2_upstream = h2_evidence.get("upstream_candidate", {})
    if (
        h2_evidence.get("layer") != layer
        or h2_evidence.get("expert") != expert
        or h2_evidence.get("role") != "fit"
        or h2_evidence.get("identity_fallback") is not False
        or h2_upstream.get("physical_permutation_sha256") != permutation.sha256
    ):
        raise ValueError("candidate H2/permutation evidence binding differs")
    tensor_manifests: dict[str, dict[str, object]] = {}
    entries = []
    shared_gate_bytes: torch.Tensor | None = None
    shared_down_bytes: torch.Tensor | None = None
    for projection in PROJECTIONS:
        item = encoded[projection]
        prefix = tensor_prefix(layer, expert, projection)
        if item.manifest.get("tensor_id") != prefix:
            raise ValueError(f"{projection}: codec tensor ID differs")
        if item.manifest.get("matrix_role") != _projection_role(projection):
            raise ValueError(f"{projection}: codec matrix role differs")
        if int(bit_map.get(prefix, -1)) != item.bits:
            raise ValueError(f"{prefix}: frozen bit assignment differs")
        validate_tensor_manifest(item.manifest)
        _require_final_full_closure(item.manifest, label=prefix)
        tensor_manifests[prefix] = dict(item.manifest)
        for name, tensor in item.named_tensors(prefix).items():
            entries.append(torch_tensor_entry(name, tensor))
        if projection in ("gate_proj", "up_proj"):
            expected = gate_up_profile.expected_stored_fp16().cpu()
            if not torch.equal(item.suh, expected):
                raise ValueError(f"{prefix}: gate/up shared suh differs")
            if shared_gate_bytes is None:
                shared_gate_bytes = item.suh
            elif not torch.equal(shared_gate_bytes, item.suh):
                raise ValueError("gate and up shared suh are not byte-identical")
        else:
            expected = down_profile.expected_stored_fp16().cpu()
            if not torch.equal(item.svh, expected):
                raise ValueError(f"{prefix}: down shared svh differs")
            shared_down_bytes = item.svh
    names = [entry.name for entry in entries]
    assert_no_forbidden_tensor_names(names)
    expected_names = {
        f"{tensor_prefix(layer, expert, projection)}.{suffix}"
        for projection in PROJECTIONS
        for suffix in ("trellis", "suh", "svh", "sqg")
    }
    if set(names) != expected_names:
        raise ValueError("expert mini-shard tensor inventory differs")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = expert_stem(layer, expert)
    shard_path = destination / f"{stem}.safetensors"
    manifest_path = destination / f"{stem}.json"
    if shard_path.exists() or manifest_path.exists():
        existing = validate_expert_artifact(manifest_path)
        expected_binding = {
            "run_id": run_id,
            "layer": layer,
            "expert": expert,
            "purpose": purpose,
            "selection_evidence_sha256": selection_evidence_sha256,
        }
        if any(existing.get(key) != value for key, value in expected_binding.items()):
            raise ValueError("existing expert artifact belongs to another run")
        return existing

    payload_hashes, shard_hash = write_safetensors_atomic(
        shard_path,
        entries,
        metadata={
            "format": "pt",
            "schema": EXPERT_ARTIFACT_SCHEMA,
            "run_id": run_id,
            "layer": str(layer),
            "expert": str(expert),
            "codebook": "sqg_xor_cheb_t12",
            "sqg_marker": "0x53514731",
            "legacy_mcg_input": "false",
        },
    )
    manifest: dict[str, object] = {
        "schema": EXPERT_ARTIFACT_SCHEMA,
        "complete": True,
        "purpose": purpose,
        "run_id": run_id,
        "layer": layer,
        "expert": expert,
        "shard": shard_path.name,
        "shard_sha256": shard_hash,
        "payload_sha256": payload_hashes,
        "bits": {
            projection: int(bit_map[tensor_prefix(layer, expert, projection)])
            for projection in PROJECTIONS
        },
        "permutation": permutation.manifest(),
        "shared_profiles": {
            "gate_up_input": gate_up_profile.manifest(),
            "down_output": down_profile.manifest(),
        },
        "shared_vector_sha256": {
            "gate_up_suh": tensor_sha256(shared_gate_bytes),
            "down_svh": tensor_sha256(shared_down_bytes),
        },
        "tensor_manifests": dict(sorted(tensor_manifests.items())),
        "h2": dict(h2_evidence),
        "calibration": dict(calibration_evidence),
        "source": dict(source_evidence),
        "selection_evidence_sha256": selection_evidence_sha256,
        "lineage": {
            "official_bf16": True,
            "fresh_capture": True,
            "frozen_bit_map_only": True,
            "mcg_payload_reads": 0,
            "mcg_transform_reads": 0,
            "mcg_scale_reads": 0,
            "mcg_permutation_reads": 0,
            "mcg_seed_reads": 0,
            "mcg_decoded_weight_reads": 0,
            "legacy_tensor_names": [],
        },
    }
    atomic_json(manifest_path, manifest)
    _seal_json(manifest_path)
    return validate_expert_artifact(manifest_path)


def validate_expert_artifact(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).absolute()
    _check_json_seal(manifest_path)
    manifest = load_json_object(manifest_path)
    if manifest.get("schema") != EXPERT_ARTIFACT_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("expert artifact is absent, incomplete, or foreign")
    layer = validate_layer(int(manifest.get("layer", -1)))
    expert = int(manifest.get("expert", -1))
    if not 0 <= expert < NUM_EXPERTS:
        raise ValueError("expert artifact expert ID differs")
    shard = _artifact_shard(
        manifest_path,
        manifest.get("shard"),
        f"{expert_stem(layer, expert)}.safetensors",
        label="expert artifact",
    )
    if sha256_file(shard) != manifest.get("shard_sha256"):
        raise ValueError("expert artifact shard hash differs")
    reader = SafeTensorReader(shard)
    names = set(reader.tensors)
    assert_no_forbidden_tensor_names(names)
    expected = {
        f"{tensor_prefix(layer, expert, projection)}.{suffix}"
        for projection in PROJECTIONS
        for suffix in ("trellis", "suh", "svh", "sqg")
    }
    if names != expected:
        raise ValueError("expert artifact tensor inventory differs")
    if set(manifest.get("payload_sha256", {})) != expected:
        raise ValueError("expert artifact payload manifest domain differs")
    for name, info in reader.tensors.items():
        if info.payload.sha256() != manifest["payload_sha256"][name]:
            raise ValueError(f"expert artifact payload hash differs: {name}")
        suffix = name.rsplit(".", 1)[-1]
        if suffix == "sqg":
            if info.dtype != "I32" or info.shape != ():
                raise ValueError(f"{name}: SQG marker geometry differs")
    tensor_manifests = manifest.get("tensor_manifests", {})
    if set(tensor_manifests) != {
        tensor_prefix(layer, expert, projection) for projection in PROJECTIONS
    }:
        raise ValueError("expert tensor-manifest domain differs")
    bits = manifest.get("bits", {})
    if set(bits) != set(PROJECTIONS) or any(
        type(value) is not int or value not in (3, 4) for value in bits.values()
    ):
        raise ValueError("expert bit assignments differ")
    permutation = manifest.get("permutation")
    shared_profiles = manifest.get("shared_profiles", {})
    if not isinstance(permutation, Mapping) or set(shared_profiles) != {
        "gate_up_input",
        "down_output",
    }:
        raise ValueError("expert permutation/shared-profile evidence differs")
    source_evidence = manifest.get("source", {})
    expected_source_fields = {
        "tensor_payload_sha256",
        "shard_names",
        "shard_sha256",
    }
    if expected_source_fields - set(source_evidence):
        raise ValueError("expert source evidence is incomplete")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        for projection in PROJECTIONS:
            prefix = tensor_prefix(layer, expert, projection)
            item = tensor_manifests[prefix]
            validate_tensor_manifest(item)
            _require_final_full_closure(item, label=prefix)
            expected_role = _projection_role(projection)
            expected_profile = (
                shared_profiles["gate_up_input"]
                if projection in ("gate_proj", "up_proj")
                else shared_profiles["down_output"]
            )
            source = item.get("source", {})
            transform = item.get("transform", {})
            if (
                item.get("tensor_id") != prefix
                or item.get("matrix_role") != expected_role
                or item.get("bits") != bits[projection]
                or source.get("tensor_name") != f"{prefix}.weight"
                or source.get("tensor_payload_sha256")
                != source_evidence["tensor_payload_sha256"].get(projection)
                or source.get("shard_name")
                != source_evidence["shard_names"].get(projection)
                or source.get("shard_sha256")
                != source_evidence["shard_sha256"].get(projection)
                or transform.get("physical_permutation") != permutation
                or transform.get("shared_residual_profile") != expected_profile
            ):
                raise ValueError(f"{prefix}: nested expert lineage differs")
            trellis_name = f"{prefix}.trellis"
            suh_name = f"{prefix}.suh"
            svh_name = f"{prefix}.svh"
            marker_name = f"{prefix}.sqg"
            if (
                reader.tensors[trellis_name].payload.sha256()
                != item.get("packed_trellis_payload_sha256")
                or tensor_sha256(handle.get_tensor(suh_name))
                != item["scales"]["suh_sha256"]
                or tensor_sha256(handle.get_tensor(svh_name))
                != item["scales"]["svh_sha256"]
                or int(handle.get_tensor(marker_name)) != SQG_MARKER
            ):
                raise ValueError(f"{prefix}: shard/tensor-manifest binding differs")
    gate_id = tensor_prefix(layer, expert, "gate_proj")
    up_id = tensor_prefix(layer, expert, "up_proj")
    down_id = tensor_prefix(layer, expert, "down_proj")
    gate_manifest = tensor_manifests[gate_id]
    up_manifest = tensor_manifests[up_id]
    down_manifest = tensor_manifests[down_id]
    h13_fields = (
        "evidence_id",
        "construction",
        "split_id",
        "matrix_sha256",
        "normalization_count",
        "sigma_reg",
    )
    if any(
        gate_manifest["dense_h"].get(field) != up_manifest["dense_h"].get(field)
        for field in h13_fields
    ):
        raise ValueError("expert gate/up H13 evidence differs")
    gate_session = gate_manifest["dense_h"].get("session", {})
    up_session = up_manifest["dense_h"].get("session", {})
    session_fields = (
        "session_id",
        "hessian_sha256",
        "device",
        "sigma_reg",
        "input_transform_binding",
        "factorization_sha256",
        "caller_owned",
        "module_global",
    )
    gate_ordinal = gate_session.get("use_ordinal")
    up_ordinal = up_session.get("use_ordinal")
    if (
        any(gate_session.get(field) != up_session.get(field) for field in session_fields)
        or not isinstance(gate_ordinal, int)
        or not isinstance(up_ordinal, int)
        or up_ordinal != gate_ordinal + 1
    ):
        raise ValueError("expert gate/up reusable H13 session evidence differs")
    h2 = manifest.get("h2", {})
    upstream = h2.get("upstream_candidate", {})
    if (
        h2.get("layer") != layer
        or h2.get("expert") != expert
        or h2.get("role") != "fit"
        or h2.get("identity_fallback") is not False
        or h2.get("pooled_expert_basis") is not False
        or h2.get("evidence_id") != down_manifest["dense_h"].get("evidence_id")
        or h2.get("matrix_sha256")
        != down_manifest["dense_h"].get("matrix_sha256")
        or upstream.get("gate_tensor_manifest_sha256")
        != canonical_sha256(gate_manifest)
        or upstream.get("up_tensor_manifest_sha256")
        != canonical_sha256(up_manifest)
        or upstream.get("gate_decoded_exl_sha256")
        != gate_manifest["decoded_closure"].get("decoded_exl_sha256")
        or upstream.get("up_decoded_exl_sha256")
        != up_manifest["decoded_closure"].get("decoded_exl_sha256")
        or upstream.get("physical_permutation_sha256")
        != permutation.get("new_to_old_sha256")
    ):
        raise ValueError("expert conditional H2 lineage differs")
    calibration = manifest.get("calibration", {})
    if (
        calibration.get("h13_evidence_id")
        != gate_manifest["dense_h"].get("evidence_id")
        or calibration.get("h13_matrix_sha256")
        != gate_manifest["dense_h"].get("matrix_sha256")
        or calibration.get("fit_only") is not True
        or calibration.get("selection_used_for_encoding_calibration") is not False
        or calibration.get("holdout_used") is not False
        or calibration.get("fit_routes", {}).get("expert") != expert
        or calibration.get("fit_routes", {}).get("role") != "fit"
    ):
        raise ValueError("expert calibration/H13 lineage differs")
    if manifest.get("shared_vector_sha256") != {
        "gate_up_suh": gate_manifest["scales"]["suh_sha256"],
        "down_svh": down_manifest["scales"]["svh_sha256"],
    }:
        raise ValueError("expert shared-vector hashes differ")
    selection_hash = manifest.get("selection_evidence_sha256")
    if (
        not isinstance(selection_hash, str)
        or len(selection_hash) != 64
        or any(char not in "0123456789abcdef" for char in selection_hash)
    ):
        raise ValueError("expert selection evidence hash differs")
    lineage = manifest.get("lineage", {})
    zero_keys = (
        "mcg_payload_reads",
        "mcg_transform_reads",
        "mcg_scale_reads",
        "mcg_permutation_reads",
        "mcg_seed_reads",
        "mcg_decoded_weight_reads",
    )
    if any(lineage.get(key) != 0 for key in zero_keys) or lineage.get(
        "legacy_tensor_names"
    ) != []:
        raise ValueError("expert artifact does not prove zero-MCG lineage")
    return manifest


def load_decoded_expert(
    manifest_path: str | Path,
    *,
    lut_by_bits: Mapping[int, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Independently decode a mini-shard for routed replay."""

    manifest = validate_expert_artifact(manifest_path)
    shard = Path(manifest_path).parent / str(manifest["shard"])
    layer = int(manifest["layer"])
    expert = int(manifest["expert"])
    output: dict[str, torch.Tensor] = {}
    with safe_open(shard, framework="pt", device="cpu") as handle:
        for projection in PROJECTIONS:
            prefix = tensor_prefix(layer, expert, projection)
            bits = int(manifest["bits"][projection])
            lut = lut_by_bits[bits]
            result = decode_stored_fp16(
                handle.get_tensor(f"{prefix}.trellis"),
                handle.get_tensor(f"{prefix}.suh"),
                handle.get_tensor(f"{prefix}.svh"),
                bits=bits,
                codebook_e4m3=lut,
            )
            expected = manifest["tensor_manifests"][prefix]["decoded_closure"][
                "decoded_exl_sha256"
            ]
            if tensor_sha256(result) != expected:
                raise ValueError(f"{prefix}: independent artifact decode hash differs")
            output[projection] = result
    return output


def assemble_layer_artifact(
    expert_dir: str | Path,
    output_dir: str | Path,
    *,
    run_id: str,
    layer: int,
    bit_map: Mapping[str, int],
    gate_up_profile: SharedResidualProfile,
    down_profile: SharedResidualProfile,
    layer_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Deduplicate shared sides and stream 256 sealed mini-shards to one layer."""

    layer = validate_layer(layer)
    if len(bit_map) != NUM_EXPERTS * len(PROJECTIONS):
        raise ValueError("layer assembly bit-map count differs")
    values = tuple(bit_map.values())
    if (values.count(3), values.count(4), sum(values)) != (
        EXPECTED_K3_PER_LAYER,
        EXPECTED_K4_PER_LAYER,
        EXPECTED_BIT_UNITS_PER_LAYER,
    ):
        raise ValueError("layer assembly bit budget is not exact 384/384")
    expert_root = Path(expert_dir)
    manifests: dict[int, dict[str, Any]] = {}
    readers: dict[int, SafeTensorReader] = {}
    for expert in range(NUM_EXPERTS):
        path = expert_root / f"{expert_stem(layer, expert)}.json"
        item = validate_expert_artifact(path)
        if item.get("run_id") != run_id or item.get("purpose") != "final_treatment":
            raise ValueError(f"expert {expert} artifact binding differs")
        if item["shared_profiles"]["gate_up_input"] != gate_up_profile.manifest():
            raise ValueError(f"expert {expert} gate/up shared profile differs")
        if item["shared_profiles"]["down_output"] != down_profile.manifest():
            raise ValueError(f"expert {expert} down shared profile differs")
        manifests[expert] = item
        readers[expert] = SafeTensorReader(expert_root / item["shard"])

    entries: list[TensorEntry] = [
        torch_tensor_entry(shared_gate_up_name(layer), gate_up_profile.expected_stored_fp16()),
        torch_tensor_entry(shared_down_name(layer), down_profile.expected_stored_fp16()),
    ]
    vector_refs: dict[str, dict[str, str]] = {}
    tensor_manifests: list[Mapping[str, object]] = []
    permutations: list[Mapping[str, object]] = []
    for expert in range(NUM_EXPERTS):
        reader = readers[expert]
        item = manifests[expert]
        permutations.append(item["permutation"])
        for projection in PROJECTIONS:
            prefix = tensor_prefix(layer, expert, projection)
            private_side = "svh" if projection != "down_proj" else "suh"
            shared_name = (
                shared_gate_up_name(layer)
                if projection != "down_proj"
                else shared_down_name(layer)
            )
            vector_refs[prefix] = (
                {"suh": shared_name, "svh": f"{prefix}.svh"}
                if projection != "down_proj"
                else {"suh": f"{prefix}.suh", "svh": shared_name}
            )
            for suffix in ("trellis", private_side, "sqg"):
                source = reader.tensors[f"{prefix}.{suffix}"]
                entries.append(
                    TensorEntry(
                        f"{prefix}.{suffix}",
                        source.dtype,
                        source.shape,
                        source.payload,
                    )
                )
            tensor_manifests.append(item["tensor_manifests"][prefix])
    assert_no_forbidden_tensor_names(entry.name for entry in entries)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    shard_path = destination / f"fresh-sqg-layer-{layer:03d}.safetensors"
    manifest_path = destination / f"fresh-sqg-layer-{layer:03d}.json"
    if shard_path.exists() or manifest_path.exists():
        existing = validate_layer_artifact(manifest_path)
        if existing.get("run_id") != run_id:
            raise ValueError("existing layer artifact belongs to another run")
        return existing
    payload_hashes, shard_hash = write_safetensors_atomic(
        shard_path,
        entries,
        metadata={
            "format": "pt",
            "schema": LAYER_ARTIFACT_SCHEMA,
            "run_id": run_id,
            "layer": str(layer),
            "codebook": "sqg_xor_cheb_t12",
            "sqg_marker": "0x53514731",
            "topology": "shared_gate_up_suh_shared_down_svh",
            "legacy_mcg_input": "false",
        },
    )
    budget = FrozenBitBudget(
        bit_map=dict(bit_map),
        expected_k3=EXPECTED_K3_PER_LAYER,
        expected_k4=EXPECTED_K4_PER_LAYER,
    )
    run_manifest = build_run_manifest(
        run_id=f"{run_id}/layer-{layer:03d}",
        bit_budget=budget,
        tensor_manifests=tensor_manifests,
        fresh_permutations=permutations,
    )
    expert_manifest_hashes = {
        str(expert): sha256_file(
            expert_root / f"{expert_stem(layer, expert)}.json"
        )
        for expert in range(NUM_EXPERTS)
    }
    manifest: dict[str, object] = {
        "schema": LAYER_ARTIFACT_SCHEMA,
        "complete": True,
        "run_id": run_id,
        "layer": layer,
        "shard": shard_path.name,
        "shard_sha256": shard_hash,
        "payload_sha256": payload_hashes,
        "bit_map": dict(sorted(bit_map.items())),
        "bit_histogram": {"3": values.count(3), "4": values.count(4)},
        "allocation_bit_units": sum(values),
        "shared_vectors": {
            "gate_up_suh": shared_gate_up_name(layer),
            "down_svh": shared_down_name(layer),
        },
        "shared_profiles": {
            "gate_up_input": gate_up_profile.manifest(),
            "down_output": down_profile.manifest(),
        },
        "vector_refs": dict(sorted(vector_refs.items())),
        "expert_manifest_sha256": expert_manifest_hashes,
        "fresh_sqg_run_manifest": run_manifest,
        "evidence": dict(layer_evidence),
        "lineage": {
            "mcg_tensor_count": 0,
            "sqg_tensor_count": NUM_EXPERTS * len(PROJECTIONS),
            "official_bf16_source": True,
            "fresh_calibration": True,
            "frozen_bit_map_only": True,
        },
    }
    atomic_json(manifest_path, manifest)
    _seal_json(manifest_path)
    return validate_layer_artifact(manifest_path)


def validate_layer_artifact(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).absolute()
    _check_json_seal(manifest_path)
    manifest = load_json_object(manifest_path)
    if manifest.get("schema") != LAYER_ARTIFACT_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("layer artifact is incomplete or foreign")
    layer = validate_layer(int(manifest.get("layer", -1)))
    shard = _artifact_shard(
        manifest_path,
        manifest.get("shard"),
        f"fresh-sqg-layer-{layer:03d}.safetensors",
        label="layer artifact",
    )
    if sha256_file(shard) != manifest.get("shard_sha256"):
        raise ValueError("layer artifact shard hash differs")
    reader = SafeTensorReader(shard)
    names = set(reader.tensors)
    assert_no_forbidden_tensor_names(names)
    expected = {shared_gate_up_name(layer), shared_down_name(layer)}
    for expert in range(NUM_EXPERTS):
        for projection in PROJECTIONS:
            prefix = tensor_prefix(layer, expert, projection)
            expected.update(
                {
                    f"{prefix}.trellis",
                    f"{prefix}.sqg",
                    f"{prefix}.svh" if projection != "down_proj" else f"{prefix}.suh",
                }
            )
    if names != expected or len(names) != 2_306:
        raise ValueError("layer artifact topology-neutral tensor inventory differs")
    if set(manifest.get("payload_sha256", {})) != expected:
        raise ValueError("layer artifact payload hash domain differs")
    marker_names: list[str] = []
    for name, info in reader.tensors.items():
        if info.payload.sha256() != manifest["payload_sha256"][name]:
            raise ValueError(f"layer payload hash differs: {name}")
        if name.endswith(".sqg"):
            if info.dtype != "I32" or info.shape != ():
                raise ValueError(f"layer SQG marker geometry differs: {name}")
            marker_names.append(name)
    if len(marker_names) != NUM_EXPERTS * len(PROJECTIONS):
        raise ValueError("layer SQG marker count differs")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        for name in marker_names:
            if int(handle.get_tensor(name)) != SQG_MARKER:
                raise ValueError(f"layer SQG marker value differs: {name}")
    values = tuple(manifest.get("bit_map", {}).values())
    if (len(values), values.count(3), values.count(4), sum(values)) != (
        768,
        384,
        384,
        2688,
    ):
        raise ValueError("layer artifact exact bit budget differs")
    expected_prefixes = {
        tensor_prefix(layer, expert, projection)
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    }
    if set(manifest.get("bit_map", {})) != expected_prefixes:
        raise ValueError("layer artifact bit-map domain differs")
    vector_refs = manifest.get("vector_refs", {})
    if set(vector_refs) != expected_prefixes:
        raise ValueError("layer artifact shared-vector reference domain differs")
    for prefix in expected_prefixes:
        projection = prefix.rsplit(".", 1)[-1]
        expected_refs = (
            {
                "suh": shared_gate_up_name(layer),
                "svh": f"{prefix}.svh",
            }
            if projection != "down_proj"
            else {
                "suh": f"{prefix}.suh",
                "svh": shared_down_name(layer),
            }
        )
        if vector_refs[prefix] != expected_refs:
            raise ValueError(f"layer shared-vector reference differs: {prefix}")
    run = manifest.get("fresh_sqg_run_manifest", {})
    if run.get("forbidden_reads", {}).get("observed") != []:
        raise ValueError("layer run manifest reports a forbidden read")
    if (
        run.get("run_id") != f"{manifest.get('run_id')}/layer-{layer:03d}"
        or run.get("production") is not True
        or run.get("codec", {}).get("marker", {}).get("int32") != SQG_MARKER
        or run.get("codec", {}).get("fallback_allowed") is not False
        or run.get("frozen_bit_budget", {}).get("assignments")
        != manifest.get("bit_map")
    ):
        raise ValueError("layer nested fresh-SQG run manifest differs")
    run_items = run.get("tensor_manifests", [])
    if not isinstance(run_items, list) or len(run_items) != len(expected_prefixes):
        raise ValueError("layer nested tensor-manifest count differs")
    run_by_id = {str(item.get("tensor_id")): item for item in run_items}
    if set(run_by_id) != expected_prefixes:
        raise ValueError("layer nested tensor-manifest domain differs")
    for prefix, item in run_by_id.items():
        validate_tensor_manifest(item)
        if item.get("bits") != manifest["bit_map"][prefix]:
            raise ValueError(f"layer nested frozen rate differs: {prefix}")
    permutations = run.get("fresh_physical_permutations", [])
    expected_scopes = {
        permutation_scope(layer, expert) for expert in range(NUM_EXPERTS)
    }
    if (
        not isinstance(permutations, list)
        or len(permutations) != NUM_EXPERTS
        or {item.get("scope") for item in permutations} != expected_scopes
        or any(item.get("production_qualified") is not True for item in permutations)
    ):
        raise ValueError("layer nested fresh-permutation census differs")
    shared_profiles = manifest.get("shared_profiles", {})
    if set(shared_profiles) != {"gate_up_input", "down_output"}:
        raise ValueError("layer shared-profile evidence differs")
    expert_hashes = manifest.get("expert_manifest_sha256", {})
    if set(expert_hashes) != {str(expert) for expert in range(NUM_EXPERTS)}:
        raise ValueError("layer expert-manifest hash domain differs")
    expert_root = manifest_path.parent.parent / "experts"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        shared_gate_hash = tensor_sha256(handle.get_tensor(shared_gate_up_name(layer)))
        shared_down_hash = tensor_sha256(handle.get_tensor(shared_down_name(layer)))
        if (
            shared_gate_hash
            != shared_profiles["gate_up_input"].get("expected_stored_fp16_sha256")
            or shared_down_hash
            != shared_profiles["down_output"].get("expected_stored_fp16_sha256")
        ):
            raise ValueError("layer persisted shared-profile vectors differ")
        run_permutation_by_scope = {item["scope"]: item for item in permutations}
        for expert in range(NUM_EXPERTS):
            expert_path = expert_root / f"{expert_stem(layer, expert)}.json"
            if (
                not expert_path.is_file()
                or sha256_file(expert_path) != expert_hashes[str(expert)]
            ):
                raise ValueError(f"layer expert evidence hash differs: {expert}")
            expert_manifest = validate_expert_artifact(expert_path)
            if (
                expert_manifest.get("purpose") != "final_treatment"
                or expert_manifest.get("run_id") != manifest.get("run_id")
                or expert_manifest.get("shared_profiles") != shared_profiles
                or expert_manifest.get("permutation")
                != run_permutation_by_scope[permutation_scope(layer, expert)]
            ):
                raise ValueError(f"layer expert evidence binding differs: {expert}")
            for projection in PROJECTIONS:
                prefix = tensor_prefix(layer, expert, projection)
                nested = run_by_id[prefix]
                if expert_manifest["tensor_manifests"][prefix] != nested:
                    raise ValueError(f"layer/expert tensor lineage differs: {prefix}")
                trellis_name = f"{prefix}.trellis"
                private_suffix = "svh" if projection != "down_proj" else "suh"
                private_name = f"{prefix}.{private_suffix}"
                shared_suffix = "suh" if projection != "down_proj" else "svh"
                shared_name = f"{prefix}.{shared_suffix}"
                if (
                    reader.tensors[trellis_name].payload.sha256()
                    != nested.get("packed_trellis_payload_sha256")
                    or tensor_sha256(handle.get_tensor(private_name))
                    != nested["scales"][f"{private_suffix}_sha256"]
                    or (
                        shared_gate_hash
                        if projection != "down_proj"
                        else shared_down_hash
                    )
                    != nested["scales"][f"{shared_suffix}_sha256"]
                    or manifest["payload_sha256"][trellis_name]
                    != expert_manifest["payload_sha256"][trellis_name]
                    or manifest["payload_sha256"][private_name]
                    != expert_manifest["payload_sha256"][private_name]
                    or manifest["payload_sha256"][
                        f"{prefix}.sqg"
                    ]
                    != expert_manifest["payload_sha256"][f"{prefix}.sqg"]
                    or manifest["payload_sha256"][
                        shared_gate_up_name(layer)
                        if projection != "down_proj"
                        else shared_down_name(layer)
                    ]
                    != expert_manifest["payload_sha256"][shared_name]
                ):
                    raise ValueError(f"layer/expert payload binding differs: {prefix}")
    if manifest.get("lineage", {}).get("mcg_tensor_count") != 0:
        raise ValueError("layer artifact reports a legacy MCG tensor")
    if manifest.get("lineage", {}).get("sqg_tensor_count") != len(
        expected_prefixes
    ):
        raise ValueError("layer artifact reports an invalid SQG tensor census")
    return manifest
