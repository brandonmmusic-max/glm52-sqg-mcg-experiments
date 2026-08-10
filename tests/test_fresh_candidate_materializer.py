from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from bmmlaw_r7_encoder.safetensors_io import (
    TensorEntry,
    write_safetensors_atomic,
)
import src.fresh_candidate_materializer as candidate_module
import src.fresh_pipeline_common as common_module
from src.calibration_capture import TEACHER_IDENTITY_VALIDATION
from src.fresh_candidate_materializer import (
    MANIFEST_NAME,
    RUN_SEAL_COPY_NAME,
    VERIFIED_NAME,
    materialize_candidate,
    validate_candidate,
    validate_run_seal,
)
from src.fresh_pipeline_common import (
    BIT_CONTRACT_SCHEMA,
    PROJECTIONS,
    SELECTED_LAYERS,
    SQG_MARKER,
    atomic_json,
    canonical_sha256,
    sha256_file,
    tensor_prefix,
)
from src.teacher_identity import IDENTITY_FILES, INDEX_FILE


def _bit_map(layer: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for expert in range(256):
        for projection_index, projection in enumerate(PROJECTIONS):
            prefix = tensor_prefix(layer, expert, projection)
            result[prefix] = 3 if (expert * 3 + projection_index) < 384 else 4
    return result


def _entries(layer: int, *, fresh: bool) -> list[TensorEntry]:
    entries = [
        TensorEntry(
            f"model.layers.{layer}.mlp.experts.r7_shared.gate_up_suh",
            "F16",
            (1,),
            b"\0\0",
        ),
        TensorEntry(
            f"model.layers.{layer}.mlp.experts.r7_shared.down_svh",
            "F16",
            (1,),
            b"\0\0",
        ),
    ]
    bits = _bit_map(layer)
    for expert in range(256):
        for projection in PROJECTIONS:
            prefix = tensor_prefix(layer, expert, projection)
            k = bits[prefix]
            entries.append(
                TensorEntry(prefix + ".trellis", "I16", (1, 1, 16 * k), bytes(32 * k))
            )
            if fresh:
                entries.append(
                    TensorEntry(
                        prefix + ".sqg",
                        "I32",
                        (),
                        SQG_MARKER.to_bytes(4, "little", signed=True),
                    )
                )
            else:
                entries.append(TensorEntry(prefix + ".mcg", "I32", (), b"MCG!"))
            private = ".suh" if projection == "down_proj" else ".svh"
            entries.append(TensorEntry(prefix + private, "F16", (1,), b"\0\0"))
    return entries


def _source_record(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "role": "fixture",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_bound_json(path: Path, value: dict, id_field: str) -> dict:
    result = deepcopy(value)
    result[id_field] = canonical_sha256(result)
    atomic_json(path, result)
    return result


def _sealed_run_tree(root: Path) -> tuple[Path, dict[str, object]]:
    root.mkdir()
    preflight_path = root / "preflight.json"
    atomic_json(preflight_path, {"fixture": "preflight"})
    preflight: dict[str, object] = {
        "preflight_id": "preflight-id",
        "paths": {"output_root": str(root)},
        "settings": {"run_id": "fixture-run", "sigma_reg": 0.025},
        "source_seal": {"sha256": "1" * 64},
        "capture": {"sha256": "2" * 64},
        "bit_contract": {"sha256": "3" * 64},
        "kquant": {"revision": "fixture"},
    }
    layers: dict[str, object] = {}
    for layer in SELECTED_LAYERS:
        layer_root = root / f"layer_{layer:03d}"
        profile_root = layer_root / "profile_search"
        final_root = layer_root / "final"
        profile_root.mkdir(parents=True)
        final_root.mkdir()
        binding = {
            "run_id": "fixture-run",
            "preflight_id": "preflight-id",
            "layer": layer,
            "source_seal_sha256": "1" * 64,
            "capture_manifest_sha256": "2" * 64,
            "bit_contract_sha256": "3" * 64,
            "kquant": {"revision": "fixture"},
            "sigma_reg": 0.025,
        }
        selection = _write_bound_json(
            profile_root / "selection.json",
            {
                "schema": "glm52-fresh-sqg-shared-profile-selection-v1",
                "complete": True,
                "binding": binding,
                "factorial_cell_count": 32,
                "required_factorial_cell_count": 32,
                "all_cells_exact_sqg": True,
                "all_cells_candidate_conditional_h2": True,
                "all_cells_exact_down_sqg": True,
                "all_cells_routed_selection_scored": True,
                "proxy_pruning_used": False,
                "holdout_used": False,
                "selection_frozen_before_holdout": True,
                "fallback_count": 0,
                "mcg_inputs": 0,
                "selected_cell_id": f"layer-{layer}-cell",
            },
            "selection_id",
        )
        selection_path = profile_root / "selection.json"
        selection_sha = sha256_file(selection_path)
        shard_path = final_root / f"fresh-sqg-layer-{layer:03d}.safetensors"
        shard_path.write_bytes(f"fresh-layer-{layer}".encode())
        manifest_path = final_root / f"fresh-sqg-layer-{layer:03d}.json"
        manifest = {
            "schema": "glm52-fresh-sqg-layer-artifact-v1",
            "complete": True,
            "run_id": "fixture-run",
            "layer": layer,
            "shard": shard_path.name,
            "shard_sha256": sha256_file(shard_path),
            "evidence": {
                "profile_selection_sha256": selection_sha,
                "profile_selection_id": selection["selection_id"],
                "selected_cell_id": selection["selected_cell_id"],
            },
        }
        atomic_json(manifest_path, manifest)
        (final_root / f"fresh-sqg-layer-{layer:03d}.json.sha256").write_text(
            f"{sha256_file(manifest_path)}  {manifest_path.name}\n",
            encoding="ascii",
        )
        score = {
            "role": "holdout",
            "expert_count": 256,
        }
        score["score_id"] = canonical_sha256(score)
        holdout = _write_bound_json(
            final_root / "holdout.json",
            {
                "schema": "glm52-fresh-sqg-holdout-routed-functional-v1",
                "complete": True,
                "binding": binding,
                "role": "holdout",
                "selection_sha256": selection_sha,
                "layer_artifact_sha256": sha256_file(manifest_path),
                "selection_frozen_before_holdout": True,
                "artifact_changed_by_holdout": False,
                "holdout_used_for_calibration": False,
                "holdout_used_for_profile_choice": False,
                "holdout_report_only": True,
                "routed_functional_score": score,
                "fallback_count": 0,
                "mcg_inputs": 0,
            },
            "report_id",
        )
        layers[str(layer)] = {
            "layer_artifact": str(manifest_path),
            "layer_artifact_sha256": sha256_file(manifest_path),
            "layer_shard_sha256": sha256_file(shard_path),
            "selection_sha256": selection_sha,
            "selection_id": selection["selection_id"],
            "selected_cell_id": selection["selected_cell_id"],
            "holdout_sha256": sha256_file(final_root / "holdout.json"),
            "holdout_report_id": holdout["report_id"],
            "sqg_tensors": 768,
            "mcg_tensors": 0,
            "K3": 384,
            "K4": 384,
        }
    _write_bound_json(
        root / "run_seal.json",
        {
            "schema": "glm52-fresh-sqg-four-layer-run-seal-v1",
            "complete": True,
            "run_id": "fixture-run",
            "preflight_sha256": sha256_file(preflight_path),
            "layers": layers,
            "census": {
                "layers": 4,
                "experts": 1_024,
                "sqg_tensors": 3_072,
                "mcg_tensors": 0,
                "K3": 1_536,
                "K4": 1_536,
                "other_rates": 0,
                "factorial_cells_per_layer": 32,
                "exact_factorial_cells_total": 128,
            },
            "lineage": {
                "official_bf16": True,
                "fresh_capture": True,
                "frozen_bit_map_only": True,
                "fresh_sqg_encoding": True,
                "legacy_mcg_artifact_reads": 0,
                "fallback_count": 0,
            },
            "scope": {
                "four_layer_replacement_artifacts": True,
                "runnable_model_materialized": False,
                "production_container_touched": False,
                "existing_model_mutated": False,
            },
        },
        "run_seal_id",
    )
    return root / "run_seal.json", preflight


def _convert_to_time_priority_no_holdout(run_seal: Path) -> None:
    seal = json.loads(run_seal.read_text(encoding="utf-8"))
    for layer in SELECTED_LAYERS:
        layer_root = run_seal.parent / f"layer_{layer:03d}"
        selection_path = layer_root / "profile_search/selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection.pop("selection_id")
        selection.update(
            {
                "factorial_cell_count": 16,
                "required_factorial_cell_count": 16,
                "preregistered_factorial_cell_count": 32,
                "selection_scope": {
                    "completed_draw_prefix": 4,
                    "selected_cell_count": 16,
                    "preregistered_draws": 8,
                    "preregistered_cell_count": 32,
                    "operator_time_priority": True,
                    "unscored_cells_excluded": 16,
                    "all_selected_cells_exact_sqg": True,
                    "selection_math_uses_original_31_comparison_bonferroni": True,
                },
            }
        )
        selection["selection_id"] = canonical_sha256(selection)
        atomic_json(selection_path, selection, overwrite=True)

        manifest_path = layer_root / f"final/fresh-sqg-layer-{layer:03d}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evidence"].update(
            {
                "profile_selection_sha256": sha256_file(selection_path),
                "profile_selection_id": selection["selection_id"],
            }
        )
        atomic_json(manifest_path, manifest, overwrite=True)
        manifest_path.with_suffix(".json.sha256").write_text(
            f"{sha256_file(manifest_path)}  {manifest_path.name}\n",
            encoding="ascii",
        )
        (layer_root / "final/holdout.json").unlink()

        record = seal["layers"][str(layer)]
        record.update(
            {
                "layer_artifact_sha256": sha256_file(manifest_path),
                "selection_sha256": sha256_file(selection_path),
                "selection_id": selection["selection_id"],
                "holdout": {
                    "status": "skipped",
                    "reason": "operator_time_priority",
                    "report_present": False,
                    "routed_score_present": False,
                    "used_for_calibration": False,
                    "used_for_profile_choice": False,
                    "selection_sha256": sha256_file(selection_path),
                    "layer_artifact_sha256": sha256_file(manifest_path),
                },
            }
        )
        record.pop("holdout_sha256")
        record.pop("holdout_report_id")
    seal["census"].update(
        {"factorial_cells_per_layer": 16, "exact_factorial_cells_total": 64}
    )
    seal["scope"]["routed_holdout"] = {
        "status": "skipped",
        "reason": "operator_time_priority",
        "layers": list(SELECTED_LAYERS),
        "report_count": 0,
        "routed_score_count": 0,
    }
    seal.pop("run_seal_id")
    seal["run_seal_id"] = canonical_sha256(seal)
    atomic_json(run_seal, seal, overwrite=True)


@pytest.fixture(scope="module")
def candidate_fixture(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("candidate-materializer")
    source = root / "source"
    artifacts = root / "artifacts"
    source.mkdir()
    artifacts.mkdir()

    bit_contract = root / "bit-contract.json"
    bit_layers = {}
    for layer in SELECTED_LAYERS:
        bit_layers[str(layer)] = {
            "bit_map": _bit_map(layer),
            "histogram": {"3": 384, "4": 384},
        }
    atomic_json(
        bit_contract,
        {
            "schema": BIT_CONTRACT_SCHEMA,
            "purpose": "topology-neutral per-tensor K3/K4 experimental control",
            "layers": bit_layers,
            "totals": {"K3": 1536, "K4": 1536, "layers": 4, "tensors": 3072},
        },
    )

    source_weight_map: dict[str, str] = {}
    selected_shards = {}
    artifact_layers = {}
    seal_layers = {}
    for layer in SELECTED_LAYERS:
        source_shard_name = f"r7-experts-layer-{layer:03d}.safetensors"
        source_shard = source / source_shard_name
        _, _ = write_safetensors_atomic(source_shard, _entries(layer, fresh=False))
        selected_shards[layer] = source_shard_name
        from bmmlaw_r7_encoder.safetensors_io import SafeTensorReader

        for name in SafeTensorReader(source_shard).tensors:
            source_weight_map[name] = source_shard_name

        final = artifacts / f"layer_{layer:03d}" / "final"
        final.mkdir(parents=True)
        artifact_shard = final / f"fresh-sqg-layer-{layer:03d}.safetensors"
        payload_hashes, shard_hash = write_safetensors_atomic(
            artifact_shard,
            _entries(layer, fresh=True),
        )
        artifact_manifest_path = final / f"fresh-sqg-layer-{layer:03d}.json"
        vector_refs = {}
        for expert in range(256):
            for projection in PROJECTIONS:
                prefix = tensor_prefix(layer, expert, projection)
                vector_refs[prefix] = (
                    {
                        "suh": f"model.layers.{layer}.mlp.experts.r7_shared.gate_up_suh",
                        "svh": prefix + ".svh",
                    }
                    if projection != "down_proj"
                    else {
                        "suh": prefix + ".suh",
                        "svh": f"model.layers.{layer}.mlp.experts.r7_shared.down_svh",
                    }
                )
        artifact_manifest = {
            "schema": "glm52-fresh-sqg-layer-artifact-v1",
            "run_id": "fixture-run",
            "layer": layer,
            "shard": artifact_shard.name,
            "shard_sha256": shard_hash,
            "payload_sha256": payload_hashes,
            "bit_map": _bit_map(layer),
            "shared_vectors": {
                "gate_up_suh": f"model.layers.{layer}.mlp.experts.r7_shared.gate_up_suh",
                "down_svh": f"model.layers.{layer}.mlp.experts.r7_shared.down_svh",
            },
            "vector_refs": vector_refs,
        }
        atomic_json(artifact_manifest_path, artifact_manifest)
        seal_path = artifact_manifest_path.with_suffix(".json.sha256")
        seal_path.write_text(
            f"{sha256_file(artifact_manifest_path)}  {artifact_manifest_path.name}\n",
            encoding="ascii",
        )
        record = {
            "layer_artifact": str(artifact_manifest_path),
            "layer_artifact_sha256": sha256_file(artifact_manifest_path),
            "layer_shard_sha256": shard_hash,
            "sqg_tensors": 768,
            "mcg_tensors": 0,
            "K3": 384,
            "K4": 384,
        }
        seal_layers[str(layer)] = record
        artifact_layers[layer] = {
            "record": record,
            "manifest": artifact_manifest,
            "manifest_path": artifact_manifest_path,
            "manifest_sha256": sha256_file(artifact_manifest_path),
            "manifest_seal_path": seal_path,
            "manifest_seal_sha256": sha256_file(seal_path),
            "shard_path": artifact_shard,
        }

    unselected_shard = source / "model-layer-000.safetensors"
    write_safetensors_atomic(
        unselected_shard,
        [TensorEntry("model.layers.0.test", "F16", (1,), b"\0\0")],
    )
    source_weight_map["model.layers.0.test"] = unselected_shard.name
    total_size = sum(
        info.nbytes
        for name in set(source_weight_map.values())
        for info in __import__(
            "bmmlaw_r7_encoder.safetensors_io",
            fromlist=["SafeTensorReader"],
        ).SafeTensorReader(source / name).tensors.values()
    )
    source_index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(source_weight_map.items())),
    }
    atomic_json(source / INDEX_FILE, source_index)

    sidecars = {f"r7-experts-layer-{layer:03d}.json" for layer in SELECTED_LAYERS}
    sidecars.add("r7-experts-layer-003.json")
    for name in sidecars:
        atomic_json(source / name, {"source": "legacy", "name": name})
    quant = {
        "quant_method": "exl3",
        "r7_routed_experts": {
            "schema": "r7-complete-v2-checkpoint-v1",
            "bits": "mixed_tensor",
            "codebook": "mcg",
            "moe_layers": [3, 77],
            "k_values": [3, 4, 5],
            "bit_map_manifests": sorted(sidecars),
        },
    }
    config = {"model_type": "fixture", "quantization_config": quant}
    atomic_json(source / "quantization_config.json", quant)
    atomic_json(source / "config.json", config)
    for name in set(IDENTITY_FILES) - {
        INDEX_FILE,
        "config.json",
        "quantization_config.json",
    }:
        (source / name).write_bytes((name + "\n").encode())
    (source / "README.md").write_text("must not be copied", encoding="utf-8")
    (source / "calibration").mkdir()
    (source / "calibration" / "private.jsonl").write_text("{}\n", encoding="utf-8")
    (source / "MANIFEST.json").write_text("stale", encoding="utf-8")
    (source / ".manifest_verified").write_text("stale", encoding="utf-8")

    receipt = root / "teacher-receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    run_seal = artifacts / "run_seal.json"
    seal = {
        "schema": "glm52-fresh-sqg-four-layer-run-seal-v1",
        "complete": True,
        "run_id": "fixture-run",
        "layers": seal_layers,
        "run_seal_id": "a" * 64,
    }
    atomic_json(run_seal, seal)
    run_context = {
        "seal": seal,
        "path": run_seal,
        "sha256": sha256_file(run_seal),
        "layers": artifact_layers,
    }

    payloads = {unselected_shard.name} | set(selected_shards.values())
    runtime_names = payloads | sidecars | set(IDENTITY_FILES)
    file_records = {name: _source_record(source / name) for name in runtime_names}
    source_context = {
        "validation": deepcopy(TEACHER_IDENTITY_VALIDATION),
        "files": file_records,
        "payloads": payloads,
        "sidecars": sidecars,
        "index": source_index,
        "external_quant": quant,
        "embedded_config": config,
        "selected_shards": selected_shards,
    }
    source_before = {name: sha256_file(source / name) for name in runtime_names}
    return {
        "root": root,
        "source": source,
        "receipt": receipt,
        "artifacts": artifacts,
        "run_seal": run_seal,
        "bit_contract": bit_contract,
        "source_context": source_context,
        "run_context": run_context,
        "source_before": source_before,
    }


@pytest.fixture
def patched_context(
    candidate_fixture: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    monkeypatch.setattr(
        candidate_module,
        "_source_inventory",
        lambda *args, **kwargs: candidate_fixture["source_context"],
    )
    monkeypatch.setattr(
        candidate_module,
        "validate_run_seal",
        lambda *args, **kwargs: candidate_fixture["run_context"],
    )
    monkeypatch.setattr(
        common_module,
        "FROZEN_BIT_CONTRACT_SHA256",
        sha256_file(candidate_fixture["bit_contract"]),
    )
    return candidate_fixture


def _materialize(context: dict, destination: Path) -> dict:
    return materialize_candidate(
        source_model=context["source"],
        teacher_receipt=context["receipt"],
        run_seal=context["run_seal"],
        artifacts_root=context["artifacts"],
        bit_contract=context["bit_contract"],
        output=destination,
    )


def _validate(context: dict, destination: Path) -> dict:
    return validate_candidate(
        destination,
        source_model=context["source"],
        teacher_receipt=context["receipt"],
        run_seal=context["run_seal"],
        artifacts_root=context["artifacts"],
        bit_contract=context["bit_contract"],
        verify_all_hashes=True,
    )


def test_materializer_builds_only_runtime_allowlist_and_preserves_source(
    patched_context: dict,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "candidate"
    report = _materialize(patched_context, destination)
    assert report["selected_sqg_markers"] == 3_072
    assert report["selected_mcg_markers"] == 0
    assert report["selected_legacy_inodes_reused"] == 0
    assert report["construction_exclusions"] == {
        "mcg_payload_bytes": 0,
        "mcg_transform_vectors": 0,
        "mcg_scale_vectors": 0,
        "mcg_permutations": 0,
        "mcg_encoder_seeds": 0,
        "mcg_decoded_weight_reads": 0,
        "stale_per_tensor_shared_side_scales": 0,
    }
    assert (destination / VERIFIED_NAME).is_file()
    assert (destination / MANIFEST_NAME).is_file()
    assert (destination / RUN_SEAL_COPY_NAME).is_file()
    assert not (destination / "README.md").exists()
    assert not (destination / "calibration").exists()
    assert not (destination / "MANIFEST.sha256").exists()
    for name, digest in patched_context["source_before"].items():
        assert sha256_file(patched_context["source"] / name) == digest
    for layer in SELECTED_LAYERS:
        shard = destination / f"r7-experts-layer-{layer:03d}.safetensors"
        sidecar = destination / f"r7-experts-layer-{layer:03d}.json"
        assert not os.path.samefile(
            shard,
            patched_context["source"] / shard.name,
        )
        assert not os.path.samefile(
            sidecar,
            patched_context["source"] / sidecar.name,
        )
    assert _validate(patched_context, destination)["marker_payloads_verified"] is True


@pytest.mark.parametrize(
    "tamper",
    ("extra_file", "tensor_override", "selected_shard", "selected_sidecar"),
)
def test_candidate_validator_rejects_independent_tamper_classes(
    patched_context: dict,
    tmp_path: Path,
    tamper: str,
) -> None:
    destination = tmp_path / f"candidate-{tamper}"
    _materialize(patched_context, destination)
    if tamper == "extra_file":
        (destination / "README.md").write_text("injected", encoding="utf-8")
    elif tamper == "tensor_override":
        path = destination / "quantization_config.json"
        value = json.loads(path.read_text())
        value["r7_routed_experts"]["codebook_tensor_overrides"] = {
            "model.layers.6.mlp.experts.0.gate_proj": "mcg"
        }
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper == "selected_shard":
        path = destination / "r7-experts-layer-006.safetensors"
        with path.open("r+b") as handle:
            handle.seek(-1, 2)
            last = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([last[0] ^ 1]))
    else:
        path = destination / "r7-experts-layer-006.json"
        value = json.loads(path.read_text())
        value["lineage"]["legacy_mcg_permutations"] = 1
        path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        _validate(patched_context, destination)


def test_materializer_refuses_output_under_any_protected_input(
    patched_context: dict,
) -> None:
    with pytest.raises(ValueError, match="overlaps protected source model"):
        _materialize(
            patched_context,
            patched_context["source"] / "unsafe-candidate",
        )


@pytest.mark.parametrize("protected_key", ("source", "artifacts"))
def test_materializer_canonicalizes_symlinked_output_parent_before_writing(
    patched_context: dict,
    tmp_path: Path,
    protected_key: str,
) -> None:
    protected = patched_context[protected_key]
    alias = tmp_path / f"alias-{protected_key}"
    alias.symlink_to(protected, target_is_directory=True)
    escaped_destination = protected / "must-never-be-created"

    with pytest.raises(ValueError):
        _materialize(patched_context, alias / escaped_destination.name)

    assert not escaped_destination.exists()
    assert not escaped_destination.is_symlink()


def test_materializer_canonicalizes_symlinked_output_ancestor_before_writing(
    patched_context: dict,
    tmp_path: Path,
) -> None:
    protected_parent = patched_context["source"] / "calibration"
    alias = tmp_path / "source-alias"
    alias.symlink_to(patched_context["source"], target_is_directory=True)
    escaped_destination = protected_parent / "must-never-be-created"

    with pytest.raises(ValueError, match="overlaps protected source model"):
        _materialize(
            patched_context,
            alias / protected_parent.name / escaped_destination.name,
        )

    assert not escaped_destination.exists()
    assert not escaped_destination.is_symlink()


def test_materializer_rejects_same_histogram_different_tensor_assignment(
    patched_context: dict,
    tmp_path: Path,
) -> None:
    changed_contract = (
        patched_context["root"] / f"changed-bit-contract-{tmp_path.name}.json"
    )
    value = json.loads(patched_context["bit_contract"].read_text(encoding="utf-8"))
    layer_map = value["layers"]["6"]["bit_map"]
    k3_name = next(name for name, bits in layer_map.items() if bits == 3)
    k4_name = next(name for name, bits in layer_map.items() if bits == 4)
    layer_map[k3_name], layer_map[k4_name] = layer_map[k4_name], layer_map[k3_name]
    atomic_json(changed_contract, value)
    changed_context = {**patched_context, "bit_contract": changed_contract}

    with pytest.raises(ValueError, match="bit-allocation identity differs"):
        _materialize(changed_context, tmp_path / "candidate-different-map")


def test_run_seal_binds_preflight_selection_holdout_and_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_seal, preflight = _sealed_run_tree(tmp_path / "fresh-run")
    monkeypatch.setattr(
        candidate_module, "load_preflight", lambda path, **kwargs: preflight
    )
    monkeypatch.setattr(
        candidate_module,
        "validate_layer_artifact",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )

    result = validate_run_seal(run_seal, run_seal.parent)

    assert result["seal"]["run_id"] == "fixture-run"
    assert set(result["layers"]) == set(SELECTED_LAYERS)
    assert all(
        context["selection_path"].name == "selection.json"
        and context["holdout_path"].name == "holdout.json"
        for context in result["layers"].values()
    )


def test_run_seal_accepts_truthful_time_priority_no_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_seal, preflight = _sealed_run_tree(tmp_path / "fresh-run")
    _convert_to_time_priority_no_holdout(run_seal)
    monkeypatch.setattr(
        candidate_module, "load_preflight", lambda path, **kwargs: preflight
    )
    monkeypatch.setattr(
        candidate_module,
        "validate_layer_artifact",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )

    result = validate_run_seal(run_seal, run_seal.parent)

    assert result["seal"]["scope"]["routed_holdout"]["status"] == "skipped"
    assert all(
        context["holdout_path"] is None
        and context["holdout_sha256"] is None
        for context in result["layers"].values()
    )


def test_run_seal_rejects_preflight_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_seal, preflight = _sealed_run_tree(tmp_path / "fresh-run")
    monkeypatch.setattr(
        candidate_module, "load_preflight", lambda path, **kwargs: preflight
    )
    (run_seal.parent / "preflight.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact root preflight"):
        validate_run_seal(run_seal, run_seal.parent)


def test_run_seal_rejects_selection_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_seal, preflight = _sealed_run_tree(tmp_path / "fresh-run")
    monkeypatch.setattr(
        candidate_module, "load_preflight", lambda path, **kwargs: preflight
    )
    monkeypatch.setattr(
        candidate_module,
        "validate_layer_artifact",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )
    selection = run_seal.parent / "layer_006/profile_search/selection.json"
    value = json.loads(selection.read_text(encoding="utf-8"))
    value["mcg_inputs"] = 1
    atomic_json(selection, value, overwrite=True)

    with pytest.raises(ValueError, match="selection.*binding differs"):
        validate_run_seal(run_seal, run_seal.parent)


def test_run_seal_rejects_symlinked_final_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_seal, preflight = _sealed_run_tree(tmp_path / "fresh-run")
    monkeypatch.setattr(
        candidate_module, "load_preflight", lambda path, **kwargs: preflight
    )
    final = run_seal.parent / "layer_006/final"
    relocated = run_seal.parent / "relocated-final"
    final.rename(relocated)
    final.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(ValueError, match="final directory"):
        validate_run_seal(run_seal, run_seal.parent)


def test_run_seal_rejects_unsafe_layer_shard_path_before_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_seal, preflight = _sealed_run_tree(tmp_path / "fresh-run")
    monkeypatch.setattr(
        candidate_module, "load_preflight", lambda path, **kwargs: preflight
    )
    called = False

    def forbidden_validator(path: Path) -> dict:
        nonlocal called
        called = True
        raise AssertionError("unsafe path reached artifact validator")

    monkeypatch.setattr(candidate_module, "validate_layer_artifact", forbidden_validator)
    manifest_path = run_seal.parent / "layer_006/final/fresh-sqg-layer-006.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shard"] = "../../outside.safetensors"
    atomic_json(manifest_path, manifest, overwrite=True)
    seal = json.loads(run_seal.read_text(encoding="utf-8"))
    seal["layers"]["6"]["layer_artifact_sha256"] = sha256_file(manifest_path)
    seal.pop("run_seal_id")
    seal["run_seal_id"] = canonical_sha256(seal)
    atomic_json(run_seal, seal, overwrite=True)

    with pytest.raises(ValueError, match="shard path is unsafe"):
        validate_run_seal(run_seal, run_seal.parent)
    assert called is False
