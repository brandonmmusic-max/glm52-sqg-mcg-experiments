"""Resumable one-layer-per-GPU fresh SQG construction and evaluation.

The public CLI lives in :mod:`src.run_fresh_sqg`.  Nothing in this module
starts a model server or mutates the existing 3.5-bpw checkpoint.  A worker
accepts only a sealed official BF16 source, the sealed fresh capture, the
sanitized frozen bit-map contract, and hash-bound encoder code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open

from bmmlaw_r7_encoder.safetensors_io import (
    torch_tensor_entry,
    write_safetensors_atomic,
)

from .calibration_capture import validate_capture
from .fresh_pipeline_artifacts import (
    assemble_layer_artifact,
    expert_stem,
    validate_expert_artifact,
    validate_layer_artifact,
    write_expert_artifact,
)
from .fresh_pipeline_calibration import (
    LayerCaptureView,
    ProfileScaleEvidence,
    build_candidate_h2,
    build_layer_h13,
    build_profile_scale_evidence,
    derive_expert_permutation,
    make_shared_profiles,
    mass_stratified_experts,
)
from .fresh_pipeline_common import (
    CUDA_CODEC_SMOKE_SHA256,
    HOLDOUT_REPORT_SCHEMA,
    LAYER_PREP_SCHEMA,
    NUM_EXPERTS,
    PIPELINE_SCHEMA,
    PREFLIGHT_SCHEMA,
    PROFILE_SEARCH_SCHEMA,
    PROJECTIONS,
    RUN_SEAL_SCHEMA,
    SELECTED_LAYERS,
    SOURCE_SEAL_SCHEMA,
    atomic_json,
    canonical_sha256,
    code_tree_manifest,
    derive_seed,
    file_identity,
    exllamav3_provenance,
    git_provenance,
    input_allowlist_manifest,
    load_bit_contract,
    load_json_object,
    local_pipeline_code_provenance,
    rademacher,
    sha256_file,
    source_tensor_name,
    tensor_prefix,
    validate_layer,
)
from .fresh_pipeline_evaluation import (
    paired_document_bootstrap,
    score_routed_artifacts,
)
from .glm52_bf16_manifest import SOURCE_REPO, SOURCE_REVISION
from .glm52_bf16_source import BF16ExpertSource
from .glm52_fresh_sqg import (
    BF16TensorBinding,
    DenseHessian,
    FreshExpertPermutation,
    SharedResidualProfile,
    UniformSQGConfig,
    encode_uniform_sqg,
    load_kquant_runtime,
    prepare_dense_h_session,
    resume_dense_h_session,
)
from .glm52_fresh_sqg.codec import realize_shared_residual_profile
from .glm52_fresh_sqg.permutation import PRODUCTION_PERMUTATION_POLICY
from .glm52_fresh_sqg.reference import (
    tensor_sha256,
)
from .sqg_extension_seal import (
    R33_IMAGE_ID,
    validate_sqg_extension_seal,
)


DEFAULT_SIGMA_REG = 0.025
DEFAULT_PROFILE_DRAWS = 8
ACTIVE_PROFILE_DRAWS = 4
DEFAULT_PANEL_EXPERTS = 16
DEFAULT_CHUNK_ROWS = 256
PROFILE_FAMILIES = (
    "identity",
    "aggregate_rms",
    "quarter_rms",
    "inverse_quarter_rms",
)
BOOTSTRAP_ITERATIONS = 10_000
PROFILE_COMPARISONS = DEFAULT_PROFILE_DRAWS * len(PROFILE_FAMILIES) - 1
FAMILYWISE_ALPHA = 0.05
BONFERRONI_CONFIDENCE = 1.0 - FAMILYWISE_ALPHA / PROFILE_COMPARISONS
LOCAL_CODE_RECOVERY_SCHEMA = (
    "glm52-fresh-sqg-shared-profile-cuda-fp16-recovery-v1"
)
LOCAL_CODE_RECOVERY_NAME = "shared_profile_cuda_fp16_recovery.json"


@dataclass(frozen=True)
class PipelinePaths:
    source_seal: Path
    capture_dir: Path
    bit_contract: Path
    kquant_root: Path
    exllamav3_root: Path
    sqg_extension_seal: Path
    output_root: Path

    @classmethod
    def resolve(
        cls,
        *,
        source_seal: str | Path,
        capture_dir: str | Path,
        bit_contract: str | Path,
        kquant_root: str | Path,
        exllamav3_root: str | Path,
        sqg_extension_seal: str | Path,
        output_root: str | Path,
    ) -> "PipelinePaths":
        return cls(
            source_seal=Path(source_seal).resolve(),
            capture_dir=Path(capture_dir).resolve(),
            bit_contract=Path(bit_contract).resolve(),
            kquant_root=Path(kquant_root).resolve(),
            exllamav3_root=Path(exllamav3_root).resolve(),
            sqg_extension_seal=Path(sqg_extension_seal).resolve(),
            output_root=Path(output_root).resolve(),
        )

    def manifest(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PipelineSettings:
    run_id: str
    sigma_reg: float = DEFAULT_SIGMA_REG
    profile_draws: int = DEFAULT_PROFILE_DRAWS
    panel_experts: int = DEFAULT_PANEL_EXPERTS
    chunk_rows: int = DEFAULT_CHUNK_ROWS

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run ID must not be empty")
        if not math.isfinite(self.sigma_reg) or self.sigma_reg <= 0:
            raise ValueError("sigma_reg must be positive and finite")
        if self.profile_draws != DEFAULT_PROFILE_DRAWS:
            raise ValueError("the preregistered treatment requires exactly 8 sign draws")
        if self.panel_experts != DEFAULT_PANEL_EXPERTS:
            raise ValueError(
                "the preregistered treatment requires exactly 16 panel experts"
            )
        if self.chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")

    def manifest(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "sigma_reg": self.sigma_reg,
            "profile_draws": self.profile_draws,
            "panel_experts": self.panel_experts,
            "chunk_rows": self.chunk_rows,
        }


@dataclass
class LayerRuntime:
    paths: PipelinePaths
    settings: PipelineSettings
    layer: int
    device: str
    source: BF16ExpertSource
    capture: LayerCaptureView
    bit_map: Mapping[str, int]
    source_seal: Mapping[str, object]
    preflight: Mapping[str, object]

    @property
    def layer_root(self) -> Path:
        return self.paths.output_root / f"layer_{self.layer:03d}"

    @property
    def preparation_dir(self) -> Path:
        return self.layer_root / "preparation"

    @property
    def profile_dir(self) -> Path:
        return self.layer_root / "profile_search"

    @property
    def expert_dir(self) -> Path:
        return self.layer_root / "experts"

    @property
    def final_dir(self) -> Path:
        return self.layer_root / "final"


def _validate_source_seal(path: Path) -> dict[str, Any]:
    seal = load_json_object(path)
    if (
        seal.get("schema") != SOURCE_SEAL_SCHEMA
        or seal.get("repo") != SOURCE_REPO
        or seal.get("revision") != SOURCE_REVISION
        or seal.get("layers") != list(SELECTED_LAYERS)
        or seal.get("target_tensor_count") != 3_072
        or seal.get("shard_count") != 18
        or seal.get("complete_index_header_binding_validated") is not True
    ):
        raise ValueError("official BF16 source seal contract differs")
    if not Path(str(seal["index"]["path"])).is_file():
        raise FileNotFoundError(seal["index"]["path"])
    if not Path(str(seal["config"]["path"])).is_file():
        raise FileNotFoundError(seal["config"]["path"])
    shard_root = Path(str(seal["shard_root"])).resolve()
    if not shard_root.is_dir():
        raise FileNotFoundError(shard_root)
    return seal


def _assert_output_disjoint(paths: PipelinePaths, source_seal: Mapping[str, object]) -> None:
    output = paths.output_root
    protected = (
        paths.source_seal,
        paths.capture_dir,
        paths.bit_contract,
        paths.kquant_root,
        paths.exllamav3_root,
        paths.sqg_extension_seal,
        Path(str(source_seal["shard_root"])).resolve(),
    )
    for item in protected:
        candidate = item if item.is_dir() else item.parent
        if output == candidate or output in candidate.parents or candidate in output.parents:
            raise ValueError(
                f"output root must be disjoint from every read-only input: {candidate}"
            )


def _assert_disjoint_directories(output: Path, protected: Sequence[Path]) -> None:
    for candidate in protected:
        candidate = candidate.resolve()
        if output == candidate or output in candidate.parents or candidate in output.parents:
            raise ValueError(
                f"output root must be disjoint from executable input: {candidate}"
            )


def build_preflight(
    paths: PipelinePaths,
    settings: PipelineSettings,
    *,
    verify_capture_hashes: bool = True,
) -> dict[str, object]:
    """Perform all fail-closed read-only validation and bind later workers."""

    source_seal = _validate_source_seal(paths.source_seal)
    _assert_output_disjoint(paths, source_seal)
    budgets = load_bit_contract(paths.bit_contract)
    capture = validate_capture(
        paths.capture_dir, verify_hashes=verify_capture_hashes
    )
    kquant = git_provenance(paths.kquant_root)
    exllama = exllamav3_provenance(paths.exllamav3_root)
    sqg_extension = validate_sqg_extension_seal(
        paths.sqg_extension_seal,
        kquant_root=paths.kquant_root,
        require_runtime_environment=True,
    )
    project_root = Path(__file__).resolve().parents[1]
    local_code = local_pipeline_code_provenance(project_root)
    smoke_path = project_root / "evidence" / "cuda_codec_smoke.json"
    if (
        not smoke_path.is_file()
        or sha256_file(smoke_path) != CUDA_CODEC_SMOKE_SHA256
    ):
        raise RuntimeError("sealed real-CUDA codec smoke evidence differs")
    _assert_disjoint_directories(
        paths.output_root,
        (
            Path(exllama["package_tree"]["root"]),
            Path(exllama["precompiled_extension"]["directory_tree"]["root"]),
            Path(str(sqg_extension["extension"]["path"])).parent,
            project_root / "src",
            project_root / "bmmlaw_r7_encoder",
            project_root / "evidence",
        ),
    )
    # Revalidate every frozen source shard/header/payload against embedded
    # official identities.  The provider is discarded; workers will repeat
    # this before consuming BF16 tensors so a post-preflight mutation fails.
    source = BF16ExpertSource(
        index_path=source_seal["index"]["path"],
        config_path=source_seal["config"]["path"],
        shard_root=source_seal["shard_root"],
        source_revision=SOURCE_REVISION,
    )
    if {
        name: record.sha256 for name, record in source.validation.shards.items()
    } != {
        name: str(record["sha256"])
        for name, record in source_seal["shards"].items()
    }:
        raise ValueError("source provider and source seal shard identities differ")
    input_manifest = input_allowlist_manifest(
        source_seal=paths.source_seal,
        capture_manifest=paths.capture_dir / "capture_manifest.json",
        bit_contract=paths.bit_contract,
        kquant_root=paths.kquant_root,
        exllamav3_root=paths.exllamav3_root,
        sqg_extension_seal=paths.sqg_extension_seal,
    )
    capture_evidence_mode = str(capture.get("evidence_mode", "live_exact_v1"))
    value: dict[str, object] = {
        "schema": PREFLIGHT_SCHEMA,
        "complete": True,
        "pipeline_schema": PIPELINE_SCHEMA,
        "paths": paths.manifest(),
        "settings": settings.manifest(),
        "source_seal": {
            "path": str(paths.source_seal),
            "sha256": sha256_file(paths.source_seal),
            "repo": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "shard_file_identity": {
                name: file_identity(Path(str(source_seal["shard_root"])) / name)
                for name in sorted(source_seal["shards"])
            },
        },
        "capture": {
            "path": str(paths.capture_dir / "capture_manifest.json"),
            "sha256": sha256_file(paths.capture_dir / "capture_manifest.json"),
            "run_uuid": capture["capture_run_uuid"],
            "document_plan_fingerprint": capture["document_plan"]["fingerprint"],
            "evidence_mode": capture_evidence_mode,
            "reference_route_check_required": (
                capture_evidence_mode == "live_exact_v1"
            ),
            "fail_closed_route_pass_required": True,
            "hashes_verified": verify_capture_hashes,
        },
        "bit_contract": {
            "path": str(paths.bit_contract),
            "sha256": sha256_file(paths.bit_contract),
            "sanitized_layers": {
                str(layer): budget.sanitized_manifest()
                for layer, budget in budgets.items()
            },
            "source_sidecar_hash_used": False,
        },
        "kquant": kquant,
        "exllamav3": exllama,
        "sqg_extension": {
            "seal_path": str(paths.sqg_extension_seal),
            "seal_sha256": sha256_file(paths.sqg_extension_seal),
            "seal": sqg_extension,
        },
        "local_pipeline_code": local_code,
        "cuda_codec_smoke": {
            "path": str(smoke_path),
            "sha256": CUDA_CODEC_SMOKE_SHA256,
        },
        "inputs": input_manifest,
        "gates": {
            "official_bf16_exact": True,
            "capture_complete": True,
            "capture_live_router_reference_closed": True,
            "bit_budget_384_k3_384_k4_each_layer": True,
            "fit_selection_holdout_disjoint": True,
            "dense_h_required": True,
            "fallback_allowed": False,
            "mcg_artifact_inputs_allowed": False,
            "production_container_touched": False,
            "model_workload_launched": False,
        },
    }
    value["preflight_id"] = canonical_sha256(value)
    return value


def write_preflight(
    paths: PipelinePaths,
    settings: PipelineSettings,
    *,
    verify_capture_hashes: bool = True,
) -> Path:
    value = build_preflight(
        paths, settings, verify_capture_hashes=verify_capture_hashes
    )
    destination = paths.output_root / "preflight.json"
    atomic_json(destination, value)
    return destination


def _local_code_file_map(value: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for section, prefix in (
        ("src_tree", "src"),
        ("bmmlaw_r7_encoder_tree", "bmmlaw_r7_encoder"),
    ):
        tree = value.get(section)
        files = tree.get("files") if isinstance(tree, Mapping) else None
        if not isinstance(files, Mapping):
            raise ValueError(f"local-code provenance lacks {section} files")
        for relative, record in files.items():
            result[f"{prefix}/{relative}"] = record
    return result


def _local_code_changes(
    old: Mapping[str, object], new: Mapping[str, object]
) -> list[dict[str, object]]:
    old_files = _local_code_file_map(old)
    new_files = _local_code_file_map(new)
    changes: list[dict[str, object]] = []
    for path in sorted(set(old_files) | set(new_files)):
        if old_files.get(path) != new_files.get(path):
            changes.append(
                {
                    "path": path,
                    "old": old_files.get(path),
                    "new": new_files.get(path),
                }
            )
    return changes


def load_preflight(
    path: str | Path, *, require_execution_environment: bool = True
) -> dict[str, Any]:
    preflight_path = Path(path).resolve()
    value = load_json_object(preflight_path)
    if value.get("schema") != PREFLIGHT_SCHEMA or value.get("complete") is not True:
        raise ValueError("preflight is incomplete or foreign")
    expected_id = value.pop("preflight_id", None)
    observed_id = canonical_sha256(value)
    value["preflight_id"] = expected_id
    if expected_id != observed_id:
        raise ValueError("preflight canonical binding differs")
    for section, field in (
        ("source_seal", "sha256"),
        ("capture", "sha256"),
        ("bit_contract", "sha256"),
    ):
        record = value[section]
        if sha256_file(record["path"]) != record[field]:
            raise ValueError(f"preflight input drift: {section}")
    current_kquant = git_provenance(value["paths"]["kquant_root"])
    if current_kquant != value["kquant"]:
        raise ValueError("KQuant code changed after preflight")
    if require_execution_environment:
        current_exllama = exllamav3_provenance(
            value["paths"]["exllamav3_root"]
        )
        if current_exllama != value.get("exllamav3"):
            raise ValueError("ExLlamaV3 package/extension changed after preflight")
    else:
        recorded_exllama = value.get("exllamav3", {})
        recorded_extension = recorded_exllama.get("precompiled_extension", {})
        static_root = Path(
            os.environ.get(
                "FRESH_SQG_STATIC_EXLLAMAV3_ROOT",
                value["paths"]["exllamav3_root"],
            )
        ).resolve()
        recorded_extension_path = Path(str(recorded_extension.get("origin")))
        static_extension_dir = Path(
            os.environ.get(
                "FRESH_SQG_STATIC_EXLLAMA_EXTENSION_DIR",
                str(recorded_extension_path.parent),
            )
        ).resolve()
        extension_path = static_extension_dir / recorded_extension_path.name
        current_package_tree = code_tree_manifest(static_root / "exllamav3")
        current_extension_tree = code_tree_manifest(static_extension_dir)
        recorded_package_tree = recorded_exllama.get("package_tree", {})
        recorded_extension_tree = recorded_extension.get("directory_tree", {})
        if (
            recorded_package_tree.get("file_count")
            != current_package_tree.get("file_count")
            or recorded_package_tree.get("files")
            != current_package_tree.get("files")
            or not extension_path.is_file()
            or extension_path.is_symlink()
            or recorded_extension.get("bytes") != extension_path.stat().st_size
            or recorded_extension.get("sha256") != sha256_file(extension_path)
            or recorded_extension_tree.get("file_count")
            != current_extension_tree.get("file_count")
            or recorded_extension_tree.get("files")
            != current_extension_tree.get("files")
            or recorded_exllama.get("extension_pythonpath_precedence") is not True
            or recorded_exllama.get("jit_compilation_allowed") is not False
        ):
            raise ValueError("static ExLlamaV3 code/extension bytes changed")
    extension_record = value.get("sqg_extension", {})
    extension_seal_path = value["paths"]["sqg_extension_seal"]
    current_sqg_extension = validate_sqg_extension_seal(
        extension_seal_path,
        kquant_root=value["paths"]["kquant_root"],
        require_runtime_environment=require_execution_environment,
    )
    if (
        extension_record.get("seal_path") != extension_seal_path
        or extension_record.get("seal_sha256")
        != sha256_file(extension_seal_path)
        or extension_record.get("seal") != current_sqg_extension
    ):
        raise ValueError("prebuilt SQG extension changed after preflight")
    current_local = local_pipeline_code_provenance(
        Path(__file__).resolve().parents[1]
    )
    if current_local != value.get("local_pipeline_code"):
        recovery_path = (
            preflight_path.parent / "recovery" / LOCAL_CODE_RECOVERY_NAME
        )
        recovery = load_json_object(recovery_path)
        unsigned_recovery = dict(recovery)
        recovery_id = unsigned_recovery.pop("recovery_id", None)
        if (
            recovery.get("schema") != LOCAL_CODE_RECOVERY_SCHEMA
            or recovery.get("complete") is not True
            or recovery_id != canonical_sha256(unsigned_recovery)
            or recovery.get("predecessor_preflight_id") != expected_id
            or recovery.get("predecessor_preflight_sha256")
            != sha256_file(preflight_path)
            or recovery.get("old_local_pipeline_code")
            != value.get("local_pipeline_code")
            or recovery.get("new_local_pipeline_code") != current_local
            or recovery.get("changed_code_files")
            != _local_code_changes(value["local_pipeline_code"], current_local)
            or recovery.get("preflight_rewritten") is not False
            or recovery.get("preregistration_rewritten") is not False
            or recovery.get("encoded_payloads_rewritten") is not False
            or recovery.get("backend_arithmetic_changed") is not False
            or recovery.get("exact_equality_retained") is not True
        ):
            raise ValueError("local-code recovery evidence differs")
        value["_local_code_recovery"] = recovery
    smoke = value.get("cuda_codec_smoke", {})
    if (
        smoke.get("sha256") != CUDA_CODEC_SMOKE_SHA256
        or sha256_file(smoke["path"]) != CUDA_CODEC_SMOKE_SHA256
    ):
        raise ValueError("real-CUDA codec smoke evidence changed after preflight")
    return value


def open_layer_runtime(
    preflight_path: str | Path,
    *,
    layer: int,
    device: str,
) -> LayerRuntime:
    preflight = load_preflight(preflight_path)
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
            "production layer job must expose only its sealed physical GPU "
            f"{expected_visible_gpu} as cuda:0"
        )
    source_seal = _validate_source_seal(paths.source_seal)
    # Repeat official source validation in the worker.  A seal is an identity
    # statement, not permission to trust files that changed afterward.
    source = BF16ExpertSource(
        index_path=source_seal["index"]["path"],
        config_path=source_seal["config"]["path"],
        shard_root=source_seal["shard_root"],
        source_revision=SOURCE_REVISION,
    )
    capture = LayerCaptureView(
        paths.capture_dir, layer, validate=True, verify_hashes=True
    )
    bit_map = load_bit_contract(paths.bit_contract)[layer].bit_map
    return LayerRuntime(
        paths=paths,
        settings=settings,
        layer=layer,
        device=device,
        source=source,
        capture=capture,
        bit_map=bit_map,
        source_seal=source_seal,
        preflight=preflight,
    )


def _revalidate_executable_inputs(runtime: LayerRuntime) -> None:
    if git_provenance(runtime.paths.kquant_root) != runtime.preflight["kquant"]:
        raise ValueError("KQuant code changed inside a layer worker")
    if (
        exllamav3_provenance(runtime.paths.exllamav3_root)
        != runtime.preflight["exllamav3"]
    ):
        raise ValueError("ExLlamaV3 package/extension changed inside a layer worker")
    recovery = runtime.preflight.get("_local_code_recovery")
    expected_local = (
        recovery["new_local_pipeline_code"]
        if isinstance(recovery, Mapping)
        else runtime.preflight["local_pipeline_code"]
    )
    if (
        local_pipeline_code_provenance(Path(__file__).resolve().parents[1])
        != expected_local
    ):
        raise ValueError("local fresh-pipeline code changed inside a layer worker")
    extension_record = runtime.preflight["sqg_extension"]
    if (
        sha256_file(runtime.paths.sqg_extension_seal)
        != extension_record["seal_sha256"]
        or validate_sqg_extension_seal(
            runtime.paths.sqg_extension_seal,
            kquant_root=runtime.paths.kquant_root,
            require_runtime_environment=True,
        )
        != extension_record["seal"]
    ):
        raise ValueError("prebuilt SQG extension changed inside a layer worker")


def _load_bound_kquant_runtime(runtime: LayerRuntime):
    _revalidate_executable_inputs(runtime)
    loaded = load_kquant_runtime(
        runtime.paths.kquant_root, runtime.paths.exllamav3_root
    )
    quantizer = sys.modules.get("kquant.sqg_quantizer")
    if quantizer is None or not callable(getattr(quantizer, "_extension", None)):
        raise RuntimeError("KQuant SQG quantizer module did not load")
    sqg_module = quantizer._extension()
    sqg_expected = runtime.preflight["sqg_extension"]["seal"]["extension"]
    sqg_origin = Path(str(getattr(sqg_module, "__file__", ""))).resolve()
    if (
        str(sqg_origin) != sqg_expected["path"]
        or not sqg_origin.is_file()
        or sqg_origin.is_symlink()
        or sha256_file(sqg_origin) != sqg_expected["sha256"]
    ):
        raise RuntimeError("loaded KQuant SQG extension differs from the seal")
    extension = sys.modules.get("exllamav3_ext")
    origin = Path(str(getattr(extension, "__file__", ""))).resolve()
    expected = runtime.preflight["exllamav3"]["precompiled_extension"]
    if (
        extension is None
        or str(origin) != expected["origin"]
        or not origin.is_file()
        or sha256_file(origin) != expected["sha256"]
    ):
        raise RuntimeError("loaded ExLlamaV3 extension differs from the sealed binary")
    # The ExLlama extension is executable and may be imported during load.
    # Rehash immediately afterward before any official BF16 tensor is encoded.
    _revalidate_executable_inputs(runtime)
    return loaded


def _binding(runtime: LayerRuntime) -> dict[str, object]:
    return {
        "run_id": runtime.settings.run_id,
        "preflight_id": runtime.preflight["preflight_id"],
        "layer": runtime.layer,
        "source_seal_sha256": runtime.preflight["source_seal"]["sha256"],
        "capture_manifest_sha256": runtime.preflight["capture"]["sha256"],
        "bit_contract_sha256": runtime.preflight["bit_contract"]["sha256"],
        "kquant": runtime.preflight["kquant"],
        "sigma_reg": runtime.settings.sigma_reg,
    }


def _write_h13(runtime: LayerRuntime, h13: DenseHessian, evidence: Mapping[str, object]) -> None:
    path = runtime.preparation_dir / "h13.safetensors"
    manifest_path = runtime.preparation_dir / "h13.json"
    if path.exists() or manifest_path.exists():
        loaded, loaded_evidence = _load_h13(runtime)
        if tensor_sha256(loaded.matrix) != tensor_sha256(h13.matrix) or loaded_evidence != evidence:
            raise ValueError("existing H13 artifact differs")
        return
    payload_hashes, shard_hash = write_safetensors_atomic(
        path,
        [torch_tensor_entry("h13", h13.matrix)],
        metadata={
            "format": "pt",
            "schema": "glm52-fresh-sqg-h13-v1",
            "layer": str(runtime.layer),
            "role": "fit",
        },
    )
    value = {
        "schema": "glm52-fresh-sqg-h13-v1",
        "complete": True,
        "binding": _binding(runtime),
        "shard": path.name,
        "shard_sha256": shard_hash,
        "payload_sha256": payload_hashes,
        "dense_h": {
            "evidence_id": h13.evidence_id,
            "construction": h13.construction,
            "split_id": h13.split_id,
            "normalization_count": h13.normalization_count,
            "routed_sample_count": h13.routed_sample_count,
        },
        "evidence": dict(evidence),
    }
    atomic_json(manifest_path, value)


def _load_h13(runtime: LayerRuntime) -> tuple[DenseHessian, dict[str, Any]]:
    manifest = load_json_object(runtime.preparation_dir / "h13.json")
    if (
        manifest.get("schema") != "glm52-fresh-sqg-h13-v1"
        or manifest.get("complete") is not True
        or manifest.get("binding") != _binding(runtime)
    ):
        raise ValueError("H13 artifact binding differs")
    path = runtime.preparation_dir / manifest["shard"]
    if sha256_file(path) != manifest["shard_sha256"]:
        raise ValueError("H13 shard hash differs")
    with safe_open(path, framework="pt", device="cpu") as handle:
        matrix = handle.get_tensor("h13").float().contiguous()
    if tensor_sha256(matrix) != manifest["evidence"]["matrix_sha256"]:
        raise ValueError("H13 tensor hash differs")
    record = manifest["dense_h"]
    dense = DenseHessian(
        matrix=matrix,
        evidence_id=record["evidence_id"],
        construction=record["construction"],
        split_id=record["split_id"],
        normalization_count=int(record["normalization_count"]),
        routed_sample_count=int(record["routed_sample_count"]),
    )
    return dense, manifest["evidence"]


def _permutation_path(runtime: LayerRuntime, expert: int) -> Path:
    return runtime.preparation_dir / "permutations" / f"expert-{expert:03d}.json"


def _write_permutation(
    runtime: LayerRuntime,
    expert: int,
    permutation: FreshExpertPermutation,
    evidence: Mapping[str, object],
) -> None:
    value = {
        "schema": "glm52-fresh-sqg-expert-permutation-v1",
        "complete": True,
        "binding": _binding(runtime),
        "expert": expert,
        "new_to_old": permutation.new_to_old.tolist(),
        "manifest": permutation.manifest(),
        "evidence": dict(evidence),
    }
    value["artifact_id"] = canonical_sha256(value)
    atomic_json(_permutation_path(runtime, expert), value)


def _load_permutation(runtime: LayerRuntime, expert: int) -> FreshExpertPermutation:
    value = load_json_object(_permutation_path(runtime, expert))
    artifact_id = value.pop("artifact_id", None)
    observed = canonical_sha256(value)
    value["artifact_id"] = artifact_id
    if (
        artifact_id != observed
        or value.get("schema") != "glm52-fresh-sqg-expert-permutation-v1"
        or value.get("complete") is not True
        or value.get("binding") != _binding(runtime)
        or int(value.get("expert", -1)) != expert
    ):
        raise ValueError(f"expert {expert} permutation artifact binding differs")
    manifest = value["manifest"]
    result = FreshExpertPermutation(
        seed=int(manifest["seed"]),
        scope=str(manifest["scope"]),
        new_to_old=torch.tensor(value["new_to_old"], dtype=torch.int64),
        policy=str(manifest["policy"]),
        evidence_sha256=str(manifest["calibration_evidence_sha256"]),
        block_contexts_sha256=str(manifest["block_contexts_sha256"]),
        block_scores_sha256=str(manifest["block_scores_sha256"]),
        calibration_split_id=str(manifest["calibration_split_id"]),
    )
    if result.manifest() != manifest:
        raise ValueError(f"expert {expert} reconstructed permutation differs")
    return result


def _write_scale_evidence(runtime: LayerRuntime, scales: ProfileScaleEvidence) -> None:
    path = runtime.preparation_dir / "profile_scales.safetensors"
    manifest_path = runtime.preparation_dir / "profile_scales.json"
    if path.exists() or manifest_path.exists():
        existing = _load_scale_evidence(runtime)
        if existing.evidence != scales.evidence:
            raise ValueError("existing shared-profile scale evidence differs")
        return
    payload_hashes, shard_hash = write_safetensors_atomic(
        path,
        [
            torch_tensor_entry("gate_input_base", scales.gate_input_base),
            torch_tensor_entry("down_output_base", scales.down_output_base),
            torch_tensor_entry("gate_block_quarter", scales.gate_block_quarter),
            torch_tensor_entry("down_block_quarter", scales.down_block_quarter),
        ],
        metadata={
            "format": "pt",
            "schema": "glm52-fresh-sqg-profile-scales-v1",
            "layer": str(runtime.layer),
            "role": "fit",
            "legacy_scale_input": "false",
        },
    )
    atomic_json(
        manifest_path,
        {
            "schema": "glm52-fresh-sqg-profile-scales-v1",
            "complete": True,
            "binding": _binding(runtime),
            "shard": path.name,
            "shard_sha256": shard_hash,
            "payload_sha256": payload_hashes,
            "evidence": dict(scales.evidence),
        },
    )


def _load_scale_evidence(runtime: LayerRuntime) -> ProfileScaleEvidence:
    manifest = load_json_object(runtime.preparation_dir / "profile_scales.json")
    if (
        manifest.get("schema") != "glm52-fresh-sqg-profile-scales-v1"
        or manifest.get("complete") is not True
        or manifest.get("binding") != _binding(runtime)
    ):
        raise ValueError("shared-profile scale artifact binding differs")
    path = runtime.preparation_dir / manifest["shard"]
    if sha256_file(path) != manifest["shard_sha256"]:
        raise ValueError("shared-profile scale shard hash differs")
    with safe_open(path, framework="pt", device="cpu") as handle:
        values = {name: handle.get_tensor(name).float() for name in handle.keys()}
    return ProfileScaleEvidence(
        gate_input_base=values["gate_input_base"],
        down_output_base=values["down_output_base"],
        gate_block_quarter=values["gate_block_quarter"],
        down_block_quarter=values["down_block_quarter"],
        evidence=manifest["evidence"],
    )


def prepare_layer(runtime: LayerRuntime) -> dict[str, object]:
    """Build fit-only H13, all 256 h2_reverse maps, and scale families."""

    completion = runtime.preparation_dir / "preparation.json"
    if completion.exists():
        value = load_json_object(completion)
        if (
            value.get("schema") != LAYER_PREP_SCHEMA
            or value.get("complete") is not True
            or value.get("binding") != _binding(runtime)
        ):
            raise ValueError("layer preparation completion binding differs")
        _load_h13(runtime)
        _load_scale_evidence(runtime)
        for expert in range(NUM_EXPERTS):
            _load_permutation(runtime, expert)
        return value
    runtime.preparation_dir.mkdir(parents=True, exist_ok=True)
    if (runtime.preparation_dir / "h13.json").exists():
        h13, h13_evidence = _load_h13(runtime)
    else:
        h13, h13_evidence = build_layer_h13(
            runtime.capture,
            device=runtime.device,
            chunk_rows=runtime.settings.chunk_rows,
        )
        _write_h13(runtime, h13, h13_evidence)
    for expert in range(NUM_EXPERTS):
        path = _permutation_path(runtime, expert)
        if path.exists():
            _load_permutation(runtime, expert)
            continue
        weights = runtime.source.load_expert_bf16(runtime.layer, expert, device="cpu")
        permutation, evidence = derive_expert_permutation(
            runtime.capture,
            expert,
            weights.gate_hf,
            weights.up_hf,
            device=runtime.device,
            chunk_rows=runtime.settings.chunk_rows,
        )
        _write_permutation(runtime, expert, permutation, evidence)
        del weights
    if (runtime.preparation_dir / "profile_scales.json").exists():
        scales = _load_scale_evidence(runtime)
    else:
        scales = build_profile_scale_evidence(
            runtime.capture,
            runtime.source,
            h13,
            device=runtime.device,
        )
        _write_scale_evidence(runtime, scales)
    value = {
        "schema": LAYER_PREP_SCHEMA,
        "complete": True,
        "binding": _binding(runtime),
        "h13_evidence_id": h13.evidence_id,
        "h13_matrix_sha256": tensor_sha256(h13.matrix),
        "profile_scale_evidence_id": scales.evidence["evidence_id"],
        "permutation_count": NUM_EXPERTS,
        "permutation_policy": PRODUCTION_PERMUTATION_POLICY,
        "fit_only": True,
        "selection_data_used": False,
        "holdout_data_used": False,
        "fallback_count": 0,
        "mcg_inputs": 0,
    }
    atomic_json(completion, value)
    return value


def _projection_tensor(weights: object, projection: str) -> torch.Tensor:
    if projection == "gate_proj":
        return weights.gate_hf
    if projection == "up_proj":
        return weights.up_hf
    if projection == "down_proj":
        return weights.down_hf
    raise ValueError(f"unsupported projection {projection!r}")


def _source_binding(
    runtime: LayerRuntime,
    weights: object,
    projection: str,
) -> BF16TensorBinding:
    shard_name = str(weights.shard_names[projection])
    record = runtime.source.validation.shards[shard_name]
    return BF16TensorBinding(
        repository_id=SOURCE_REPO,
        revision=SOURCE_REVISION,
        shard_name=shard_name,
        shard_sha256=record.sha256,
        tensor_name=source_tensor_name(runtime.layer, weights.expert, projection),
        tensor_sha256=str(weights.tensor_sha256[projection]),
    )


def _role(projection: str) -> str:
    return {
        "gate_proj": "gate",
        "up_proj": "up",
        "down_proj": "down",
    }[projection]


def _config(
    runtime: LayerRuntime,
    weights: object,
    projection: str,
    permutation: FreshExpertPermutation,
    gate_profile: SharedResidualProfile,
    down_profile: SharedResidualProfile,
) -> UniformSQGConfig:
    expert = int(weights.expert)
    prefix = tensor_prefix(runtime.layer, expert, projection)
    return UniformSQGConfig(
        tensor_id=prefix,
        bits=int(runtime.bit_map[prefix]),
        matrix_role=_role(projection),
        transform_seed=derive_seed(
            runtime.settings.run_id,
            runtime.layer,
            expert,
            projection,
            "kquant-input-transform",
        ),
        output_sign_seed=derive_seed(
            runtime.settings.run_id,
            runtime.layer,
            expert,
            projection,
            "kquant-output-transform",
        ),
        source_binding=_source_binding(runtime, weights, projection),
        physical_permutation=permutation,
        shared_residual_profile=(
            gate_profile if projection in ("gate_proj", "up_proj") else down_profile
        ),
        sigma_reg=runtime.settings.sigma_reg,
        device=runtime.device,
        production=True,
    )


def _profiles_for_cell(
    runtime: LayerRuntime,
    scales: ProfileScaleEvidence,
    *,
    draw: int,
    family: str,
) -> tuple[SharedResidualProfile, SharedResidualProfile]:
    return make_shared_profiles(
        layer=runtime.layer,
        draw=draw,
        family=family,
        scales=scales,
        sign_factory=lambda length, layer, side, index: rademacher(
            length,
            runtime.settings.run_id,
            layer,
            side,
            index,
            "fresh-profile-sign-v1",
        ),
    )


def _cell_id(draw: int, family: str) -> str:
    if family not in PROFILE_FAMILIES:
        raise ValueError(f"unknown profile family {family!r}")
    return f"draw-{draw:02d}__{family}"


def _artifact_parts(directory: Path, layer: int, expert: int) -> tuple[Path, Path, Path]:
    stem = expert_stem(layer, expert)
    manifest = directory / f"{stem}.json"
    return manifest, directory / f"{stem}.safetensors", manifest.with_suffix(
        ".json.sha256"
    )


def _completed_prefix(
    runtime: LayerRuntime,
    directory: Path,
    order: Sequence[int],
    *,
    purpose: str,
    selection_evidence_sha256: str,
    gate_profile: SharedResidualProfile,
    down_profile: SharedResidualProfile,
) -> tuple[int, list[Mapping[str, object]], list[str]]:
    """Validate a deterministic expert prefix and return gate/up manifests."""

    prefix = 0
    gap = False
    prior: list[Mapping[str, object]] = []
    expected_ids: list[str] = []
    for expert in order:
        manifest_path, shard_path, seal_path = _artifact_parts(
            directory, runtime.layer, expert
        )
        exists = (manifest_path.exists(), shard_path.exists(), seal_path.exists())
        if any(exists) and not all(exists):
            raise ValueError(f"partial expert artifact exists: {manifest_path}")
        if not all(exists):
            gap = True
            continue
        if gap:
            raise ValueError("completed expert artifacts do not form the frozen order prefix")
        item = validate_expert_artifact(manifest_path)
        if (
            item.get("run_id") != runtime.settings.run_id
            or item.get("purpose") != purpose
            or item.get("selection_evidence_sha256")
            != selection_evidence_sha256
            or item.get("shared_profiles", {}).get("gate_up_input")
            != gate_profile.manifest()
            or item.get("shared_profiles", {}).get("down_output")
            != down_profile.manifest()
        ):
            raise ValueError(f"expert {expert} resumable artifact binding differs")
        for projection in ("gate_proj", "up_proj"):
            tensor_id = tensor_prefix(runtime.layer, expert, projection)
            prior.append(item["tensor_manifests"][tensor_id])
            expected_ids.append(tensor_id)
        prefix += 1
    return prefix, prior, expected_ids


def _encode_one_expert(
    runtime: LayerRuntime,
    kquant_runtime: object,
    h13: DenseHessian,
    h13_session: object,
    *,
    expert: int,
    output_dir: Path,
    purpose: str,
    selection_evidence_sha256: str,
    gate_profile: SharedResidualProfile,
    down_profile: SharedResidualProfile,
    cell_evidence: Mapping[str, object],
    cached_weights: object | None = None,
) -> dict[str, object]:
    """Encode gate/up, derive exact candidate H2, then encode down."""

    weights = (
        cached_weights
        if cached_weights is not None
        else runtime.source.load_expert_bf16(runtime.layer, expert, device="cpu")
    )
    if int(weights.expert) != expert:
        raise ValueError("cached BF16 expert binding differs")
    permutation = _load_permutation(runtime, expert)
    configs = {
        projection: _config(
            runtime,
            weights,
            projection,
            permutation,
            gate_profile,
            down_profile,
        )
        for projection in PROJECTIONS
    }
    gate = encode_uniform_sqg(
        weights.gate_hf,
        h13,
        configs["gate_proj"],
        runtime=kquant_runtime,
        dense_h_session=h13_session,
    )
    up = encode_uniform_sqg(
        weights.up_hf,
        h13,
        configs["up_proj"],
        runtime=kquant_runtime,
        dense_h_session=h13_session,
    )
    fit_routes = runtime.capture.routed_rows(expert, "fit")
    upstream = {
        "gate_tensor_manifest_sha256": canonical_sha256(gate.manifest),
        "up_tensor_manifest_sha256": canonical_sha256(up.manifest),
        "gate_decoded_exl_sha256": tensor_sha256(gate.reconstructed_exl),
        "up_decoded_exl_sha256": tensor_sha256(up.reconstructed_exl),
        "physical_permutation_sha256": permutation.sha256,
    }
    h2, h2_evidence = build_candidate_h2(
        runtime.capture,
        fit_routes,
        gate.reconstructed_exl,
        up.reconstructed_exl,
        device=runtime.device,
        upstream_evidence=upstream,
        chunk_rows=runtime.settings.chunk_rows,
    )
    down = encode_uniform_sqg(
        weights.down_hf,
        h2,
        configs["down_proj"],
        runtime=kquant_runtime,
    )
    encoded = {"gate_proj": gate, "up_proj": up, "down_proj": down}
    source_evidence = {
        "repository_id": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "tensor_payload_sha256": dict(sorted(weights.tensor_sha256.items())),
        "shard_names": dict(sorted(weights.shard_names.items())),
        "shard_sha256": {
            projection: runtime.source.validation.shards[
                weights.shard_names[projection]
            ].sha256
            for projection in PROJECTIONS
        },
    }
    calibration_evidence = {
        "capture": runtime.capture.binding(),
        "h13_evidence_id": h13.evidence_id,
        "h13_matrix_sha256": tensor_sha256(h13.matrix),
        "fit_routes": fit_routes.evidence(),
        "permutation_artifact_sha256": sha256_file(
            _permutation_path(runtime, expert)
        ),
        "profile_cell": dict(cell_evidence),
        "fit_only": True,
        "selection_used_for_encoding_calibration": False,
        "holdout_used": False,
        "fallback_allowed": False,
    }
    return write_expert_artifact(
        output_dir,
        run_id=runtime.settings.run_id,
        layer=runtime.layer,
        expert=expert,
        encoded=encoded,
        permutation=permutation,
        gate_up_profile=gate_profile,
        down_profile=down_profile,
        bit_map=runtime.bit_map,
        h2_evidence=h2_evidence,
        calibration_evidence=calibration_evidence,
        source_evidence=source_evidence,
        selection_evidence_sha256=selection_evidence_sha256,
        purpose=purpose,
    )


def _encode_expert_sequence(
    runtime: LayerRuntime,
    kquant_runtime: object,
    h13: DenseHessian,
    *,
    order: Sequence[int],
    output_dir: Path,
    purpose: str,
    selection_evidence_sha256: str,
    gate_profile: SharedResidualProfile,
    down_profile: SharedResidualProfile,
    cell_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Encode a deterministic expert sequence with exact H13 session resume."""

    _revalidate_executable_inputs(runtime)
    ordered = tuple(int(value) for value in order)
    prefix, prior, expected_ids = _completed_prefix(
        runtime,
        output_dir,
        ordered,
        purpose=purpose,
        selection_evidence_sha256=selection_evidence_sha256,
        gate_profile=gate_profile,
        down_profile=down_profile,
    )
    resume_evidence: Mapping[str, object] | None = None
    if prefix < len(ordered):
        first_expert = ordered[prefix]
        first_weights = runtime.source.load_expert_bf16(
            runtime.layer, first_expert, device="cpu"
        )
        first_permutation = _load_permutation(runtime, first_expert)
        first_config = _config(
            runtime,
            first_weights,
            "gate_proj",
            first_permutation,
            gate_profile,
            down_profile,
        )
        session = prepare_dense_h_session(h13, first_config)
        if prior:
            resume_evidence = resume_dense_h_session(
                session,
                h13,
                first_config,
                kquant_runtime,
                prior_tensor_manifests=prior,
                expected_tensor_ids=expected_ids,
            )
        for index, expert in enumerate(ordered[prefix:], start=prefix):
            _encode_one_expert(
                runtime,
                kquant_runtime,
                h13,
                session,
                expert=expert,
                output_dir=output_dir,
                purpose=purpose,
                selection_evidence_sha256=selection_evidence_sha256,
                gate_profile=gate_profile,
                down_profile=down_profile,
                cell_evidence=cell_evidence,
                cached_weights=first_weights if index == prefix else None,
            )
            atomic_json(
                output_dir / "progress.json",
                {
                    "schema": "glm52-fresh-sqg-expert-sequence-progress-v1",
                    "binding": _binding(runtime),
                    "purpose": purpose,
                    "selection_evidence_sha256": selection_evidence_sha256,
                    "expert_order": list(ordered),
                    "completed_prefix": index + 1,
                    "h13_session_id": session.session_id,
                    "resume_evidence": resume_evidence,
                },
                overwrite=True,
            )
    final_prefix, final_prior, final_ids = _completed_prefix(
        runtime,
        output_dir,
        ordered,
        purpose=purpose,
        selection_evidence_sha256=selection_evidence_sha256,
        gate_profile=gate_profile,
        down_profile=down_profile,
    )
    if final_prefix != len(ordered) or len(final_prior) != 2 * len(ordered):
        raise RuntimeError("expert sequence did not reach its frozen completion")
    return {
        "expert_order": list(ordered),
        "expert_count": len(ordered),
        "gate_up_tensor_ids": final_ids,
        "resume_evidence": resume_evidence,
        "complete": True,
    }


def _validate_id(value: Mapping[str, object], field: str) -> None:
    copy = dict(value)
    expected = copy.pop(field, None)
    if expected != canonical_sha256(copy):
        raise ValueError(f"canonical {field} binding differs")


def _build_profile_preregistration(
    runtime: LayerRuntime,
    scales: ProfileScaleEvidence,
    h13: DenseHessian,
) -> dict[str, object]:
    panel = mass_stratified_experts(
        runtime.capture.role_gate_square_mass_by_expert("fit"),
        runtime.settings.panel_experts,
    )
    cells: list[dict[str, object]] = []
    for draw in range(runtime.settings.profile_draws):
        for family in PROFILE_FAMILIES:
            gate, down = _profiles_for_cell(
                runtime, scales, draw=draw, family=family
            )
            cells.append(
                {
                    "cell_id": _cell_id(draw, family),
                    "draw": draw,
                    "family": family,
                    "gate_up_input_profile": gate.manifest(),
                    "down_output_profile": down.manifest(),
                }
            )
    value: dict[str, object] = {
        "schema": "glm52-fresh-sqg-profile-factorial-preregistration-v1",
        "complete": True,
        "binding": _binding(runtime),
        "factorial": {
            "fresh_sign_draws": runtime.settings.profile_draws,
            "magnitude_families": list(PROFILE_FAMILIES),
            "expected_cells": runtime.settings.profile_draws
            * len(PROFILE_FAMILIES),
            "proxy_pruning_allowed": False,
            "every_cell_exact_sqg": True,
            "every_cell_candidate_conditional_h2": True,
            "every_cell_exact_down_sqg": True,
            "every_cell_routed_selection_score": True,
        },
        "selection_panel": list(panel),
        "selection_panel_construction": "fit_gate_square_mass_stratified_v1",
        "selection_panel_count": len(panel),
        "cells": cells,
        "baseline_cell_id": _cell_id(0, "identity"),
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "per_comparison_confidence": BONFERRONI_CONFIDENCE,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "comparisons": PROFILE_COMPARISONS,
            "multiplicity_correction": "bonferroni",
            "unit": "document_epoch",
            "paired": True,
            "criterion": (
                "simultaneous_bonferroni_lower_bound_gt_zero_vs_"
                "identity_draw_00"
            ),
        },
        "h13_evidence_id": h13.evidence_id,
        "profile_scale_evidence_id": scales.evidence["evidence_id"],
        "fit_build_selection_choose_holdout_report": True,
        "fit_used_for_panel_construction": True,
        "selection_used_for_panel_construction": False,
        "holdout_available_to_selection": False,
        "mcg_inputs": 0,
    }
    value["preregistration_id"] = canonical_sha256(value)
    return value


def _write_or_validate_preregistration(
    runtime: LayerRuntime,
    scales: ProfileScaleEvidence,
    h13: DenseHessian,
) -> dict[str, object]:
    path = runtime.profile_dir / "preregistration.json"
    expected = _build_profile_preregistration(runtime, scales, h13)
    if path.exists():
        observed = load_json_object(path)
        _validate_id(observed, "preregistration_id")
        if observed != expected:
            raise ValueError("profile factorial preregistration differs")
        return observed
    runtime.profile_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(path, expected)
    return expected


def _validate_score_file(
    runtime: LayerRuntime,
    path: Path,
    *,
    panel: Sequence[int],
    artifact_dir: Path,
) -> dict[str, Any]:
    score = load_json_object(path)
    _validate_id(score, "score_id")
    if (
        score.get("complete") is not True
        or score.get("layer") != runtime.layer
        or score.get("role") != "selection"
        or score.get("expert_panel") != list(panel)
        or score.get("expert_count") != len(panel)
    ):
        raise ValueError(f"selection score binding differs: {path}")
    expected_hashes = {
        str(expert): sha256_file(
            artifact_dir / f"{expert_stem(runtime.layer, expert)}.json"
        )
        for expert in panel
    }
    if score.get("binding", {}).get("artifact_manifest_sha256") != expected_hashes:
        raise ValueError(f"selection score artifact hashes differ: {path}")
    return score


def _write_selected_profiles(
    runtime: LayerRuntime,
    gate: SharedResidualProfile,
    down: SharedResidualProfile,
) -> dict[str, object]:
    path = runtime.profile_dir / "selected_profiles.safetensors"
    entries = [
        torch_tensor_entry("gate_up_input_signs", gate.signs.float()),
        torch_tensor_entry(
            "gate_up_input_channel_scales", gate.channel_scales.float()
        ),
        torch_tensor_entry("down_output_signs", down.signs.float()),
        torch_tensor_entry(
            "down_output_channel_scales", down.channel_scales.float()
        ),
    ]
    if gate.stored_fp16_override is not None:
        entries.append(
            torch_tensor_entry(
                "gate_up_input_stored_fp16_override",
                gate.stored_fp16_override,
            )
        )
    if down.stored_fp16_override is not None:
        entries.append(
            torch_tensor_entry(
                "down_output_stored_fp16_override",
                down.stored_fp16_override,
            )
        )
    payload_hashes, shard_hash = write_safetensors_atomic(
        path,
        entries,
        metadata={
            "format": "pt",
            "schema": "glm52-fresh-sqg-selected-profiles-v1",
            "layer": str(runtime.layer),
            "legacy_scale_input": "false",
        },
    )
    return {
        "shard": path.name,
        "shard_sha256": shard_hash,
        "payload_sha256": payload_hashes,
    }


def _load_profile_selection(
    runtime: LayerRuntime,
) -> tuple[dict[str, Any], SharedResidualProfile, SharedResidualProfile]:
    path = runtime.profile_dir / "selection.json"
    value = load_json_object(path)
    _validate_id(value, "selection_id")
    if (
        value.get("schema") != PROFILE_SEARCH_SCHEMA
        or value.get("complete") is not True
        or value.get("binding") != _binding(runtime)
        or value.get("holdout_used") is not False
        or value.get("proxy_pruning_used") is not False
    ):
        raise ValueError("selected profile binding differs")
    shard_record = value["selected_profile_shard"]
    shard = runtime.profile_dir / shard_record["shard"]
    if sha256_file(shard) != shard_record["shard_sha256"]:
        raise ValueError("selected profile shard hash differs")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    gate_manifest = value["selected_profiles"]["gate_up_input"]
    down_manifest = value["selected_profiles"]["down_output"]
    gate_override = tensors.get("gate_up_input_stored_fp16_override")
    down_override = tensors.get("down_output_stored_fp16_override")
    if (gate_override is None) != (
        gate_manifest.get("stored_fp16_realization") is None
    ) or (down_override is None) != (
        down_manifest.get("stored_fp16_realization") is None
    ):
        raise ValueError("selected profile realization payload domain differs")
    gate = SharedResidualProfile(
        side="input",
        signs=tensors["gate_up_input_signs"].float(),
        channel_scales=tensors["gate_up_input_channel_scales"].float(),
        profile_id=gate_manifest["profile_id"],
        derivation=gate_manifest["derivation"],
        stored_fp16_override=(
            gate_override.contiguous() if gate_override is not None else None
        ),
        stored_fp16_realization=gate_manifest.get("stored_fp16_realization"),
    )
    down = SharedResidualProfile(
        side="output",
        signs=tensors["down_output_signs"].float(),
        channel_scales=tensors["down_output_channel_scales"].float(),
        profile_id=down_manifest["profile_id"],
        derivation=down_manifest["derivation"],
        stored_fp16_override=(
            down_override.contiguous() if down_override is not None else None
        ),
        stored_fp16_realization=down_manifest.get("stored_fp16_realization"),
    )
    if gate.manifest() != gate_manifest or down.manifest() != down_manifest:
        raise ValueError("selected profile reconstruction differs")
    score_hashes = value.get("cell_score_sha256", {})
    expected_cells = ACTIVE_PROFILE_DRAWS * len(PROFILE_FAMILIES)
    if len(score_hashes) != expected_cells:
        raise ValueError("selected profile lacks the frozen factorial-prefix hashes")
    scope = value.get("selection_scope", {})
    if (
        scope.get("completed_draw_prefix") != ACTIVE_PROFILE_DRAWS
        or scope.get("selected_cell_count") != expected_cells
        or scope.get("preregistered_draws") != runtime.settings.profile_draws
        or scope.get("preregistered_cell_count")
        != runtime.settings.profile_draws * len(PROFILE_FAMILIES)
        or scope.get("operator_time_priority") is not True
        or scope.get("unscored_cells_excluded")
        != (runtime.settings.profile_draws - ACTIVE_PROFILE_DRAWS)
        * len(PROFILE_FAMILIES)
    ):
        raise ValueError("selected profile frozen-prefix scope differs")
    for cell_id, digest in score_hashes.items():
        score_path = runtime.profile_dir / "cells" / cell_id / "score.json"
        if sha256_file(score_path) != digest:
            raise ValueError(f"profile cell score changed: {cell_id}")
    return value, gate, down


def search_profiles(runtime: LayerRuntime) -> dict[str, object]:
    """Select from the completed exact-SQG four-draw factorial prefix."""

    selection_path = runtime.profile_dir / "selection.json"
    if selection_path.exists():
        return _load_profile_selection(runtime)[0]
    prepare_layer(runtime)
    h13, _ = _load_h13(runtime)
    scales = _load_scale_evidence(runtime)
    prereg = _write_or_validate_preregistration(runtime, scales, h13)
    prereg_hash = sha256_file(runtime.profile_dir / "preregistration.json")
    panel = tuple(int(value) for value in prereg["selection_panel"])
    kquant_runtime = _load_bound_kquant_runtime(runtime)
    active_cells = [
        cell
        for cell in prereg["cells"]
        if int(cell["draw"]) < ACTIVE_PROFILE_DRAWS
    ]
    scores: dict[str, dict[str, Any]] = {}
    for cell in active_cells:
        cell_id = str(cell["cell_id"])
        draw = int(cell["draw"])
        family = str(cell["family"])
        gate, down = _profiles_for_cell(
            runtime, scales, draw=draw, family=family
        )
        if (
            gate.manifest() != cell["gate_up_input_profile"]
            or down.manifest() != cell["down_output_profile"]
        ):
            raise RuntimeError(f"preregistered profile reconstruction differs: {cell_id}")
        gate = realize_shared_residual_profile(gate, runtime.device)
        down = realize_shared_residual_profile(down, runtime.device)
        cell_root = runtime.profile_dir / "cells" / cell_id
        artifact_dir = cell_root / "experts"
        cell_evidence = {
            "preregistration_id": prereg["preregistration_id"],
            "cell_id": cell_id,
            "draw": draw,
            "family": family,
            "selection_panel": list(panel),
            "profile_choice_not_yet_made": True,
        }
        _encode_expert_sequence(
            runtime,
            kquant_runtime,
            h13,
            order=panel,
            output_dir=artifact_dir,
            purpose="profile_search_cell",
            selection_evidence_sha256=prereg_hash,
            gate_profile=gate,
            down_profile=down,
            cell_evidence=cell_evidence,
        )
        score_path = cell_root / "score.json"
        if score_path.exists():
            score = _validate_score_file(
                runtime,
                score_path,
                panel=panel,
                artifact_dir=artifact_dir,
            )
        else:
            score = score_routed_artifacts(
                runtime.capture,
                runtime.source,
                kquant_runtime,
                artifact_dir,
                cell_root / "score_scratch",
                role="selection",
                experts=panel,
                device=runtime.device,
                chunk_rows=runtime.settings.chunk_rows,
                cleanup_scratch=True,
            )
            score["factorial_cell"] = dict(cell)
            # score_id covers the cell annotation as well as the metric.
            score.pop("score_id", None)
            score["score_id"] = canonical_sha256(score)
            atomic_json(score_path, score)
            score = _validate_score_file(
                runtime,
                score_path,
                panel=panel,
                artifact_dir=artifact_dir,
            )
        scores[cell_id] = score
    expected_ids = {str(cell["cell_id"]) for cell in active_cells}
    if set(scores) != expected_ids or len(scores) != (
        ACTIVE_PROFILE_DRAWS * len(PROFILE_FAMILIES)
    ):
        raise RuntimeError("profile search lacks one or more frozen-prefix cells")

    baseline_id = str(prereg["baseline_cell_id"])
    baseline = scores[baseline_id]
    bootstraps: dict[str, dict[str, object]] = {}
    for cell_id in sorted(scores):
        if cell_id == baseline_id:
            bootstraps[cell_id] = {
                "schema": "glm52-fresh-sqg-paired-document-bootstrap-control-v1",
                "baseline_self_comparison": True,
                "lower_bound_gt_zero": False,
                "observed_mean_improvement": 0.0,
                "lower_bound": 0.0,
                "upper_bound": 0.0,
            }
        else:
            bootstraps[cell_id] = paired_document_bootstrap(
                baseline,
                scores[cell_id],
                seed=derive_seed(
                    runtime.settings.run_id,
                    runtime.layer,
                    baseline_id,
                    cell_id,
                    "paired-document-bootstrap",
                ),
                iterations=BOOTSTRAP_ITERATIONS,
                confidence=BONFERRONI_CONFIDENCE,
            )
    ranked = sorted(
        scores,
        key=lambda cell_id: (
            float(scores[cell_id]["aggregate_relative_error"]),
            cell_id,
        ),
    )
    eligible = [
        cell_id
        for cell_id in ranked
        if cell_id != baseline_id
        and bool(bootstraps[cell_id]["lower_bound_gt_zero"])
        and float(scores[cell_id]["aggregate_relative_error"])
        < float(baseline["aggregate_relative_error"])
    ]
    selected_id = eligible[0] if eligible else baseline_id
    selected_cell = next(
        cell for cell in prereg["cells"] if cell["cell_id"] == selected_id
    )
    selected_gate, selected_down = _profiles_for_cell(
        runtime,
        scales,
        draw=int(selected_cell["draw"]),
        family=str(selected_cell["family"]),
    )
    selected_gate = realize_shared_residual_profile(
        selected_gate, runtime.device
    )
    selected_down = realize_shared_residual_profile(
        selected_down, runtime.device
    )
    shard_record = _write_selected_profiles(
        runtime, selected_gate, selected_down
    )
    selection: dict[str, object] = {
        "schema": PROFILE_SEARCH_SCHEMA,
        "complete": True,
        "binding": _binding(runtime),
        "preregistration_sha256": prereg_hash,
        "preregistration_id": prereg["preregistration_id"],
        "factorial_cell_count": len(scores),
        "required_factorial_cell_count": ACTIVE_PROFILE_DRAWS
        * len(PROFILE_FAMILIES),
        "preregistered_factorial_cell_count": runtime.settings.profile_draws
        * len(PROFILE_FAMILIES),
        "selection_scope": {
            "completed_draw_prefix": ACTIVE_PROFILE_DRAWS,
            "selected_cell_count": len(scores),
            "preregistered_draws": runtime.settings.profile_draws,
            "preregistered_cell_count": runtime.settings.profile_draws
            * len(PROFILE_FAMILIES),
            "operator_time_priority": True,
            "unscored_cells_excluded": (
                runtime.settings.profile_draws - ACTIVE_PROFILE_DRAWS
            )
            * len(PROFILE_FAMILIES),
            "all_selected_cells_exact_sqg": True,
            "selection_math_uses_original_31_comparison_bonferroni": True,
        },
        "all_cells_exact_sqg": True,
        "all_cells_candidate_conditional_h2": True,
        "all_cells_exact_down_sqg": True,
        "all_cells_routed_selection_scored": True,
        "proxy_pruning_used": False,
        "selection_panel": list(panel),
        "baseline_cell_id": baseline_id,
        "selected_cell_id": selected_id,
        "selected_cell": dict(selected_cell),
        "selected_by": (
            "lowest_aggregate_relative_error_with_positive_familywise_95pct_"
            "bonferroni_paired_document_bootstrap_lower_bound_vs_identity_"
            "draw_00_else_baseline"
        ),
        "multiplicity_control": {
            "method": "bonferroni",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "comparisons": PROFILE_COMPARISONS,
            "per_comparison_confidence": BONFERRONI_CONFIDENCE,
        },
        "ranked_cell_ids": ranked,
        "cell_metrics": {
            cell_id: {
                "score_id": score["score_id"],
                "aggregate_relative_error": score["aggregate_relative_error"],
                "mean_document_relative_error": score[
                    "mean_document_relative_error"
                ],
                "bootstrap_vs_baseline": bootstraps[cell_id],
            }
            for cell_id, score in sorted(scores.items())
        },
        "cell_score_sha256": {
            cell_id: sha256_file(
                runtime.profile_dir / "cells" / cell_id / "score.json"
            )
            for cell_id in sorted(scores)
        },
        "selected_profiles": {
            "gate_up_input": selected_gate.manifest(),
            "down_output": selected_down.manifest(),
        },
        "selected_profile_shard": shard_record,
        "fit_used_for_profile_construction": True,
        "selection_used_once_for_choice": True,
        "holdout_used": False,
        "selection_frozen_before_holdout": True,
        "fallback_count": 0,
        "mcg_inputs": 0,
    }
    selection["selection_id"] = canonical_sha256(selection)
    atomic_json(selection_path, selection)
    return _load_profile_selection(runtime)[0]


def encode_layer(runtime: LayerRuntime) -> dict[str, object]:
    """Encode all 768 layer tensors under the frozen selected fresh profile."""

    search_profiles(runtime)
    selection, gate, down = _load_profile_selection(runtime)
    selection_path = runtime.profile_dir / "selection.json"
    selection_hash = sha256_file(selection_path)
    h13, _ = _load_h13(runtime)
    kquant_runtime = _load_bound_kquant_runtime(runtime)
    sequence = _encode_expert_sequence(
        runtime,
        kquant_runtime,
        h13,
        order=tuple(range(NUM_EXPERTS)),
        output_dir=runtime.expert_dir,
        purpose="final_treatment",
        selection_evidence_sha256=selection_hash,
        gate_profile=gate,
        down_profile=down,
        cell_evidence={
            "selection_id": selection["selection_id"],
            "selection_sha256": selection_hash,
            "selected_cell_id": selection["selected_cell_id"],
            "profile_frozen": True,
            "holdout_unseen": True,
        },
    )
    assemble_layer_artifact(
        runtime.expert_dir,
        runtime.final_dir,
        run_id=runtime.settings.run_id,
        layer=runtime.layer,
        bit_map=runtime.bit_map,
        gate_up_profile=gate,
        down_profile=down,
        layer_evidence={
            "pipeline": _binding(runtime),
            "preparation_sha256": sha256_file(
                runtime.preparation_dir / "preparation.json"
            ),
            "profile_selection_sha256": selection_hash,
            "profile_selection_id": selection["selection_id"],
            "selected_cell_id": selection["selected_cell_id"],
            "expert_sequence": sequence,
            "fit_calibration_complete": True,
            "selection_choice_frozen": True,
            "holdout_used": False,
            "fallback_count": 0,
            "mcg_inputs": 0,
        },
    )
    return validate_layer_artifact(
        runtime.final_dir / f"fresh-sqg-layer-{runtime.layer:03d}.json"
    )


def _validate_holdout_report(
    runtime: LayerRuntime,
    path: Path,
) -> dict[str, Any]:
    value = load_json_object(path)
    _validate_id(value, "report_id")
    selection_path = runtime.profile_dir / "selection.json"
    layer_path = runtime.final_dir / f"fresh-sqg-layer-{runtime.layer:03d}.json"
    if (
        value.get("schema") != HOLDOUT_REPORT_SCHEMA
        or value.get("complete") is not True
        or value.get("binding") != _binding(runtime)
        or value.get("role") != "holdout"
        or value.get("selection_sha256") != sha256_file(selection_path)
        or value.get("layer_artifact_sha256") != sha256_file(layer_path)
        or value.get("selection_frozen_before_holdout") is not True
        or value.get("artifact_changed_by_holdout") is not False
    ):
        raise ValueError("holdout report binding differs")
    score = value.get("routed_functional_score")
    if not isinstance(score, Mapping):
        raise ValueError("holdout report lacks routed functional score")
    _validate_id(score, "score_id")
    if score.get("role") != "holdout" or score.get("expert_count") != NUM_EXPERTS:
        raise ValueError("holdout routed score coverage differs")
    return value


def evaluate_holdout(runtime: LayerRuntime) -> dict[str, object]:
    """Report full routed holdout loss after artifacts and profiles are frozen."""

    report_path = runtime.final_dir / "holdout.json"
    if report_path.exists():
        return _validate_holdout_report(runtime, report_path)
    encode_layer(runtime)
    selection_path = runtime.profile_dir / "selection.json"
    layer_path = runtime.final_dir / f"fresh-sqg-layer-{runtime.layer:03d}.json"
    selection_hash = sha256_file(selection_path)
    layer_hash = sha256_file(layer_path)
    kquant_runtime = _load_bound_kquant_runtime(runtime)
    score = score_routed_artifacts(
        runtime.capture,
        runtime.source,
        kquant_runtime,
        runtime.expert_dir,
        runtime.final_dir / "holdout_scratch",
        role="holdout",
        experts=tuple(range(NUM_EXPERTS)),
        device=runtime.device,
        chunk_rows=runtime.settings.chunk_rows,
        cleanup_scratch=False,
    )
    # Recheck immutable decision/artifact identities after the long replay.
    if sha256_file(selection_path) != selection_hash or sha256_file(layer_path) != layer_hash:
        raise RuntimeError("selected profile or layer artifact changed during holdout")
    report: dict[str, object] = {
        "schema": HOLDOUT_REPORT_SCHEMA,
        "complete": True,
        "binding": _binding(runtime),
        "role": "holdout",
        "selection_sha256": selection_hash,
        "layer_artifact_sha256": layer_hash,
        "selection_frozen_before_holdout": True,
        "artifact_changed_by_holdout": False,
        "holdout_used_for_calibration": False,
        "holdout_used_for_profile_choice": False,
        "holdout_report_only": True,
        "routed_functional_score": score,
        "fallback_count": 0,
        "mcg_inputs": 0,
    }
    report["report_id"] = canonical_sha256(report)
    atomic_json(report_path, report)
    return _validate_holdout_report(runtime, report_path)


def run_layer(runtime: LayerRuntime) -> dict[str, object]:
    """Run one resumable, one-GPU layer job through report-only holdout."""

    preparation = prepare_layer(runtime)
    selection = search_profiles(runtime)
    layer = encode_layer(runtime)
    holdout = evaluate_holdout(runtime)
    completion: dict[str, object] = {
        "schema": "glm52-fresh-sqg-layer-job-v1",
        "complete": True,
        "binding": _binding(runtime),
        "preparation_sha256": sha256_file(
            runtime.preparation_dir / "preparation.json"
        ),
        "selection_sha256": sha256_file(runtime.profile_dir / "selection.json"),
        "layer_artifact_sha256": sha256_file(
            runtime.final_dir / f"fresh-sqg-layer-{runtime.layer:03d}.json"
        ),
        "holdout_sha256": sha256_file(runtime.final_dir / "holdout.json"),
        "ids": {
            "h13": preparation["h13_evidence_id"],
            "selection": selection["selection_id"],
            "layer_run": layer["fresh_sqg_run_manifest"]["run_id"],
            "holdout": holdout["report_id"],
        },
        "production_container_touched": False,
        "existing_model_mutated": False,
        "mcg_inputs": 0,
    }
    completion["job_id"] = canonical_sha256(completion)
    atomic_json(runtime.layer_root / "layer_job.json", completion)
    return completion


def _time_priority_selection_receipt(
    selection: Mapping[str, object],
    settings: PipelineSettings,
) -> dict[str, object] | None:
    """Return the explicit no-holdout receipt for the frozen four-draw run."""

    selected_cells = ACTIVE_PROFILE_DRAWS * len(PROFILE_FAMILIES)
    preregistered_cells = settings.profile_draws * len(PROFILE_FAMILIES)
    expected_scope = {
        "completed_draw_prefix": ACTIVE_PROFILE_DRAWS,
        "selected_cell_count": selected_cells,
        "preregistered_draws": settings.profile_draws,
        "preregistered_cell_count": preregistered_cells,
        "operator_time_priority": True,
        "unscored_cells_excluded": preregistered_cells - selected_cells,
        "all_selected_cells_exact_sqg": True,
        "selection_math_uses_original_31_comparison_bonferroni": True,
    }
    if (
        selection.get("factorial_cell_count") != selected_cells
        or selection.get("required_factorial_cell_count") != selected_cells
        or selection.get("preregistered_factorial_cell_count")
        != preregistered_cells
        or selection.get("selection_scope") != expected_scope
    ):
        return None
    return {
        "status": "skipped",
        "reason": "operator_time_priority",
        "report_present": False,
        "routed_score_present": False,
        "used_for_calibration": False,
        "used_for_profile_choice": False,
    }


def seal_run(preflight_path: str | Path) -> dict[str, object]:
    """Seal four completed layers; this does not materialize a runnable model."""

    preflight = load_preflight(preflight_path)
    paths = PipelinePaths.resolve(**preflight["paths"])
    settings = PipelineSettings(**preflight["settings"])
    layers: dict[str, object] = {}
    total_k3 = 0
    total_k4 = 0
    total_sqg = 0
    seal_modes: set[str] = set()
    for layer in SELECTED_LAYERS:
        layer_root = paths.output_root / f"layer_{layer:03d}"
        layer_path = layer_root / "final" / f"fresh-sqg-layer-{layer:03d}.json"
        manifest = validate_layer_artifact(layer_path)
        selection_path = layer_root / "profile_search" / "selection.json"
        selection = load_json_object(selection_path)
        _validate_id(selection, "selection_id")
        selection_sha256 = sha256_file(selection_path)
        layer_sha256 = sha256_file(layer_path)
        skipped_holdout = _time_priority_selection_receipt(selection, settings)
        legacy_holdout = (
            selection.get("factorial_cell_count") == 32
            and selection.get("required_factorial_cell_count") == 32
            and skipped_holdout is None
        )
        if (
            manifest.get("run_id") != settings.run_id
            or selection.get("proxy_pruning_used") is not False
            or selection.get("holdout_used") is not False
            or selection.get("all_cells_exact_sqg") is not True
            or selection.get("all_cells_routed_selection_scored") is not True
            or (not legacy_holdout and skipped_holdout is None)
        ):
            raise ValueError(f"layer {layer} completion evidence differs")
        holdout_path = layer_root / "final" / "holdout.json"
        if skipped_holdout is not None:
            if holdout_path.exists() or holdout_path.is_symlink():
                raise ValueError(
                    f"layer {layer} claims a skipped routed holdout but a report exists"
                )
            seal_modes.add("time_priority_no_holdout")
            holdout_record: dict[str, object] = {
                **skipped_holdout,
                "selection_sha256": selection_sha256,
                "layer_artifact_sha256": layer_sha256,
            }
        else:
            holdout = load_json_object(holdout_path)
            _validate_id(holdout, "report_id")
            if (
                holdout.get("selection_sha256") != selection_sha256
                or holdout.get("layer_artifact_sha256") != layer_sha256
            ):
                raise ValueError(f"layer {layer} holdout evidence differs")
            seal_modes.add("routed_holdout_complete")
            holdout_record = {
                "holdout_sha256": sha256_file(holdout_path),
                "holdout_report_id": holdout["report_id"],
            }
        histogram = manifest["bit_histogram"]
        total_k3 += int(histogram["3"])
        total_k4 += int(histogram["4"])
        total_sqg += int(manifest["lineage"]["sqg_tensor_count"])
        layers[str(layer)] = {
            "layer_artifact": str(layer_path),
            "layer_artifact_sha256": layer_sha256,
            "layer_shard_sha256": manifest["shard_sha256"],
            "selection_sha256": selection_sha256,
            "selection_id": selection["selection_id"],
            "selected_cell_id": selection["selected_cell_id"],
            "sqg_tensors": manifest["lineage"]["sqg_tensor_count"],
            "mcg_tensors": manifest["lineage"]["mcg_tensor_count"],
            "K3": histogram["3"],
            "K4": histogram["4"],
            **(
                {"holdout": holdout_record}
                if skipped_holdout is not None
                else holdout_record
            ),
        }
    if (total_k3, total_k4, total_sqg) != (1_536, 1_536, 3_072):
        raise RuntimeError("four-layer SQG/bit census differs")
    if len(seal_modes) != 1:
        raise RuntimeError("four-layer routed-holdout policy differs by layer")
    time_priority = seal_modes == {"time_priority_no_holdout"}
    seal: dict[str, object] = {
        "schema": RUN_SEAL_SCHEMA,
        "complete": True,
        "run_id": settings.run_id,
        "preflight_sha256": sha256_file(preflight_path),
        "layers": layers,
        "census": {
            "layers": 4,
            "experts": 1_024,
            "sqg_tensors": 3_072,
            "mcg_tensors": 0,
            "K3": total_k3,
            "K4": total_k4,
            "other_rates": 0,
            "factorial_cells_per_layer": 16 if time_priority else 32,
            "exact_factorial_cells_total": 64 if time_priority else 128,
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
            **(
                {
                    "routed_holdout": {
                        "status": "skipped",
                        "reason": "operator_time_priority",
                        "layers": list(SELECTED_LAYERS),
                        "report_count": 0,
                        "routed_score_count": 0,
                    }
                }
                if time_priority
                else {}
            ),
        },
    }
    seal["run_seal_id"] = canonical_sha256(seal)
    atomic_json(paths.output_root / "run_seal.json", seal)
    return seal


def build_job_plan(
    preflight_path: str | Path,
    *,
    python_executable: str = "python3",
) -> dict[str, object]:
    """Write four inert one-layer/GPU command records; execute nothing."""

    preflight = load_preflight(preflight_path)
    paths = PipelinePaths.resolve(**preflight["paths"])
    jobs: list[dict[str, object]] = []
    for gpu, layer in enumerate(SELECTED_LAYERS):
        pythonpath = preflight["exllamav3"]["execution_environment"]["PYTHONPATH"]
        jobs.append(
            {
                "job_id": f"layer-{layer:03d}-gpu-{gpu}",
                "layer": layer,
                "visible_physical_gpu": gpu,
                "working_directory": str(Path(__file__).resolve().parents[1]),
                "environment": {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONPATH": pythonpath,
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "safe.directory",
                    "GIT_CONFIG_VALUE_0": str(paths.kquant_root),
                    "FRESH_SQG_RUNTIME_IMAGE_ID": R33_IMAGE_ID,
                    "KQUANT_SQG_REQUIRE_PREBUILT": "1",
                    "KQUANT_SQG_EXTENSION_PATH": preflight["sqg_extension"][
                        "seal"
                    ]["extension"]["path"],
                    "KQUANT_SQG_EXTENSION_SHA256": preflight["sqg_extension"][
                        "seal"
                    ]["extension"]["sha256"],
                    "TORCH_CUDA_ARCH_LIST": "12.0",
                },
                "argv": [
                    python_executable,
                    "-m",
                    "src.run_fresh_sqg",
                    "run-layer",
                    "--preflight",
                    str(Path(preflight_path).resolve()),
                    "--layer",
                    str(layer),
                    "--device",
                    "cuda:0",
                ],
                "resumable": True,
                "starts_only_when_operator_executes_argv": True,
            }
        )
    value: dict[str, object] = {
        "schema": "glm52-fresh-sqg-four-gpu-job-plan-v1",
        "complete": True,
        "preflight_id": preflight["preflight_id"],
        "preflight_sha256": sha256_file(preflight_path),
        "jobs": jobs,
        "concurrency": "one_layer_per_visible_gpu",
        "model_workload_launched": False,
        "production_container_touched": False,
        "existing_model_mutated": False,
    }
    value["job_plan_id"] = canonical_sha256(value)
    atomic_json(paths.output_root / "job_plan.json", value)
    return value
