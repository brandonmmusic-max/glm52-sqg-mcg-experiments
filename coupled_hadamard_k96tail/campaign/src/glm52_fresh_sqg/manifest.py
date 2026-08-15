"""Machine-checkable zero-MCG lineage for fresh GLM SQG candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

import torch

from .reference import payload_sha256, tensor_sha256


RUN_SCHEMA = "glm52_fresh_sqg_run_v1"
TENSOR_SCHEMA = "glm52_fresh_sqg_tensor_v1"
W4A8_DERIVED_SOURCE_KIND = "official_bf16_w4a8_derived_fit_target"
SQG_CODEBOOK = "sqg_xor_cheb_t12"
SQG_MARKER = 0x53514731
TAILBITE_CONTEXT = 128
SQG_LUT_SHA256 = {
    2: "62027916386245a84c86156a0a08b6cf07e41548871af0e56fb40780558f6293",
    3: "afe7b3633e7d243b00b379b18ec4dca573722b3727cafef47fcb6470d7e7e6c9",
    4: "5a9620f0c4d8f0a60d0b6fbea921dcecbd193e6febf31043aaa6403c20389c2f",
    5: "a1caf5572c67d4face423478f423e1ed6149a8c920be5c28d4702946c93cae2f",
    6: "8e0edc92e20eb92f5287dbdadc8e870fc500fc03c5eed996cae0781d197ebcb0",
}
FROZEN_KQUANT_REVISION = "104dd9233f850a3955f4991bea68b07dd34deeb8"
APPROVED_KQUANT_BACKEND_SHA256 = (
    "fb63082016d2adc331be58f538622e1382c86390b0d1715c9a755433fb14c624"
)
APPROVED_KQUANT_TRACKED_DIFF_SHA256 = (
    "82c994a6fa1e1c996f18c85f723c562fbe08bb8a7edf705ef9ece1ae1606958f"
)
APPROVED_KQUANT_STATUS_PORCELAIN = (
    " M kquant/candidate_hessian.py\n"
    " M kquant/csrc/qsrt_quantize_tiles_kernel.cuh\n"
    " M kquant/csrc/sqg_quantize.cu\n"
    " M kquant/exl3_encoder_backend.py\n"
    " M kquant/exl3_reference.py\n"
    " M kquant/qsrt.py\n"
    " M kquant/sqg_e4m3.py\n"
    " M kquant/sqg_quantizer.py\n"
    " M tests/test_qsrt.py\n"
    " M tests/test_sqg_e4m3.py\n"
    " M tests/test_sqg_quantizer.py\n"
)
APPROVED_KQUANT_STATUS_SHA256 = (
    "8ad2cf65ff6656fd4cbf01df725a8d6055229f90142997ae90176cf3b37909ed"
)

FORBIDDEN_MCG_READS = (
    "mcg.suh",
    "mcg.svh",
    "mcg.signs",
    "mcg.scales",
    "mcg.transform_seed",
    "mcg.permutation",
    "mcg.decoded_weight",
    "mcg.trellis",
    "mcg.packed_bytes",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GLM_EXPERT_TENSOR = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_GLM_SHARED_DOWN_TENSOR = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.shared_experts\.down_proj\.weight$"
)


def _require_sha256(value: str, name: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _canonical_json_object(
    value: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    """Return an isolated, JSON-closed copy of a provenance object."""

    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty JSON object")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{name} keys must be nonempty strings")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be canonical-JSON serializable") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - guarded above
        raise ValueError(f"{name} must be a JSON object")
    return decoded


def _canonical_sha256(value: Mapping[str, object], name: str) -> str:
    canonical = _canonical_json_object(value, name)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BF16TensorBinding:
    """Immutable binding from one official BF16 tensor to its source shard.

    ``tensor_sha256`` is the raw safetensors payload digest, matching the
    streaming source provider's hash before any dtype conversion.
    """

    repository_id: str
    revision: str
    shard_name: str
    shard_sha256: str
    tensor_name: str
    tensor_sha256: str

    def __post_init__(self) -> None:
        for name in ("repository_id", "revision", "shard_name", "tensor_name"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("official BF16 revision must be an immutable 40-hex commit")
        if not self.shard_name.endswith(".safetensors"):
            raise ValueError("official BF16 shard must be a safetensors file")
        _require_sha256(self.shard_sha256, "shard_sha256")
        _require_sha256(self.tensor_sha256, "tensor_sha256")

    def validate_tensor(self, tensor: torch.Tensor) -> None:
        if tensor.dtype != torch.bfloat16:
            raise TypeError("production SQG source tensors must be official BF16")
        if payload_sha256(tensor) != self.tensor_sha256:
            raise ValueError("BF16 source tensor hash does not match its immutable binding")

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "official_bf16",
            "repository_id": self.repository_id,
            "revision": self.revision,
            "shard_name": self.shard_name,
            "shard_sha256": self.shard_sha256,
            "tensor_name": self.tensor_name,
            "tensor_payload_sha256": self.tensor_sha256,
            "dtype": "bfloat16",
            "mcg_source": False,
        }


@dataclass(frozen=True)
class W4A8DerivedTensorBinding:
    """Production binding for a fit-only ``(H, B)``-derived down target.

    The encoded tensor is not itself an official checkpoint tensor: it is a
    float32 EXL-oriented target solved from the exact W4A8 execution path.
    Production lineage therefore seals both origins independently: the
    official BF16 down tensor/shard used as the shrinkage parent, and the
    fit-only sufficient-statistic evidence used to derive the replacement.

    ``execution_contract`` is retained in full and checked against the
    caller-supplied digest on every use.  Selection and holdout evidence are
    not constructor options and are unconditionally forbidden in the emitted
    manifest.
    """

    official_bf16_parent: BF16TensorBinding
    execution_contract: Mapping[str, object]
    execution_contract_sha256: str
    fit_hb_evidence_id: str
    fit_hb_evidence_sha256: str
    fit_hessian_sha256: str | None
    fit_cross_term_sha256: str | None
    beta: float
    derived_tensor_sha256: str | None
    candidate_hashes_deferred: bool = False

    def __post_init__(self) -> None:
        parent = self.official_bf16_parent
        if not isinstance(parent, BF16TensorBinding):
            raise TypeError("W4A8-derived target requires an official BF16 parent")
        match = _GLM_EXPERT_TENSOR.fullmatch(parent.tensor_name)
        routed_down = match is not None and match.group("projection") == "down_proj"
        shared_down = _GLM_SHARED_DOWN_TENSOR.fullmatch(parent.tensor_name) is not None
        if not routed_down and not shared_down:
            raise ValueError(
                "W4A8-derived target parent must be a routed or shared GLM down projection"
            )
        contract = _canonical_json_object(
            self.execution_contract,
            "W4A8 execution contract",
        )
        object.__setattr__(self, "execution_contract", contract)
        _require_sha256(
            self.execution_contract_sha256,
            "execution_contract_sha256",
        )
        if (
            _canonical_sha256(contract, "W4A8 execution contract")
            != self.execution_contract_sha256
        ):
            raise ValueError("W4A8 execution contract hash differs")
        for name in (
            "fit_hb_evidence_id",
            "fit_hb_evidence_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.candidate_hashes_deferred, bool):
            raise TypeError("candidate_hashes_deferred must be boolean")
        for name in (
            "fit_hessian_sha256",
            "fit_cross_term_sha256",
            "derived_tensor_sha256",
        ):
            value = getattr(self, name)
            if self.candidate_hashes_deferred:
                if value is not None:
                    raise ValueError(f"deferred candidate {name} must be null")
            else:
                _require_sha256(value, name)
        if (
            isinstance(self.beta, bool)
            or not isinstance(self.beta, (int, float))
            or not math.isfinite(float(self.beta))
            or not 0.0 <= float(self.beta) <= 1.0
        ):
            raise ValueError("W4A8-derived target beta must be finite and lie in [0,1]")

    @property
    def tensor_name(self) -> str:
        """The logical tensor name remains the official parent projection."""

        return self.official_bf16_parent.tensor_name

    def _validated_execution_contract(self) -> dict[str, object]:
        contract = _canonical_json_object(
            self.execution_contract,
            "W4A8 execution contract",
        )
        if (
            _canonical_sha256(contract, "W4A8 execution contract")
            != self.execution_contract_sha256
        ):
            raise ValueError("W4A8 execution contract hash differs")
        return contract

    def validate_tensor(self, tensor: torch.Tensor) -> None:
        self._validated_execution_contract()
        if tensor.dtype != torch.float32 or tensor.ndim != 2:
            raise TypeError(
                "W4A8-derived production targets must be rank-two float32 EXL tensors"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("W4A8-derived production target must be finite")
        if not self.candidate_hashes_deferred and (
            tensor_sha256(tensor) != self.derived_tensor_sha256
        ):
            raise ValueError("W4A8-derived target hash does not match its binding")

    def manifest(self) -> dict[str, object]:
        contract = self._validated_execution_contract()
        parent = self.official_bf16_parent.manifest()
        return {
            "kind": W4A8_DERIVED_SOURCE_KIND,
            "tensor_name": self.tensor_name,
            "official_bf16_parent": parent,
            "official_bf16_parent_binding_sha256": _canonical_sha256(
                parent,
                "official BF16 parent binding",
            ),
            "execution_contract": contract,
            "execution_contract_sha256": self.execution_contract_sha256,
            "fit_hb_evidence": {
                "evidence_id": self.fit_hb_evidence_id,
                "evidence_sha256": self.fit_hb_evidence_sha256,
                "hessian_sha256": self.fit_hessian_sha256,
                "cross_term_sha256": self.fit_cross_term_sha256,
                "split_id": "fit",
                "selection_evidence_used": False,
                "holdout_evidence_used": False,
            },
            "candidate_hashes_deferred": self.candidate_hashes_deferred,
            "beta": float(self.beta),
            "derived_target_tensor_sha256": self.derived_tensor_sha256,
            "dtype": "float32",
            "orientation": (
                "physical_permutation_exl_input_output"
                if _GLM_EXPERT_TENSOR.fullmatch(self.tensor_name) is not None
                else "topology_neutral_exl_input_output"
            ),
            "mcg_source": False,
        }


@dataclass(frozen=True)
class SyntheticTensorBinding:
    """Explicitly test-only source binding; rejected when production=True."""

    tensor_name: str
    tensor_sha256: str
    fixture_id: str

    def __post_init__(self) -> None:
        if not self.tensor_name or not self.fixture_id:
            raise ValueError("synthetic tensor name and fixture_id must not be empty")
        _require_sha256(self.tensor_sha256, "tensor_sha256")

    def validate_tensor(self, tensor: torch.Tensor) -> None:
        if tensor_sha256(tensor) != self.tensor_sha256:
            raise ValueError("synthetic source tensor hash does not match its binding")

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "synthetic_test_fixture",
            "fixture_id": self.fixture_id,
            "tensor_name": self.tensor_name,
            "tensor_sha256": self.tensor_sha256,
            "dtype": str(torch.float32).removeprefix("torch."),
            "mcg_source": False,
        }


SourceBinding = BF16TensorBinding | W4A8DerivedTensorBinding | SyntheticTensorBinding


def validate_tensor_manifest(manifest: Mapping[str, object]) -> None:
    """Reject an incomplete treatment tensor lineage record."""

    required = {
        "schema",
        "tensor_id",
        "source",
        "bits",
        "codebook",
        "codebook_lut_sha256",
        "tailbite_context",
        "transform",
        "scales",
        "dense_h",
        "packed_trellis_sha256",
        "decoded_closure",
        "marker",
        "encoder",
        "forbidden_input_reads",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"tensor manifest is missing required fields: {missing}")
    if manifest["schema"] != TENSOR_SCHEMA:
        raise ValueError("unsupported fresh-SQG tensor manifest schema")
    if manifest["bits"] not in range(2, 7):
        raise ValueError("treatment tensors must retain frozen K2--K6 assignments")
    if manifest["codebook"] != SQG_CODEBOOK:
        raise ValueError("treatment tensor must use the frozen SQG codebook")
    if manifest["codebook_lut_sha256"] != SQG_LUT_SHA256[manifest["bits"]]:
        raise ValueError("treatment tensor SQG LUT does not match its frozen K")
    if manifest["tailbite_context"] != TAILBITE_CONTEXT:
        raise ValueError("treatment tensor must use exact C128 tail-biting")
    encoder_candidate_fast = (
        isinstance(manifest.get("encoder"), Mapping)
        and manifest["encoder"].get("candidate_sweep_fast_requested") is True
    )
    marker = manifest["marker"]
    if marker != {"suffix": ".sqg", "int32": SQG_MARKER, "hex": "0x53514731"}:
        raise ValueError("treatment tensor must have the exclusive SQG1 marker")
    if tuple(manifest["forbidden_input_reads"]) != FORBIDDEN_MCG_READS:
        raise ValueError("tensor forbidden-read declaration is incomplete")
    source = manifest["source"]
    if not isinstance(source, Mapping) or source.get("mcg_source") is not False:
        raise ValueError("treatment tensor source must prove zero-MCG lineage")
    source_kind = source.get("kind")
    production_source = source_kind in (
        "official_bf16",
        W4A8_DERIVED_SOURCE_KIND,
    ) and manifest.get("encoder", {}).get("production", True) is True
    derived_evidence: Mapping[str, object] | None = None
    if source_kind == "official_bf16":
        for name in (
            "repository_id",
            "revision",
            "shard_name",
            "shard_sha256",
            "tensor_name",
            "tensor_payload_sha256",
            "dtype",
        ):
            if name not in source:
                raise ValueError(f"official BF16 binding is missing {name}")
        _require_sha256(str(source["shard_sha256"]), "source shard_sha256")
        _require_sha256(
            str(source["tensor_payload_sha256"]),
            "source tensor_payload_sha256",
        )
        if source["dtype"] != "bfloat16":
            raise ValueError("official source binding dtype must be bfloat16")
    elif source_kind == W4A8_DERIVED_SOURCE_KIND:
        required_derived = {
            "tensor_name",
            "official_bf16_parent",
            "official_bf16_parent_binding_sha256",
            "execution_contract",
            "execution_contract_sha256",
            "fit_hb_evidence",
            "candidate_hashes_deferred",
            "beta",
            "derived_target_tensor_sha256",
            "dtype",
            "orientation",
        }
        missing_derived = sorted(required_derived - set(source))
        if missing_derived:
            raise ValueError(
                f"W4A8-derived source binding is missing {missing_derived}"
            )
        parent = source["official_bf16_parent"]
        if not isinstance(parent, Mapping) or parent.get("kind") != "official_bf16":
            raise ValueError("W4A8-derived source lacks an official BF16 parent")
        if parent.get("mcg_source") is not False:
            raise ValueError("W4A8-derived parent must prove zero-MCG lineage")
        for name in (
            "repository_id",
            "revision",
            "shard_name",
            "shard_sha256",
            "tensor_name",
            "tensor_payload_sha256",
            "dtype",
        ):
            if name not in parent:
                raise ValueError(f"official BF16 parent binding is missing {name}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(parent["revision"])):
            raise ValueError("official BF16 parent revision is not immutable")
        if not str(parent["shard_name"]).endswith(".safetensors"):
            raise ValueError("official BF16 parent shard must be safetensors")
        _require_sha256(str(parent["shard_sha256"]), "parent shard_sha256")
        _require_sha256(
            str(parent["tensor_payload_sha256"]),
            "parent tensor_payload_sha256",
        )
        if parent["dtype"] != "bfloat16":
            raise ValueError("official BF16 parent dtype must be bfloat16")
        parent_name = str(parent["tensor_name"])
        parent_match = _GLM_EXPERT_TENSOR.fullmatch(parent_name)
        routed_down = (
            parent_match is not None
            and parent_match.group("projection") == "down_proj"
        )
        shared_down = _GLM_SHARED_DOWN_TENSOR.fullmatch(parent_name) is not None
        if not routed_down and not shared_down:
            raise ValueError("W4A8-derived parent must be a GLM down projection")
        if source["tensor_name"] != parent["tensor_name"]:
            raise ValueError("W4A8-derived tensor name differs from its BF16 parent")
        _require_sha256(
            str(source["official_bf16_parent_binding_sha256"]),
            "official_bf16_parent_binding_sha256",
        )
        if source["official_bf16_parent_binding_sha256"] != _canonical_sha256(
            parent,
            "official BF16 parent binding",
        ):
            raise ValueError("official BF16 parent binding hash differs")
        contract = source["execution_contract"]
        if not isinstance(contract, Mapping):
            raise ValueError("W4A8 execution contract must be an object")
        _require_sha256(
            str(source["execution_contract_sha256"]),
            "execution_contract_sha256",
        )
        if source["execution_contract_sha256"] != _canonical_sha256(
            contract,
            "W4A8 execution contract",
        ):
            raise ValueError("W4A8 execution contract hash differs")
        evidence = source["fit_hb_evidence"]
        evidence_fields = {
            "evidence_id",
            "evidence_sha256",
            "hessian_sha256",
            "cross_term_sha256",
            "split_id",
            "selection_evidence_used",
            "holdout_evidence_used",
        }
        if not isinstance(evidence, Mapping) or set(evidence) != evidence_fields:
            raise ValueError("W4A8-derived fit H/B evidence field set differs")
        source_hashes_deferred = source.get("candidate_hashes_deferred") is True
        if source_hashes_deferred is not encoder_candidate_fast:
            raise ValueError("W4A8 candidate hash-deferral contract differs")
        for name in ("evidence_id", "evidence_sha256"):
            _require_sha256(str(evidence[name]), f"fit H/B {name}")
        if source_hashes_deferred:
            if (
                evidence["hessian_sha256"] is not None
                or evidence["cross_term_sha256"] is not None
                or source["derived_target_tensor_sha256"] is not None
            ):
                raise ValueError("deferred W4A8 candidate hashes must be null")
        else:
            for name in ("hessian_sha256", "cross_term_sha256"):
                _require_sha256(str(evidence[name]), f"fit H/B {name}")
            _require_sha256(
                str(source["derived_target_tensor_sha256"]),
                "derived_target_tensor_sha256",
            )
        if (
            evidence["split_id"] != "fit"
            or evidence["selection_evidence_used"] is not False
            or evidence["holdout_evidence_used"] is not False
        ):
            raise ValueError(
                "W4A8-derived targets forbid selection and holdout evidence"
            )
        beta = source["beta"]
        if (
            isinstance(beta, bool)
            or not isinstance(beta, (int, float))
            or not math.isfinite(float(beta))
            or not 0.0 <= float(beta) <= 1.0
        ):
            raise ValueError("W4A8-derived beta must be finite and lie in [0,1]")
        if source["dtype"] != "float32":
            raise ValueError("W4A8-derived target dtype must be float32")
        matrix_role = manifest.get("matrix_role")
        expected_orientation = (
            "physical_permutation_exl_input_output"
            if matrix_role == "down"
            else "topology_neutral_exl_input_output"
        )
        if source["orientation"] != expected_orientation:
            raise ValueError("W4A8-derived target orientation differs")
        if matrix_role not in ("down", "shared_down"):
            raise ValueError(
                "W4A8-derived source is valid only for routed/shared down tensors"
            )
        derived_evidence = evidence
    elif source_kind == "synthetic_test_fixture":
        for name in ("fixture_id", "tensor_name", "tensor_sha256", "dtype"):
            if name not in source:
                raise ValueError(f"synthetic source binding is missing {name}")
        _require_sha256(str(source["tensor_sha256"]), "synthetic tensor_sha256")
    else:
        raise ValueError("unsupported treatment tensor source binding kind")

    if production_source:
        encoder = manifest["encoder"]
        approved_encoder = {
            "kquant_revision": FROZEN_KQUANT_REVISION,
            "backend_sha256": APPROVED_KQUANT_BACKEND_SHA256,
            "working_tree_dirty": True,
            "tracked_diff_sha256": APPROVED_KQUANT_TRACKED_DIFF_SHA256,
            "status_sha256": APPROVED_KQUANT_STATUS_SHA256,
            "status_porcelain": APPROVED_KQUANT_STATUS_PORCELAIN,
        }
        if not isinstance(encoder, Mapping) or {
            key: encoder.get(key) for key in approved_encoder
        } != approved_encoder:
            raise ValueError("official BF16 treatment encoder provenance differs")
        gpu_numerical_contract = {
            "numerical_device": "cuda:0",
            "cpu_numerical_work": False,
            "bf16_source_on_cuda": True,
            "hessian_on_cuda": True,
            "closure_on_cuda": True,
            "fp32_accumulate": True,
            "cpu_roles_after_numerical_closure": ["hashing", "serialization"],
        }
        if {
            key: encoder.get(key) for key in gpu_numerical_contract
        } != gpu_numerical_contract:
            raise ValueError(
                "official BF16 treatment does not prove the GPU-only numerical contract"
            )
        transform = manifest["transform"]
        permutation = (
            transform.get("physical_permutation")
            if isinstance(transform, Mapping)
            else None
        )
        dense_topology = bool(
            isinstance(transform, Mapping)
            and transform.get("topology_neutral_dense_input") is True
            and transform.get("physical_input_order") == "identity"
        )
        if not dense_topology and (
            not isinstance(permutation, Mapping)
            or permutation.get("production_qualified") is not True
        ):
            raise ValueError(
                "official BF16 treatment requires either a calibration-derived "
                "expert permutation or an explicit topology-neutral dense input"
            )
    transform = manifest["transform"]
    if not isinstance(transform, Mapping):
        raise ValueError("tensor transform lineage must be an object")
    transform_fields = {
        "kquant_input_sign_seed",
        "kquant_output_sign_seed",
        "input_signs_sha256",
        "output_signs_sha256",
        "normalized_hadamard_sha256",
        "global_scale",
        "transform_sha256",
        "legacy_transform_input",
    }
    if transform_fields - set(transform):
        raise ValueError("tensor transform lineage is incomplete")
    if transform["legacy_transform_input"] is not False:
        raise ValueError("tensor transform lineage contains a legacy transform")
    for name in (
        "input_signs_sha256",
        "output_signs_sha256",
        "normalized_hadamard_sha256",
        "transform_sha256",
    ):
        _require_sha256(str(transform[name]), name)
    if source_kind == W4A8_DERIVED_SOURCE_KIND:
        if (
            transform.get("prepared_exl_sha256")
            != source["derived_target_tensor_sha256"]
        ):
            raise ValueError("W4A8-derived prepared target hash differs")
        expected_operation = (
            "w4a8_derived_exl_clone_only"
            if manifest.get("matrix_role") == "down"
            else "w4a8_derived_dense_exl_clone_only"
        )
        if transform.get("operation") != expected_operation:
            raise ValueError("W4A8-derived target was transformed again")
    scales = manifest["scales"]
    if not isinstance(scales, Mapping):
        raise ValueError("fresh scale lineage must be an object")
    if scales.get("freshly_generated") is not True or scales.get("legacy_scale_input") is not False:
        raise ValueError("scale lineage does not prove fresh zero-MCG generation")
    _require_sha256(str(scales.get("suh_sha256")), "suh_sha256")
    _require_sha256(str(scales.get("svh_sha256")), "svh_sha256")
    closure = manifest["decoded_closure"]
    if not isinstance(closure, Mapping) or closure.get("passed") is not True:
        raise ValueError("stored-FP16 independent decode closure did not pass")
    dense_h = manifest["dense_h"]
    if not isinstance(dense_h, Mapping) or dense_h.get("fallback") is not False:
        raise ValueError("dense-H evidence must prove no fallback")
    if dense_h.get("block_ldlq") is not True:
        raise ValueError("dense-H evidence must prove BlockLDLQ")
    if encoder_candidate_fast:
        if dense_h.get("matrix_sha256") is not None:
            raise ValueError("fast candidate unexpectedly hashed its dense Hessian")
    else:
        _require_sha256(str(dense_h.get("matrix_sha256")), "dense-H matrix_sha256")
    if derived_evidence is not None:
        if dense_h.get("split_id") != "fit":
            raise ValueError("W4A8-derived dense H must use only the fit split")
        if dense_h.get("evidence_id") != derived_evidence["evidence_id"]:
            raise ValueError("W4A8-derived dense-H evidence identity differs")
        if (
            not encoder_candidate_fast
            and dense_h.get("matrix_sha256") != derived_evidence["hessian_sha256"]
        ):
            raise ValueError("W4A8-derived dense-H payload hash differs")
    _require_sha256(
        str(manifest["packed_trellis_sha256"]),
        "packed_trellis_sha256",
    )
    encoder = manifest["encoder"]
    if not isinstance(encoder, Mapping):
        raise ValueError("tensor encoder provenance must be an object")
    candidate_fast = encoder.get("candidate_sweep_fast_requested") is True
    if candidate_fast:
        if (
            transform.get("candidate_prepared_hash_deferred")
            is not bool(encoder.get("candidate_source_hash_deferred"))
            or
            closure.get("mode") != "candidate_sweep_fast"
            or closure.get("implementation")
            != "candidate_sweep_structural_cuda_v1"
            or closure.get("full_decode_deferred") is not True
            or closure.get("selected_model_eligible") is not False
            or closure.get("decoded_exl_sha256") is not None
            or closure.get("source_relative_rmse") is not None
            or closure.get("encoder_relative_rmse") is not None
            or encoder.get("full_decode_closure_performed") is not False
        ):
            raise ValueError("fast candidate closure contract differs")
    elif "candidate_sweep_fast_requested" in encoder:
        if (
            transform.get("candidate_prepared_hash_deferred") is not False
            or
            closure.get("mode") != "selected_full"
            or closure.get("implementation")
            != "independent_pytorch_packed_fp16_v1"
            or closure.get("full_decode_deferred") is not False
            or closure.get("selected_model_eligible") is not True
            or not isinstance(closure.get("decoded_exl_sha256"), str)
            or not isinstance(closure.get("source_relative_rmse"), (int, float))
            or not isinstance(closure.get("encoder_relative_rmse"), (int, float))
            or encoder.get("full_decode_closure_performed") is not True
        ):
            raise ValueError("selected full closure contract differs")
    for name in (
        "backend_sha256",
        "tracked_diff_sha256",
        "status_sha256",
    ):
        _require_sha256(str(encoder.get(name)), f"encoder {name}")
    if not isinstance(encoder.get("kquant_revision"), str) or not encoder[
        "kquant_revision"
    ]:
        raise ValueError("encoder KQuant revision is absent")
    if not isinstance(encoder.get("working_tree_dirty"), bool):
        raise ValueError("encoder dirty-state evidence is not boolean")
    if not isinstance(encoder.get("status_porcelain"), str):
        raise ValueError("encoder porcelain status is absent")


@dataclass(frozen=True)
class FrozenBitBudget:
    """The only controlled input retained from the previous GLM allocation."""

    bit_map: Mapping[str, int]
    expected_k3: int
    expected_k4: int

    def __post_init__(self) -> None:
        if not self.bit_map:
            raise ValueError("frozen bit_map must not be empty")
        if any(value not in (3, 4) for value in self.bit_map.values()):
            raise ValueError("frozen bit_map may contain only K3 and K4")
        actual_k3 = sum(value == 3 for value in self.bit_map.values())
        actual_k4 = sum(value == 4 for value in self.bit_map.values())
        if (actual_k3, actual_k4) != (self.expected_k3, self.expected_k4):
            raise ValueError(
                "frozen K3/K4 budget does not match the declared exact census"
            )

    def manifest(self) -> dict[str, object]:
        return {
            "controlled_input": "bit_map_only",
            "assignments": dict(sorted(self.bit_map.items())),
            "k3_count": self.expected_k3,
            "k4_count": self.expected_k4,
            "other_rate_count": 0,
            "topology_neutral_per_tensor": True,
        }


def build_run_manifest(
    *,
    run_id: str,
    bit_budget: FrozenBitBudget,
    tensor_manifests: Sequence[Mapping[str, object]],
    fresh_permutations: Sequence[Mapping[str, object]],
    production: bool = True,
) -> dict[str, object]:
    """Build and validate the complete zero-MCG treatment manifest."""

    if not run_id:
        raise ValueError("run_id must not be empty")
    manifests = [dict(item) for item in tensor_manifests]
    for item in manifests:
        validate_tensor_manifest(item)
        if production and item["source"].get("kind") not in (
            "official_bf16",
            W4A8_DERIVED_SOURCE_KIND,
        ):
            raise ValueError("production run contains a non-production test source")
    by_id = {str(item["tensor_id"]): item for item in manifests}
    if len(by_id) != len(manifests):
        raise ValueError("treatment tensor IDs must be unique")
    if set(by_id) != set(bit_budget.bit_map):
        raise ValueError("tensor manifests must cover the frozen bit_map exactly")
    for tensor_id, bits in bit_budget.bit_map.items():
        if by_id[tensor_id]["bits"] != bits:
            raise ValueError(f"tensor {tensor_id} changed its frozen bit assignment")
    permutations = [dict(item) for item in fresh_permutations]
    if not permutations:
        raise ValueError("a treatment run must bind fresh physical permutations")
    for item in permutations:
        if item.get("legacy_permutation_input") is not False:
            raise ValueError("run contains a non-fresh physical permutation")
        if production and item.get("production_qualified") is not True:
            raise ValueError(
                "run contains a random/test permutation instead of "
                "calibration-derived h2_reverse"
            )
    if production:
        declared_permutations = {
            str(item.get("new_to_old_sha256")) for item in permutations
        }
        expert_groups: dict[tuple[int, int], dict[str, Mapping[str, object]]] = {}
        role_by_projection = {
            "gate_proj": "gate",
            "up_proj": "up",
            "down_proj": "down",
        }
        for item in manifests:
            source = item["source"]
            official_source = (
                source.get("official_bf16_parent")
                if source.get("kind") == W4A8_DERIVED_SOURCE_KIND
                else source
            )
            if not isinstance(official_source, Mapping):
                raise ValueError("production tensor lacks an official BF16 identity")
            match = _GLM_EXPERT_TENSOR.fullmatch(
                str(official_source.get("tensor_name"))
            )
            if match is None:
                raise ValueError("production tensor has a non-GLM expert source binding")
            projection = match.group("projection")
            if item.get("matrix_role") != role_by_projection[projection]:
                raise ValueError("tensor role disagrees with its official BF16 binding")
            key = (int(match.group("layer")), int(match.group("expert")))
            group = expert_groups.setdefault(key, {})
            if projection in group:
                raise ValueError("production run repeats an expert projection")
            group[projection] = item
        used_permutations: set[str] = set()
        for key, group in expert_groups.items():
            if set(group) != {"gate_proj", "up_proj", "down_proj"}:
                raise ValueError(f"expert {key} does not contain all three projections")
            hashes = {
                str(item["transform"]["physical_permutation"]["new_to_old_sha256"])
                for item in group.values()
            }
            if len(hashes) != 1:
                raise ValueError(
                    f"expert {key} does not share one gate/up/down physical permutation"
                )
            used_permutations.update(hashes)
        if used_permutations != declared_permutations:
            raise ValueError(
                "run-level fresh permutation declarations do not match tensor usage"
            )
    result = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "production": production,
        "codec": {
            "name": "glm52_fresh_uniform_sqg",
            "codebook": SQG_CODEBOOK,
            "supported_bits": [3, 4],
            "tailbite_context": TAILBITE_CONTEXT,
            "dense_h_required": True,
            "fallback_allowed": False,
            "marker": {"suffix": ".sqg", "int32": SQG_MARKER, "hex": "0x53514731"},
        },
        "forbidden_reads": {
            "policy": "zero_mcg_lineage_v1",
            "items": list(FORBIDDEN_MCG_READS),
            "observed": [],
            "enforced": True,
        },
        "frozen_bit_budget": bit_budget.manifest(),
        "fresh_physical_permutations": permutations,
        "tensor_manifests": manifests,
    }
    # JSON closure catches tensors or other accidental non-serializable state.
    json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return result
