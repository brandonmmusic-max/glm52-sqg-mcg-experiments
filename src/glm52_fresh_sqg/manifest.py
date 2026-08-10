"""Machine-checkable zero-MCG lineage for fresh GLM SQG candidates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Mapping, Sequence

import torch

from .reference import payload_sha256, tensor_sha256


RUN_SCHEMA = "glm52_fresh_sqg_run_v1"
TENSOR_SCHEMA = "glm52_fresh_sqg_tensor_v1"
SQG_CODEBOOK = "sqg_xor_cheb_t12"
SQG_MARKER = 0x53514731
TAILBITE_CONTEXT = 128
SQG_LUT_SHA256 = {
    3: "afe7b3633e7d243b00b379b18ec4dca573722b3727cafef47fcb6470d7e7e6c9",
    4: "5a9620f0c4d8f0a60d0b6fbea921dcecbd193e6febf31043aaa6403c20389c2f",
}
FROZEN_KQUANT_REVISION = "104dd9233f850a3955f4991bea68b07dd34deeb8"
APPROVED_KQUANT_BACKEND_SHA256 = (
    "2352f06ec515c886ed896203df4be2374aa265a7c910fb06727095ba443f6254"
)
APPROVED_KQUANT_TRACKED_DIFF_SHA256 = (
    "8ba6746b722342a11f2944e7421258a3d0238b72dd785d49b7b49483476b4c2e"
)
APPROVED_KQUANT_STATUS_PORCELAIN = (
    " M kquant/exl3_encoder_backend.py\n"
    " M kquant/sqg_quantizer.py\n"
    " M tests/test_sqg_quantizer.py\n"
)
APPROVED_KQUANT_STATUS_SHA256 = (
    "6204eaab3737d5858492d3b5d7ee8d96d57b41cb361720122b8ff639ac94f92d"
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


def _require_sha256(value: str, name: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


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


SourceBinding = BF16TensorBinding | SyntheticTensorBinding


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
    if manifest["bits"] not in (3, 4):
        raise ValueError("treatment tensors must retain frozen K3/K4 assignments")
    if manifest["codebook"] != SQG_CODEBOOK:
        raise ValueError("treatment tensor must use the frozen SQG codebook")
    if manifest["codebook_lut_sha256"] != SQG_LUT_SHA256[manifest["bits"]]:
        raise ValueError("treatment tensor SQG LUT does not match its frozen K")
    if manifest["tailbite_context"] != TAILBITE_CONTEXT:
        raise ValueError("treatment tensor must use exact C128 tail-biting")
    marker = manifest["marker"]
    if marker != {"suffix": ".sqg", "int32": SQG_MARKER, "hex": "0x53514731"}:
        raise ValueError("treatment tensor must have the exclusive SQG1 marker")
    if tuple(manifest["forbidden_input_reads"]) != FORBIDDEN_MCG_READS:
        raise ValueError("tensor forbidden-read declaration is incomplete")
    source = manifest["source"]
    if not isinstance(source, Mapping) or source.get("mcg_source") is not False:
        raise ValueError("treatment tensor source must prove zero-MCG lineage")
    if source.get("kind") == "official_bf16":
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
        transform = manifest["transform"]
        permutation = (
            transform.get("physical_permutation")
            if isinstance(transform, Mapping)
            else None
        )
        if (
            not isinstance(permutation, Mapping)
            or permutation.get("production_qualified") is not True
        ):
            raise ValueError(
                "official BF16 treatment requires a calibration-derived "
                "physical permutation"
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
    _require_sha256(str(dense_h.get("matrix_sha256")), "dense-H matrix_sha256")
    _require_sha256(
        str(manifest["packed_trellis_sha256"]),
        "packed_trellis_sha256",
    )
    encoder = manifest["encoder"]
    if not isinstance(encoder, Mapping):
        raise ValueError("tensor encoder provenance must be an object")
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
        if production and item["source"].get("kind") != "official_bf16":
            raise ValueError("production run contains a non-BF16 test source")
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
            match = _GLM_EXPERT_TENSOR.fullmatch(str(source.get("tensor_name")))
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
