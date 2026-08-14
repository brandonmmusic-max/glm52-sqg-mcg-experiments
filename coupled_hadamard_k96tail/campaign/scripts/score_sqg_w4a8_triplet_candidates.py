#!/usr/bin/env python3
"""Build fit-only realized full-W4A8 K3/K4 triplet scores for one GLM layer.

This is the candidate producer for the v3 exact-budget allocator.  It does not
inherit an MCG rate map and it never reads selection or holdout rows.  The fit
documents are deterministically divided a second time:

* ``fit/calibration`` constructs the profile-specific W4A8 H13, encodes the
  gate/up candidates, fits candidate-specific ``(H,B)`` down targets, and
  encodes the anchored down candidates;
* ``fit/allocation`` executes every complete gate/up/down K3/K4 triplet through
  h-A8, native-E4M3 weights, exact GLM ``SiLU(gate) * up``, act-A8, and the
  output transform, then measures raw gate-square-weighted expert-output SSE.

The final manifest contains exactly eight unary candidates for every expert
and conforms to ``TRIPLET_SCORE_SCHEMA``.  Signed top-8 cross-expert terms are
deliberately deferred until after the exact additive DP allocation.

The script has three fail-closed modes.  ``--prepare-h13`` writes the single
layer-global caller-coordinate W4A8 H13.  Worker mode (``--start/--end``)
produces resumable expert JSON records while retaining only payload hashes.
``--finalize`` validates all 256 expert records and seals the layer manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "kquant"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts.score_glm52_w4a8_activation_quality import (  # noqa: E402
    NativeProjection,
    apply_gate_up_output_transform_silu,
    apply_output_transform,
    native_label_gemm,
    prepare_down_operand,
    prepare_gate_up_operand,
)
from scripts.coupled_gate_up_scale_selector import (  # noqa: E402
    CoupledScaleContext,
    CEILING_TOLERANCE,
    ExactEvaluationBatch,
    ExactGLMW4A8UpstreamEvaluator,
    FitCalibrationPartition,
    HARD_CEILING,
    ProjectionCandidate,
    select_coupled_gate_up_scales,
    with_activation_parametric_scale_override,
)
from scripts.w4a8_cross_term import (  # noqa: E402
    CrossTermStatistics,
    effective_canonical_operand_from_quantized_transform,
    kquant_prefinalize_operand_from_label_operand,
    shrink_cross_term_objective,
)
from scripts.w4a8_stable_solve import solve_with_minimal_official_prior  # noqa: E402
from src.sqg_k34_allocation import (  # noqa: E402
    FULL_W4A8_ENDPOINT,
    TRIPLET_LOSS_DEFINITION,
    TRIPLET_PROJECTIONS,
    TRIPLET_SCORE_SCHEMA,
    _validate_triplet_scores,
    canonical_json_bytes,
    enumerate_rate_triplets,
    sha256_file,
)


NUM_EXPERTS = 256
HIDDEN = 6144
INTERMEDIATE = 2048
HADAMARD_BLOCK = 128
CALIBRATION_BUCKETS = 5
ALLOCATION_BUCKET = 0
LOCAL_H13_ALPHA = 0.25
H13_CONSTRUCTION = (
    "fit_calibration_profile_specific_full_w4a8_qpre_gate_square_"
    "fixed_alpha_0p25_expert_local_layer_global_prior_v1"
)
GLOBAL_H13_CONSTRUCTION = (
    "fit_calibration_profile_specific_full_w4a8_qpre_gate_square_layer_global_v1"
)
PRELIMINARY_H2_CONSTRUCTION = (
    "fit_calibration_official_bf16_swiglu_permuted_weighted_oas_"
    "scaled_identity_cap_0p75_for_down_anchor_v1"
)
DERIVED_H2_CONSTRUCTION = (
    "fit_calibration_candidate_specific_full_w4a8_cross_term_"
    "qpre_encoder_h_anchored_suh_v1"
)
EXPERT_SCHEMA = "glm52-sqg-w4a8-expert-triplet-scores-v4"
GLOBAL_H13_SCHEMA = "glm52-sqg-w4a8-global-qpre-h13-v1"
SUBFOLD_SCHEMA = "glm52-fit-document-secondary-split-v1"


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def fit_document_bucket(layer: int, document_epoch: int) -> int:
    """Return a stable document-level secondary-fit bucket in ``[0,4]``."""

    if layer < 0 or document_epoch < 0:
        raise ValueError("layer and document epoch must be nonnegative")
    material = (
        f"glm52-sqg-w4a8-triplet-fit-subfold-v1/{layer}/{document_epoch}"
    ).encode("ascii")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big") % CALIBRATION_BUCKETS


def fit_subfold_mask(
    document_epochs: Iterable[int] | np.ndarray | torch.Tensor,
    *,
    layer: int,
    subfold: str,
) -> np.ndarray:
    """Select whole documents for calibration or allocation within fit."""

    if subfold not in ("calibration", "allocation"):
        raise ValueError("fit subfold must be calibration or allocation")
    if isinstance(document_epochs, torch.Tensor):
        epochs = document_epochs.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        epochs = np.asarray(document_epochs, dtype=np.int64)
    if epochs.ndim != 1 or (epochs.size and int(epochs.min()) < 0):
        raise ValueError("document epochs must be a nonnegative vector")
    unique, inverse = np.unique(epochs, return_inverse=True)
    allocation_by_document = np.array(
        [
            fit_document_bucket(layer, int(epoch)) == ALLOCATION_BUCKET
            for epoch in unique
        ],
        dtype=np.bool_,
    )
    allocation = allocation_by_document[inverse]
    return allocation if subfold == "allocation" else ~allocation


def subfold_contract(layer: int) -> dict[str, Any]:
    return {
        "schema": SUBFOLD_SCHEMA,
        "layer": int(layer),
        "parent_role": "fit",
        "unit": "document_epoch",
        "hash": "sha256",
        "namespace": "glm52-sqg-w4a8-triplet-fit-subfold-v1",
        "buckets": CALIBRATION_BUCKETS,
        "allocation_bucket": ALLOCATION_BUCKET,
        "calibration_buckets": [1, 2, 3, 4],
        "document_disjoint": True,
        "selection_used": False,
        "holdout_used": False,
    }


def gate_square_weighted_complete_expert_sse(
    candidate_output: torch.Tensor,
    teacher_output: torch.Tensor,
    route_gates: torch.Tensor,
) -> torch.Tensor:
    """Return the raw additive unary objective consumed by the v3 DP.

    This is an inner-loop CUDA reduction.  Deliberately do not materialize
    data-dependent Python booleans here: doing so synchronizes the device once
    (or several times) for every candidate.  Callers validate the route-gate
    vector once before entering their chunk loop and validate the single
    frozen score vector after all candidates have been accumulated.
    """

    if candidate_output.ndim != 2 or teacher_output.shape != candidate_output.shape:
        raise ValueError(
            "candidate and teacher outputs must be aligned rank-two tensors"
        )
    if route_gates.ndim != 1 or route_gates.shape[0] != candidate_output.shape[0]:
        raise ValueError("route gates must supply one value per output row")
    candidate = candidate_output.float()
    teacher = teacher_output.float()
    gates = route_gates.float()
    return torch.sum(
        (candidate - teacher).square().sum(dim=1) * gates.square(),
        dtype=torch.float64,
    )


def validate_route_gates_once(route_gates: torch.Tensor, *, rows: int) -> None:
    """Validate a complete expert/role gate vector outside candidate loops."""

    if route_gates.ndim != 1 or route_gates.shape[0] != rows:
        raise ValueError("route gates must supply one value per output row")
    # Captured routing metadata is host-resident.  Preserve that contract so
    # this check cannot introduce a device synchronization on the paid path.
    if route_gates.is_cuda:
        raise ValueError("route gates must be validated while host-resident")
    gates = route_gates.float()
    if not bool(torch.isfinite(gates).all()) or bool((gates < 0).any()):
        raise ValueError("route gates must be finite and nonnegative")


def validate_frozen_sse_vector(values: Sequence[float], *, expected: int) -> None:
    """Fail closed once, after the terminal score-vector device transfer."""

    if len(values) != expected or any(
        not math.isfinite(float(value)) or float(value) < 0 for value in values
    ):
        raise ValueError("complete-expert weighted SSE vector is invalid")


def deferred_down_objective_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonical losing-candidate receipt without full CUDA tensor hashes."""

    body = dict(evidence)
    body.pop("evidence_id", None)
    body.pop("candidate_evidence_id", None)
    for field in (
        "canonical_h_sha256",
        "cross_term_sha256",
        "encoder_h_sha256",
        "derived_target_sha256",
    ):
        body[field] = None
    body["candidate_hashes_deferred"] = True
    return body


def deferred_h13_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical H13 construction receipt without hashing the 6144^2 matrix."""

    body = dict(evidence)
    body.pop("evidence_id", None)
    body.pop("candidate_evidence_id", None)
    body["matrix_sha256"] = None
    body["candidate_hashes_deferred"] = True
    return body


def execute_full_w4a8_expert(
    hidden: torch.Tensor,
    gate: NativeProjection,
    up: NativeProjection,
    down: NativeProjection,
    hadamard: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the exact numeric full-W4A8 candidate path and return output/act."""

    activations = execute_full_w4a8_upstream_grid(
        hidden,
        {gate.bits: gate},
        {up.bits: up},
        hadamard,
    )
    activation = activations[(gate.bits, up.bits)]
    output = execute_full_w4a8_down(activation, down, hadamard)
    return output, activation


def execute_full_w4a8_upstream_grid(
    hidden: torch.Tensor,
    gates: Mapping[int, NativeProjection],
    ups: Mapping[int, NativeProjection],
    hadamard: torch.Tensor,
) -> dict[tuple[int, int], torch.Tensor]:
    """Execute the shared h-A8 operand once for a gate/up candidate grid.

    Every K3/K4 upstream candidate is encoded under the same topology-shared
    input profile.  Re-quantizing that identical operand once per rate pair is
    byte-neutral work and also repeats the two projection GEMMs.  This helper
    preserves each candidate's exact native labels and output transform while
    sharing only the mathematically identical input-side operation.
    """

    if not gates or not ups:
        raise ValueError("gate/up candidate grids must be nonempty")
    reference_suh = next(iter(gates.values())).suh
    for projection in (*gates.values(), *ups.values()):
        if not torch.equal(projection.suh, reference_suh):
            raise ValueError("gate/up candidate input scales must be topology-shared")
    h_a8, h_observation, _ = prepare_gate_up_operand(
        hidden,
        reference_suh,
        hadamard,
        quantize_a8=True,
    )
    if h_observation is None or bool(h_observation.preclamp_overflow.any()):
        raise RuntimeError("h-A8 candidate overflowed")
    gate_products = {
        bits: native_label_gemm(h_a8, projection.weight)
        for bits, projection in gates.items()
    }
    up_products = {
        bits: native_label_gemm(h_a8, projection.weight)
        for bits, projection in ups.items()
    }
    activations: dict[tuple[int, int], torch.Tensor] = {}
    for gate_bits, gate in gates.items():
        for up_bits, up in ups.items():
            _, _, activation = apply_gate_up_output_transform_silu(
                gate_products[gate_bits],
                up_products[up_bits],
                gate.svh,
                up.svh,
                hadamard,
            )
            activations[(gate_bits, up_bits)] = activation
    return activations


def execute_full_w4a8_upstream_pairs(
    hidden: torch.Tensor,
    candidates: Mapping[
        tuple[int, int], Mapping[str, NativeProjection]
    ],
    hadamard: torch.Tensor,
) -> dict[tuple[int, int], torch.Tensor]:
    """Execute partner-conditional upstream bytes for explicit rate pairs."""

    expected = {(3, 3), (3, 4), (4, 3), (4, 4)}
    if not candidates or not set(candidates).issubset(expected):
        raise ValueError("coupled upstream candidate pair domain differs")
    first = next(iter(candidates.values()))
    reference_suh = first["gate_proj"].suh
    for pair, projections in candidates.items():
        if set(projections) != {"gate_proj", "up_proj"}:
            raise ValueError(f"coupled upstream projection census differs: {pair}")
        gate = projections["gate_proj"]
        up = projections["up_proj"]
        if (gate.bits, up.bits) != pair:
            raise ValueError("coupled upstream rate/payload identity differs")
        if not torch.equal(gate.suh, reference_suh) or not torch.equal(
            up.suh, reference_suh
        ):
            raise ValueError("coupled upstream candidates changed shared h profile")
    h_a8, observation, _ = prepare_gate_up_operand(
        hidden, reference_suh, hadamard, quantize_a8=True
    )
    if observation is None or bool(observation.preclamp_overflow.any()):
        raise RuntimeError("h-A8 coupled upstream candidate overflowed")
    activations: dict[tuple[int, int], torch.Tensor] = {}
    product_cache: dict[int, torch.Tensor] = {}
    for pair, projections in candidates.items():
        gate = projections["gate_proj"]
        up = projections["up_proj"]
        if id(gate) not in product_cache:
            product_cache[id(gate)] = native_label_gemm(h_a8, gate.weight)
        if id(up) not in product_cache:
            product_cache[id(up)] = native_label_gemm(h_a8, up.weight)
        gate_product = product_cache[id(gate)]
        up_product = product_cache[id(up)]
        _, _, activation = apply_gate_up_output_transform_silu(
            gate_product,
            up_product,
            gate.svh,
            up.svh,
            hadamard,
        )
        activations[pair] = activation
    return activations


def execute_full_w4a8_down(
    activation: torch.Tensor,
    down: NativeProjection,
    hadamard: torch.Tensor,
) -> torch.Tensor:
    """Execute one exact act-A8/native-E4M3 down candidate."""

    act_a8, act_observation, _ = prepare_down_operand(
        activation,
        down.suh,
        hadamard,
        quantize_a8=True,
    )
    if act_observation is None or bool(act_observation.preclamp_overflow.any()):
        raise RuntimeError("act-A8 candidate overflowed")
    output = apply_output_transform(
        native_label_gemm(act_a8, down.weight),
        down.svh,
        hadamard,
    )
    return output


def _projection_payload_sha256(encoded: Any) -> str:
    """Hash only persisted model bytes, independently of audit manifests.

    Candidate sweeps use deferred closure while selected materialization uses
    full closure.  Their manifests must differ; their stored SQG bytes must
    not.  Keep the manifest digest as a separate lineage field wherever it is
    needed and use this digest for deterministic candidate/final equality.
    """

    from src.glm52_fresh_sqg.reference import tensor_sha256

    payload = {
        "trellis": tensor_sha256(encoded.trellis),
        "suh": tensor_sha256(encoded.suh),
        "svh": tensor_sha256(encoded.svh),
        # torch does not permit a dtype-changing view of a 0-D scalar. The
        # marker payload is unchanged; reshape only gives the byte hasher a
        # one-element view over the same dtype and storage bytes.
        "sqg": tensor_sha256(encoded.sqg.reshape(1)),
    }
    return canonical_sha256(payload)


def _config_at_rate(
    runtime: Any,
    weights: Any,
    projection: str,
    permutation: Any,
    gate_profile: Any,
    down_profile: Any,
    *,
    bits: int,
    candidate_sweep: bool = False,
) -> Any:
    """Build a K3/K4 config without inheriting a legacy K5 source-map cell."""

    from src.fresh_pipeline_common import tensor_prefix
    from src.fresh_pipeline_runner import _config

    if bits not in (3, 4):
        raise ValueError("triplet candidates permit only K3 or K4")
    prefix = tensor_prefix(runtime.layer, int(weights.expert), projection)
    scoped_bit_map = dict(runtime.bit_map)
    scoped_bit_map[prefix] = bits
    scoped_runtime = replace(runtime, bit_map=scoped_bit_map)
    config = _config(
        scoped_runtime,
        weights,
        projection,
        permutation,
        gate_profile,
        down_profile,
    )
    if config.bits != bits:
        raise AssertionError("rate-scoped config changed the requested SQG rate")
    return replace(config, candidate_sweep_fast=candidate_sweep)


def _native_projection(
    encoded: Any, *, bits: int, lut: torch.Tensor, device: torch.device
) -> NativeProjection:
    from src.glm52_fresh_sqg.reference import (
        decode_regularized_states,
        unpack_trellis_states,
    )

    states = unpack_trellis_states(encoded.trellis.to(device), bits)
    native = decode_regularized_states(states, lut).contiguous()
    if not torch.equal(native, native.to(torch.float8_e4m3fn).float()):
        raise RuntimeError("SQG candidate labels are not exact finite E4M3")
    return NativeProjection(
        weight=native,
        suh=encoded.suh.to(device),
        svh=encoded.svh.to(device),
        bits=bits,
    )


@dataclass(frozen=True)
class CoupledUpstreamCandidates:
    """Frozen partner-conditional gate/up candidates and their scale receipt."""

    encoded_by_pair: Mapping[tuple[int, int], Mapping[str, Any]]
    native_by_pair: Mapping[
        tuple[int, int], Mapping[str, NativeProjection]
    ]
    scale_evidence: Mapping[str, Any]
    raw_center_receipts: Mapping[str, Mapping[int, Mapping[str, Any]]]

    def choice_receipt(self, pair: tuple[int, int]) -> dict[str, Any]:
        if pair not in self.encoded_by_pair or pair not in self.native_by_pair:
            raise ValueError(f"coupled upstream rate pair is absent: {pair}")
        results = self.scale_evidence.get("rate_pair_results")
        if not isinstance(results, list):
            raise ValueError("coupled scale evidence lacks rate-pair results")
        matches = [
            item
            for item in results
            if (int(item.get("gate_bits", -1)), int(item.get("up_bits", -1)))
            == pair
        ]
        if len(matches) != 1:
            raise ValueError("coupled scale evidence rate-pair choice is not unique")
        selected = matches[0].get("selected")
        candidates = self.scale_evidence.get("projection_candidates")
        if not isinstance(selected, Mapping) or not isinstance(candidates, list):
            raise ValueError("coupled scale evidence candidate receipt is absent")
        by_id = {str(item.get("candidate_id")): item for item in candidates}
        gate = by_id.get(str(selected.get("gate_candidate_id")))
        up = by_id.get(str(selected.get("up_candidate_id")))
        if not isinstance(gate, Mapping) or not isinstance(up, Mapping):
            raise ValueError("coupled selected projection receipt is absent")
        value = {
            "schema": "glm52-full-w4a8-coupled-upstream-choice-v1",
            "rate_pair": {"gate_proj": pair[0], "up_proj": pair[1]},
            "scale_evidence_id": self.scale_evidence["evidence_id"],
            "search_contract_id": self.scale_evidence["search_contract_id"],
            "gate": dict(gate),
            "up": dict(up),
            "fit_calibration_only": True,
            "selection_used": False,
            "holdout_used": False,
            "fit_allocation_used": False,
        }
        value["choice_id"] = canonical_sha256(value)
        return value


def validate_coupled_scale_choice(
    value: Mapping[str, Any], *, gate_bits: int, up_bits: int
) -> dict[str, Any]:
    """Validate one frozen pair-specific scale/payload choice for consumption."""

    body = dict(value)
    choice_id = body.pop("choice_id", None)
    if (
        value.get("schema") != "glm52-full-w4a8-coupled-upstream-choice-v1"
        or value.get("rate_pair")
        != {"gate_proj": gate_bits, "up_proj": up_bits}
        or value.get("fit_calibration_only") is not True
        or value.get("selection_used") is not False
        or value.get("holdout_used") is not False
        or value.get("fit_allocation_used") is not False
        or choice_id != canonical_sha256(body)
    ):
        raise ValueError("frozen coupled scale choice contract differs")
    contract_id = value.get("search_contract_id")
    evidence_id = value.get("scale_evidence_id")
    for label, digest in (
        ("search contract", contract_id),
        ("scale evidence", evidence_id),
        ("choice", choice_id),
    ):
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise ValueError(f"frozen coupled {label} ID differs")
    for projection, bits in (("gate", gate_bits), ("up", up_bits)):
        receipt = value.get(projection)
        expected_projection = f"{projection}_proj"
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("projection") != expected_projection
            or int(receipt.get("bits", -1)) != bits
            or receipt.get("override_policy")
            != "activation_parametric_coupled_v1"
            or receipt.get("override_evidence_id") != contract_id
        ):
            raise ValueError(f"frozen coupled {projection} receipt differs")
        scale = float(receipt.get("scale", math.nan))
        if (
            not math.isfinite(scale)
            or scale <= 0.0
            or scale >= HARD_CEILING - CEILING_TOLERANCE
            or receipt.get("scale_hex") != scale.hex()
        ):
            raise ValueError(f"frozen coupled {projection} scale differs")
        for key in (
            "candidate_id",
            "payload_sha256",
            "trellis_sha256",
            "suh_sha256",
            "svh_sha256",
            "decoded_label_sha256",
        ):
            digest = receipt.get(key)
            if not isinstance(digest, str) or len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError(f"frozen coupled {projection} {key} differs")
    return dict(value)


def _config_scale_binding(config: Any) -> dict[str, Any]:
    """Serialize only immutable, arithmetic-relevant gate/up config fields."""

    permutation = config.physical_permutation
    profile = config.shared_residual_profile
    return {
        "tensor_id": config.tensor_id,
        "bits": int(config.bits),
        "matrix_role": config.matrix_role,
        "transform_seed": int(config.transform_seed),
        "output_sign_seed": int(config.output_sign_seed),
        "sigma_reg": float(config.sigma_reg),
        "apply_out_scales": config.apply_out_scales,
        "global_scale_into": config.global_scale_into,
        "source": config.source_binding.manifest(),
        "physical_permutation_sha256": (
            None if permutation is None else permutation.sha256
        ),
        "shared_residual_profile": (
            None if profile is None else profile.manifest()
        ),
    }


def _strict_projection_receipt(
    encoded: Any,
    native: NativeProjection,
) -> dict[str, Any]:
    from src.glm52_fresh_sqg.reference import tensor_sha256

    closure = encoded.manifest["decoded_closure"]
    source_relative_rmse = closure["source_relative_rmse"]
    if source_relative_rmse is not None:
        source_relative_rmse = float(source_relative_rmse)
    elif closure.get("full_decode_deferred") is not True:
        raise RuntimeError("candidate source RMSE is absent without deferred closure")
    return {
        "payload_sha256": _projection_payload_sha256(encoded),
        "trellis_sha256": tensor_sha256(encoded.trellis),
        "suh_sha256": tensor_sha256(encoded.suh),
        "svh_sha256": tensor_sha256(encoded.svh),
        "decoded_label_sha256": tensor_sha256(native.weight),
        "source_relative_rmse": source_relative_rmse,
        "full_decode_deferred": bool(closure.get("full_decode_deferred")),
        "hessian_proxy_error": float(encoded.manifest["encoder"]["proxy_error"]),
    }


def build_coupled_upstream_candidates(
    runtime: Any,
    weights: Any,
    permutation: Any,
    gate_profile: Any,
    down_profile: Any,
    h13: Any,
    h13_evidence: Mapping[str, Any],
    routed: Any,
    calibration_mask: np.ndarray,
    *,
    hadamard: torch.Tensor,
    device: torch.device,
    chunk_rows: int,
    kquant_runtime: Any,
    lut_by_bits: Mapping[int, torch.Tensor],
    candidate_encode_mode: str = "batch",
    rate_pairs: Sequence[tuple[int, int]] = ((3, 3), (3, 4), (4, 3), (4, 4)),
) -> CoupledUpstreamCandidates:
    """Search strict coupled scales once on fit/calibration and freeze bytes.

    Raw KQuant centers are encoded exactly once for each projection/rate.  The
    activation-parametric candidates then use the reviewed fixed-candidate
    batch API by search stage and one shared dense-H session.  An explicit
    serial diagnostic mode remains available; candidate and pair caches
    eliminate duplicate encodes and exact-path evaluations in either mode.
    """

    from src.fresh_pipeline_common import canonical_sha256 as pipeline_sha256
    from src.glm52_fresh_sqg import (
        UniformSQGCandidateRequest,
        encode_uniform_sqg,
        encode_uniform_sqg_candidate_batch,
        prepare_dense_h_session,
    )
    from src.glm52_fresh_sqg.reference import tensor_sha256

    rows = routed.row_indices[calibration_mask]
    gates = routed.applied_gates[torch.from_numpy(calibration_mask)]
    documents = routed.document_epochs[torch.from_numpy(calibration_mask)]
    if rows.size == 0 or gates.numel() != rows.size or documents.numel() != rows.size:
        raise ValueError("coupled scale fit/calibration partition differs")
    if h13_evidence.get("evidence_id") != h13.evidence_id:
        raise ValueError("coupled scale H13 evidence binding differs")
    partition = FitCalibrationPartition(
        rows=int(rows.size),
        documents=int(torch.unique(documents).numel()),
        gate_square_sum=float(gates.float().square().double().sum()),
        row_binding_sha256=pipeline_sha256(
            {
                "row_indices_sha256": tensor_sha256(
                    torch.from_numpy(np.asarray(rows, dtype=np.int64).copy())
                ),
                "document_epochs_sha256": tensor_sha256(documents),
                "applied_gates_sha256": tensor_sha256(gates),
            }
        ),
        subfold_contract_sha256=pipeline_sha256(subfold_contract(runtime.layer)),
    )
    base_configs = {
        projection: {
            bits: _config_at_rate(
                runtime,
                weights,
                projection,
                permutation,
                gate_profile,
                down_profile,
                bits=bits,
                candidate_sweep=True,
            )
            for bits in (3, 4)
        }
        for projection in ("gate_proj", "up_proj")
    }
    bindings = {
        "capture": pipeline_sha256(runtime.capture.binding()),
        "h13": str(h13.evidence_id),
        "profile": pipeline_sha256(gate_profile.manifest()),
        "source_gate": str(weights.tensor_sha256["gate_proj"]),
        "source_up": str(weights.tensor_sha256["up_proj"]),
        "config": pipeline_sha256(
            {
                projection: {
                    str(bits): _config_scale_binding(base_configs[projection][bits])
                    for bits in (3, 4)
                }
                for projection in ("gate_proj", "up_proj")
            }
        ),
        "codec": sha256_file(PROJECT_ROOT / "src/glm52_fresh_sqg/codec.py"),
        "kquant": pipeline_sha256(
            {
                "revision": kquant_runtime.revision,
                "backend_sha256": kquant_runtime.backend_sha256,
                "tracked_diff_sha256": kquant_runtime.tracked_diff_sha256,
                "status_sha256": kquant_runtime.status_sha256,
            }
        ),
        "lut_k3": tensor_sha256(lut_by_bits[3]),
        "lut_k4": tensor_sha256(lut_by_bits[4]),
        "selector_code": sha256_file(
            PROJECT_ROOT / "scripts/coupled_gate_up_scale_selector.py"
        ),
    }
    context = CoupledScaleContext(
        layer=int(runtime.layer),
        expert=int(weights.expert),
        partition=partition,
        bindings=bindings,
    )
    session = prepare_dense_h_session(h13, base_configs["gate_proj"][3])
    if candidate_encode_mode not in ("batch", "serial_diagnostic"):
        raise ValueError("coupled candidate encode mode differs")
    raw_scales: dict[tuple[str, int], float] = {}
    raw_receipts: dict[str, dict[int, dict[str, Any]]] = {
        "gate_proj": {},
        "up_proj": {},
    }
    for projection in ("gate_proj", "up_proj"):
        source = weights.gate_hf if projection == "gate_proj" else weights.up_hf
        for bits in (3, 4):
            encoded = encode_uniform_sqg(
                source,
                h13,
                base_configs[projection][bits],
                runtime=kquant_runtime,
                dense_h_session=session,
            )
            transform = encoded.manifest["transform"]
            scale = float(transform["global_scale"])
            raw_search = transform.get("global_scale_search")
            if not isinstance(raw_search, Mapping):
                raise RuntimeError("raw coupled scale center lacks search evidence")
            raw_scales[(projection, bits)] = scale
            raw_receipts[projection][bits] = {
                "scale": scale,
                "scale_hex": scale.hex(),
                "payload_sha256": _projection_payload_sha256(encoded),
                "tensor_manifest_sha256": pipeline_sha256(encoded.manifest),
                "global_scale_search": dict(raw_search),
            }
            del encoded

    encoded_by_id: dict[str, Any] = {}

    def wrap_encoded(
        projection: str,
        bits: int,
        scale: float,
        search_contract_id: str,
        encoded: Any,
    ) -> ProjectionCandidate:
        native = _native_projection(
            encoded, bits=bits, lut=lut_by_bits[bits], device=device
        )
        receipt = _strict_projection_receipt(encoded, native)
        candidate_id = pipeline_sha256(
            {
                "search_contract_id": search_contract_id,
                "projection": projection,
                "bits": bits,
                "scale_hex": scale.hex(),
                "tensor_manifest_sha256": pipeline_sha256(encoded.manifest),
                "receipt": receipt,
            }
        )
        candidate = ProjectionCandidate(
            projection=projection,
            bits=bits,
            scale=scale,
            candidate_id=candidate_id,
            native=native,
            receipt=receipt,
            override_policy="activation_parametric_coupled_v1",
            override_evidence_id=search_contract_id,
        )
        encoded_by_id[candidate_id] = encoded
        return candidate

    def encode_candidate(
        projection: str,
        bits: int,
        scale: float,
        search_contract_id: str,
    ) -> ProjectionCandidate:
        source = weights.gate_hf if projection == "gate_proj" else weights.up_hf
        config = with_activation_parametric_scale_override(
            base_configs[projection][bits],
            scale=scale,
            search_contract_id=search_contract_id,
        )
        encoded = encode_uniform_sqg(
            source,
            h13,
            config,
            runtime=kquant_runtime,
            dense_h_session=session,
        )
        return wrap_encoded(
            projection, bits, scale, search_contract_id, encoded
        )

    def encode_candidate_batch(
        requests: Sequence[tuple[str, int, float]],
        search_contract_id: str,
    ) -> list[ProjectionCandidate]:
        codec_requests = []
        for projection, bits, scale in requests:
            source = weights.gate_hf if projection == "gate_proj" else weights.up_hf
            config = with_activation_parametric_scale_override(
                base_configs[projection][bits],
                scale=scale,
                search_contract_id=search_contract_id,
            )
            codec_requests.append(
                UniformSQGCandidateRequest(source=source, config=config)
            )
        encoded_batch = encode_uniform_sqg_candidate_batch(
            codec_requests,
            h13,
            runtime=kquant_runtime,
            dense_h_session=session,
        )
        return [
            wrap_encoded(
                projection,
                bits,
                scale,
                search_contract_id,
                encoded,
            )
            for (projection, bits, scale), encoded in zip(
                requests, encoded_batch, strict=True
            )
        ]

    cached_batches: tuple[ExactEvaluationBatch, ...] = tuple(
        ExactEvaluationBatch(
            hidden=hidden,
            applied_gates=gates[begin : begin + chunk_rows].to(device),
            document_epochs=documents[begin : begin + chunk_rows].to(device),
        )
        for begin, _end, hidden in runtime.capture.iter_hidden_prefetch(
            rows, chunk_rows=chunk_rows, device=device, dtype=torch.float32
        )
    )
    evaluator = ExactGLMW4A8UpstreamEvaluator(
        batch_factory=lambda: iter(cached_batches),
        gate_hf=weights.gate_hf.to(device),
        up_hf=weights.up_hf.to(device),
        new_to_old=permutation.new_to_old.to(device),
        hadamard=hadamard,
        expected_partition=partition,
    )
    winners, scale_evidence = select_coupled_gate_up_scales(
        context=context,
        raw_scales=raw_scales,
        encode_candidate=encode_candidate,
        encode_candidate_batch=(
            encode_candidate_batch if candidate_encode_mode == "batch" else None
        ),
        evaluate_pair=evaluator,
        rate_pairs=rate_pairs,
    )
    encoded_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    native_by_pair: dict[tuple[int, int], dict[str, NativeProjection]] = {}
    for pair, (gate, up) in winners.items():
        encoded_by_pair[pair] = {
            "gate_proj": encoded_by_id[gate.candidate_id],
            "up_proj": encoded_by_id[up.candidate_id],
        }
        native_by_pair[pair] = {
            "gate_proj": gate.native,
            "up_proj": up.native,
        }
    return CoupledUpstreamCandidates(
        encoded_by_pair=encoded_by_pair,
        native_by_pair=native_by_pair,
        scale_evidence=scale_evidence,
        raw_center_receipts=raw_receipts,
    )


def _teacher_output(
    hidden: torch.Tensor, source_gpu: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    gate = torch.matmul(hidden.float(), source_gpu["gate_proj"])
    up = torch.matmul(hidden.float(), source_gpu["up_proj"])
    return torch.matmul(F.silu(gate) * up, source_gpu["down_proj"])


def _permuted_teacher_activation(
    hidden: torch.Tensor,
    source_gpu: Mapping[str, torch.Tensor],
    permutation: torch.Tensor,
) -> torch.Tensor:
    gate = torch.matmul(hidden.float(), source_gpu["gate_proj"]).index_select(
        1, permutation
    )
    up = torch.matmul(hidden.float(), source_gpu["up_proj"]).index_select(
        1, permutation
    )
    return (F.silu(gate) * up).contiguous()


def _expert_path(root: Path, layer: int, expert: int) -> Path:
    return root / f"layer_{layer:03d}" / "experts" / f"expert_{expert:03d}.json"


def _global_h13_paths(root: Path, layer: int) -> tuple[Path, Path]:
    directory = root / f"layer_{layer:03d}" / "preparation"
    return directory / "w4a8_global_h13.json", directory / "w4a8_global_h13.safetensors"


def _profile_from_selection(
    runtime: Any, selection_path: Path
) -> tuple[Any, Any, dict[str, Any]]:
    from src.fresh_pipeline_common import (
        load_json_object,
        sha256_file as pipeline_sha256_file,
    )
    from src.fresh_pipeline_runner import _load_scale_evidence, _profiles_for_cell
    from src.glm52_fresh_sqg.codec import realize_shared_residual_profile

    selection = load_json_object(selection_path)
    selected = selection.get("selected_cell")
    if (
        selection.get("complete") is not True
        or not isinstance(selected, Mapping)
        or selection.get("selected_cell_id") != selected.get("cell_id")
        or selection.get("holdout_used_for_choice") is not False
        or int(selection.get("mcg_inputs", -1)) != 0
    ):
        raise ValueError("frozen profile selection contract differs")
    scales = _load_scale_evidence(runtime)
    gate_profile, down_profile = _profiles_for_cell(
        runtime,
        scales,
        draw=int(selected["draw"]),
        family=str(selected["family"]),
    )
    for profile, key in (
        (gate_profile, "gate_up_input_profile"),
        (down_profile, "down_output_profile"),
    ):
        expected = selected.get(key)
        if expected is not None and profile.manifest() != expected:
            actual = profile.manifest()
            differing = {
                field: {"expected": expected.get(field), "actual": actual.get(field)}
                for field in sorted(set(expected) | set(actual))
                if expected.get(field) != actual.get(field)
            }
            raise ValueError(
                f"selected {key} reconstruction differs: "
                f"{json.dumps(differing, sort_keys=True)}"
            )
    gate_profile = realize_shared_residual_profile(gate_profile, runtime.device)
    down_profile = realize_shared_residual_profile(down_profile, runtime.device)
    evidence = {
        "path": str(selection_path.resolve()),
        "sha256": pipeline_sha256_file(selection_path),
        "selection_id": str(
            selection.get("selection_id", selection["selected_cell_id"])
        ),
        "selected_cell_id": str(selection["selected_cell_id"]),
        "draw": int(selected["draw"]),
        "family": str(selected["family"]),
        "profile_selection_precedes_rate_allocation": True,
    }
    return gate_profile, down_profile, evidence


def _prepare_global_h13(args: argparse.Namespace) -> None:
    from bmmlaw_r7_encoder.safetensors_io import (
        torch_tensor_entry,
        write_safetensors_atomic,
    )
    from scripts.encode_final_shard import _open_fast_sealed_runtime
    from src.fresh_pipeline_common import atomic_json
    from src.glm52_fresh_sqg.reference import normalized_hadamard, tensor_sha256

    runtime = _open_fast_sealed_runtime(
        args.preflight, layer=args.layer, device=args.device
    )
    gate_profile, _, selection_evidence = _profile_from_selection(
        runtime, args.profile_selection.resolve()
    )
    manifest_path, shard_path = _global_h13_paths(
        args.output_root.resolve(), args.layer
    )
    if manifest_path.exists() or shard_path.exists():
        raise ValueError("global W4A8 H13 artifact already exists")
    device = torch.device(args.device)
    hadamard = normalized_hadamard(
        device=device, dtype=torch.float32, size=HADAMARD_BLOCK
    )
    suh = gate_profile.expected_stored_fp16().to(device)
    rows = runtime.capture.role_rows("fit")
    docs = np.asarray(runtime.capture.doc_epochs[rows], dtype=np.int64)
    mask = fit_subfold_mask(docs, layer=args.layer, subfold="calibration")
    selected_rows = rows[mask]
    if selected_rows.size == 0:
        raise ValueError("fit/calibration has no layer rows")
    accumulator = torch.zeros((HIDDEN, HIDDEN), dtype=torch.float32, device=device)
    denominator = torch.zeros((), dtype=torch.float64, device=device)
    for begin, end, hidden in runtime.capture.iter_hidden_prefetch(
        selected_rows,
        chunk_rows=args.chunk_rows,
        device=device,
        dtype=torch.float32,
    ):
        local = selected_rows[begin:end]
        q_label, observation, _ = prepare_gate_up_operand(
            hidden, suh, hadamard, quantize_a8=True
        )
        if observation is None or bool(observation.preclamp_overflow.any()):
            raise RuntimeError("global H13 h-A8 path overflowed")
        q_pre = kquant_prefinalize_operand_from_label_operand(
            q_label, gate_profile.signs, hadamard, validate_values=False
        )
        gates = torch.from_numpy(
            np.array(runtime.capture.topk_weights[local], dtype=np.float32, copy=True)
        ).to(device)
        importance = gates.square().sum(dim=1)
        accumulator.addmm_(q_pre.T, q_pre * importance[:, None])
        denominator.add_(importance.double().sum())
        del hidden, q_label, observation, q_pre, gates, importance
    if not bool(torch.isfinite(denominator)) or not bool(denominator > 0):
        raise ValueError("global W4A8 H13 has no routed mass")
    hessian = accumulator.div_(denominator.to(torch.float32))
    # Keep the numerical artifact resident on the execution device.  The
    # safetensors writer owns the only host copy below as an explicit file-I/O
    # boundary; hashing likewise owns its temporary host staging.
    hessian = ((hessian + hessian.T) * 0.5).contiguous()
    evidence: dict[str, Any] = {
        "schema": GLOBAL_H13_SCHEMA,
        "complete": True,
        "layer": args.layer,
        "role": "fit",
        "subfold": "calibration",
        "subfold_contract": subfold_contract(args.layer),
        "construction": GLOBAL_H13_CONSTRUCTION,
        "profile_selection": selection_evidence,
        "gate_input_profile": gate_profile.manifest(),
        "rows": int(selected_rows.size),
        "documents": int(np.unique(docs[mask]).size),
        "gate_square_sum": float(denominator),
        "matrix_sha256": tensor_sha256(hessian),
        "capture": runtime.capture.binding(),
        "selection_rows_used": False,
        "holdout_used": False,
    }
    evidence["evidence_id"] = canonical_sha256(evidence)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    payloads, shard_hash = write_safetensors_atomic(
        shard_path,
        [torch_tensor_entry("h13", hessian)],
        metadata={
            "format": "pt",
            "schema": GLOBAL_H13_SCHEMA,
            "layer": str(args.layer),
        },
    )
    evidence["shard"] = shard_path.name
    evidence["shard_sha256"] = shard_hash
    evidence["payload_sha256"] = payloads
    atomic_json(manifest_path, evidence)


def _load_global_h13(
    root: Path,
    layer: int,
    *,
    device: str | torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    from safetensors import safe_open
    from src.glm52_fresh_sqg.reference import tensor_sha256

    manifest_path, shard_path = _global_h13_paths(root, layer)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != GLOBAL_H13_SCHEMA
        or value.get("complete") is not True
        or int(value.get("layer", -1)) != layer
        or value.get("role") != "fit"
        or value.get("subfold") != "calibration"
        or value.get("selection_rows_used") is not False
        or value.get("holdout_used") is not False
        or sha256_file(shard_path) != value.get("shard_sha256")
    ):
        raise ValueError("global W4A8 H13 binding differs")
    numerical_device = torch.device(device)
    if numerical_device.type != "cuda":
        raise ValueError("production global W4A8 H13 must load directly on CUDA")
    with safe_open(
        shard_path, framework="pt", device=str(numerical_device)
    ) as handle:
        if set(handle.keys()) != {"h13"}:
            raise ValueError("global W4A8 H13 tensor census differs")
        matrix = handle.get_tensor("h13").float().contiguous()
    if matrix.device != numerical_device:
        raise RuntimeError("global W4A8 H13 loader staged through a non-CUDA device")
    if tensor_sha256(matrix) != value.get("matrix_sha256"):
        raise ValueError("global W4A8 H13 payload differs")
    return matrix, value


def _build_expert_h13(
    runtime: Any,
    *,
    expert: int,
    global_h13: torch.Tensor,
    global_evidence: Mapping[str, Any],
    gate_profile: Any,
    hadamard: torch.Tensor,
    chunk_rows: int,
    device: torch.device,
    defer_hashes: bool = True,
) -> tuple[Any, dict[str, Any], Any, Any]:
    from src.fresh_pipeline_common import canonical_sha256 as pipeline_canonical_sha256
    from src.glm52_fresh_sqg import DenseHessian
    from src.glm52_fresh_sqg.reference import tensor_sha256

    routed = runtime.capture.routed_rows(expert, "fit")
    mask = fit_subfold_mask(
        routed.document_epochs, layer=runtime.layer, subfold="calibration"
    )
    rows = routed.row_indices[mask]
    gates_gpu = routed.applied_gates[torch.from_numpy(mask)].to(device)
    docs = routed.document_epochs[torch.from_numpy(mask)]
    if rows.size == 0:
        raise ValueError(f"expert {expert}: fit/calibration has no routed rows")
    local = torch.zeros((HIDDEN, HIDDEN), dtype=torch.float32, device=device)
    denominator = torch.zeros((), dtype=torch.float64, device=device)
    suh = gate_profile.expected_stored_fp16().to(device)
    for begin, end, hidden in runtime.capture.iter_hidden_prefetch(
        rows, chunk_rows=chunk_rows, device=device, dtype=torch.float32
    ):
        q_label, observation, _ = prepare_gate_up_operand(
            hidden, suh, hadamard, quantize_a8=True
        )
        if observation is None or bool(observation.preclamp_overflow.any()):
            raise RuntimeError(f"expert {expert}: local H13 h-A8 overflowed")
        q_pre = kquant_prefinalize_operand_from_label_operand(
            q_label, gate_profile.signs, hadamard, validate_values=False
        )
        importance = gates_gpu[begin:end].square()
        local.addmm_(q_pre.T, q_pre * importance[:, None])
        denominator.add_(importance.double().sum())
        del hidden, q_label, observation, q_pre, importance
    if not bool(torch.isfinite(denominator)) or not bool(denominator > 0):
        raise ValueError(f"expert {expert}: local H13 has no positive routed mass")
    local.div_(denominator.to(torch.float32))
    local = ((local + local.T) * 0.5).contiguous()
    blended = torch.lerp(global_h13.to(device), local, LOCAL_H13_ALPHA)
    blended = ((blended + blended.T) * 0.5).contiguous()
    evidence: dict[str, Any] = {
        "construction": H13_CONSTRUCTION,
        "role": "fit",
        "subfold": "calibration",
        "layer": runtime.layer,
        "expert": expert,
        "rows": int(rows.size),
        "documents": int(torch.unique(docs).numel()),
        "gate_square_sum": float(denominator),
        "global_alpha": 1.0 - LOCAL_H13_ALPHA,
        "local_alpha": LOCAL_H13_ALPHA,
        "global_evidence_id": global_evidence["evidence_id"],
        "matrix_sha256": (
            None if defer_hashes else tensor_sha256(blended)
        ),
        "candidate_hashes_deferred": defer_hashes,
        "gate_input_profile": gate_profile.manifest(),
        "selection_rows_used": False,
        "holdout_used": False,
    }
    candidate_evidence_id = pipeline_canonical_sha256(
        deferred_h13_evidence(evidence)
    )
    evidence_id = pipeline_canonical_sha256(evidence)
    evidence["evidence_id"] = evidence_id
    if not defer_hashes:
        evidence["candidate_evidence_id"] = candidate_evidence_id
    dense = DenseHessian(
        matrix=blended,
        evidence_id=evidence_id,
        construction=H13_CONSTRUCTION,
        split_id="fit",
        normalization_count=1,
        routed_sample_count=int(rows.size),
        matrix_sha256=evidence["matrix_sha256"],
    )
    return dense, evidence, routed, mask


def _preliminary_down_anchors(
    runtime: Any,
    weights: Any,
    permutation: Any,
    gate_profile: Any,
    down_profile: Any,
    routed: Any,
    calibration_mask: np.ndarray,
    source_gpu: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    chunk_rows: int,
    kquant_runtime: Any,
) -> tuple[dict[int, Any], dict[str, Any]]:
    from src.calibration_hessian import apply_frozen_h2_shrinkage
    from src.fresh_pipeline_common import canonical_sha256 as pipeline_canonical_sha256
    from src.glm52_fresh_sqg import (
        DenseHessian,
        encode_uniform_sqg,
        prepare_dense_h_session,
    )
    rows = routed.row_indices[calibration_mask]
    gates = routed.applied_gates[torch.from_numpy(calibration_mask)]
    permutation_gpu = permutation.new_to_old.to(device)
    if set(source_gpu) != {"gate_proj", "up_proj", "down_proj"}:
        raise ValueError("source GPU cache has an unexpected projection census")
    raw = torch.zeros((INTERMEDIATE, INTERMEDIATE), dtype=torch.float32, device=device)
    importance_chunks: list[torch.Tensor] = []
    denominator = torch.zeros((), dtype=torch.float64, device=device)
    for begin, end, hidden in runtime.capture.iter_hidden_prefetch(
        rows, chunk_rows=chunk_rows, device=device, dtype=torch.float32
    ):
        act = _permuted_teacher_activation(hidden, source_gpu, permutation_gpu)
        importance = gates[begin:end].to(device).square()
        raw.addmm_(act.T, act * importance[:, None])
        importance_chunks.append(importance)
        denominator.add_(importance.double().sum())
        del hidden, act, importance
    raw.div_(denominator.to(torch.float32))
    raw = ((raw + raw.T) * 0.5).contiguous()
    h2, shrinkage = apply_frozen_h2_shrinkage(raw, torch.cat(importance_chunks))
    evidence: dict[str, Any] = {
        "construction": PRELIMINARY_H2_CONSTRUCTION,
        "role": "fit",
        "subfold": "calibration",
        "layer": runtime.layer,
        "expert": int(weights.expert),
        "rows": int(rows.size),
        "gate_square_sum": float(denominator),
        "matrix_sha256": None,
        "candidate_hashes_deferred": True,
        "purpose": "derive independent K3/K4 private down suh anchors",
        "shrinkage": {key: float(value) for key, value in shrinkage.items()},
        "selection_rows_used": False,
        "holdout_used": False,
    }
    evidence_id = pipeline_canonical_sha256(evidence)
    dense = DenseHessian(
        matrix=h2.contiguous(),
        evidence_id=evidence_id,
        construction=PRELIMINARY_H2_CONSTRUCTION,
        split_id="fit",
        normalization_count=1,
        routed_sample_count=int(rows.size),
    )
    base = _config_at_rate(
        runtime,
        weights,
        "down_proj",
        permutation,
        gate_profile,
        down_profile,
        bits=3,
        candidate_sweep=True,
    )
    # K3 and K4 use the same preliminary H2 and input-transform binding.  A
    # DenseHSession is deliberately rate-neutral, so factorize this 2048^2
    # matrix once and reuse the exact finalized BlockLDL state for both rates.
    session = prepare_dense_h_session(dense, replace(base, bits=3))
    encoded = {
        bits: encode_uniform_sqg(
            weights.down_hf,
            dense,
            replace(base, bits=bits),
            runtime=kquant_runtime,
            dense_h_session=session,
        )
        for bits in (3, 4)
    }
    return encoded, {**evidence, "evidence_id": evidence_id}


def _fit_and_encode_down_candidate_grid(
    runtime: Any,
    weights: Any,
    permutation: Any,
    down_profile: Any,
    routed: Any,
    calibration_mask: np.ndarray,
    upstream_by_pair: Mapping[
        tuple[int, int], Mapping[str, NativeProjection]
    ],
    scale_choice_by_pair: Mapping[tuple[int, int], Mapping[str, Any]],
    preliminary_down: Mapping[int, Any],
    source_gpu: Mapping[str, torch.Tensor],
    *,
    beta: float,
    device: torch.device,
    hadamard: torch.Tensor,
    chunk_rows: int,
    kquant_runtime: Any,
    lut_by_bits: Mapping[int, torch.Tensor],
) -> tuple[
    dict[tuple[int, int], dict[int, Any]],
    dict[tuple[int, int], dict[int, NativeProjection]],
    dict[tuple[int, int], dict[int, dict[str, Any]]],
]:
    """Fit and encode all eight candidate-specific down projections.

    The old loop reread every calibration row and recomputed the BF16 teacher,
    h-A8 operand, and upstream candidate GEMMs once per gate/up rate pair.
    Here those common operations execute once per row chunk.  Each of the
    eight rate-triplet cells still receives its own exact activation, down-rate
    anchor, canonical ``(H,B)``, q-pre encoder Hessian, target solve, and SQG
    encode; only byte-neutral common work is shared.
    """

    from src.calibration_hessian import apply_frozen_h2_shrinkage
    from src.fresh_pipeline_common import canonical_sha256 as pipeline_canonical_sha256
    from src.glm52_fresh_sqg import (
        DenseHessian,
        SharedResidualProfile,
        W4A8DerivedTensorBinding,
        encode_uniform_sqg,
    )
    from src.glm52_fresh_sqg.codec import realize_shared_residual_profile
    from src.glm52_fresh_sqg.reference import tensor_sha256

    rows = routed.row_indices[calibration_mask]
    gates_gpu = routed.applied_gates[torch.from_numpy(calibration_mask)].to(device)
    if set(source_gpu) != {"gate_proj", "up_proj", "down_proj"}:
        raise ValueError("source GPU cache has an unexpected projection census")
    official_down_exl = (
        weights.down_hf.index_select(
            1, permutation.new_to_old.to(device=weights.down_hf.device)
        ).T.float().to(device)
    ).contiguous()
    anchor_profiles: dict[int, Any] = {}
    for bits in (3, 4):
        anchor = SharedResidualProfile.from_stored_vector(
            preliminary_down[bits].suh,
            side="input",
            profile_id=(
                f"layer-{runtime.layer:03d}/expert-{int(weights.expert):03d}/"
                f"down-k{bits}-w4a8-allocation-anchor-v1"
            ),
            derivation="fit_calibration_official_bf16_preliminary_down_encode",
        )
        anchor_profiles[bits] = realize_shared_residual_profile(anchor, runtime.device)

    pairs = tuple((gate_bits, up_bits) for gate_bits in (3, 4) for up_bits in (3, 4))
    if set(upstream_by_pair) != set(pairs):
        raise ValueError("down grid requires all four coupled upstream rate pairs")
    if set(scale_choice_by_pair) != set(pairs):
        raise ValueError("down grid requires all four coupled scale choices")
    cells = tuple((*pair, down_bits) for pair in pairs for down_bits in (3, 4))
    raw_h = {
        cell: torch.zeros(
            (INTERMEDIATE, INTERMEDIATE), dtype=torch.float32, device=device
        )
        for cell in cells
    }
    raw_encoder_h = {cell: torch.zeros_like(raw_h[cell]) for cell in cells}
    raw_b = {
        cell: torch.zeros((INTERMEDIATE, HIDDEN), dtype=torch.float32, device=device)
        for cell in cells
    }
    anchor_vectors = {
        bits: anchor_profiles[bits].expected_stored_fp16().to(device) for bits in (3, 4)
    }
    denominator = torch.zeros((), dtype=torch.float64, device=device)
    teacher_energy = torch.zeros((), dtype=torch.float64, device=device)
    importance_chunks: list[torch.Tensor] = []
    for begin, end, hidden in runtime.capture.iter_hidden_prefetch(
        rows, chunk_rows=chunk_rows, device=device, dtype=torch.float32
    ):
        activations = execute_full_w4a8_upstream_pairs(
            hidden,
            upstream_by_pair,
            hadamard,
        )
        teacher = _teacher_output(hidden, source_gpu)
        importance = gates_gpu[begin:end].square()
        importance_chunks.append(importance)
        for pair, activation in activations.items():
            for bits in (3, 4):
                cell = (*pair, bits)
                anchor = anchor_profiles[bits]
                q_label, observation, _ = prepare_down_operand(
                    activation,
                    anchor_vectors[bits],
                    hadamard,
                    quantize_a8=True,
                )
                if observation is None or bool(observation.preclamp_overflow.any()):
                    raise RuntimeError(
                        f"candidate-specific down K{bits} act-A8 overflowed"
                    )
                q_eff = effective_canonical_operand_from_quantized_transform(
                    q_label,
                    anchor_vectors[bits],
                    hadamard,
                    validate_values=False,
                )
                q_pre = kquant_prefinalize_operand_from_label_operand(
                    q_label,
                    anchor.signs,
                    hadamard,
                    validate_values=False,
                )
                raw_h[cell].addmm_(q_eff.T, q_eff * importance[:, None])
                raw_b[cell].addmm_(q_eff.T, teacher * importance[:, None])
                raw_encoder_h[cell].addmm_(q_pre.T, q_pre * importance[:, None])
                del q_label, observation, q_eff, q_pre
        denominator.add_(importance.double().sum())
        teacher_energy.add_(
            torch.sum(teacher.square().sum(dim=1) * importance, dtype=torch.float64)
        )
        del hidden, activations, teacher, importance

    encoded: dict[tuple[int, int], dict[int, Any]] = {pair: {} for pair in pairs}
    native: dict[tuple[int, int], dict[int, NativeProjection]] = {
        pair: {} for pair in pairs
    }
    evidences: dict[tuple[int, int], dict[int, dict[str, Any]]] = {
        pair: {} for pair in pairs
    }
    base_config = _config_at_rate(
        runtime,
        weights,
        "down_proj",
        permutation,
        None,
        down_profile,
        bits=3,
        candidate_sweep=True,
    )
    official_binding = base_config.source_binding
    importance_vector = torch.cat(importance_chunks)
    for gate_bits, up_bits, bits in cells:
        cell = (gate_bits, up_bits, bits)
        for matrix in (raw_h[cell], raw_b[cell], raw_encoder_h[cell]):
            matrix.div_(denominator.to(torch.float32))
        canonical_h_raw = raw_h.pop(cell)
        canonical_h = ((canonical_h_raw + canonical_h_raw.T) * 0.5).contiguous()
        encoder_h_unscaled = raw_encoder_h.pop(cell)
        encoder_h_raw = ((encoder_h_unscaled + encoder_h_unscaled.T) * 0.5).contiguous()
        cross_term = raw_b.pop(cell)
        identity_scale = canonical_h.diagonal().double().mean()
        stats = CrossTermStatistics(
            hessian=canonical_h,
            cross_term=cross_term,
            weight_sum=denominator,
            rows=int(rows.size),
            validate_values=False,
        )
        fitted = shrink_cross_term_objective(
            stats,
            official_down_exl,
            local_alpha=beta,
            identity_scale=identity_scale,
        )
        target, solver = solve_with_minimal_official_prior(
            fitted,
            official_down_exl,
            identity_scale=identity_scale,
        )
        # The fitted canonical target uses the explicit preregistered beta.
        # The BlockLDLQ metric is a separate coordinate problem: shrink the
        # exact q_pre covariance with the frozen reliability policy, then let
        # KQuant finalize it once into the native-label covariance.
        encoder_h, encoder_shrinkage = apply_frozen_h2_shrinkage(
            encoder_h_raw,
            importance_vector,
        )
        encoder_h = encoder_h.contiguous()
        target = target.contiguous()
        contract = {
            "schema": "glm52-full-w4a8-triplet-candidate-execution-v1",
            "activation_endpoint": FULL_W4A8_ENDPOINT,
            "h_a8": "mxfp8_e4m3_ue8m0_k32",
            "weight_endpoint": "native_finite_e4m3_sqg_labels",
            "nonlinearity": "torch_silu_gate_times_up_with_fp16_inter_gemm_boundary",
            "act_a8": "mxfp8_e4m3_ue8m0_k32",
            "down_input_anchor_sha256": tensor_sha256(
                anchor_profiles[bits].expected_stored_fp16()
            ),
            "down_output_profile": down_profile.manifest(),
            "h_b_beta": beta,
            "coupled_upstream_scale_choice_id": scale_choice_by_pair[
                (gate_bits, up_bits)
            ]["choice_id"],
            "selection_rows_used": False,
            "holdout_used": False,
        }
        contract_id = pipeline_canonical_sha256(contract)
        evidence: dict[str, Any] = {
            "schema": "glm52-sqg-w4a8-triplet-down-objective-v1",
            "layer": runtime.layer,
            "expert": int(weights.expert),
            "down_bits": bits,
            "role": "fit",
            "subfold": "calibration",
            "beta": beta,
            "rows": int(rows.size),
            "gate_square_sum": float(denominator),
            "teacher_energy": float(teacher_energy),
            "canonical_h_sha256": None,
            "cross_term_sha256": None,
            "encoder_h_sha256": None,
            "encoder_h_shrinkage": {
                key: float(value) for key, value in encoder_shrinkage.items()
            },
            "derived_target_sha256": None,
            "candidate_hashes_deferred": True,
            "execution_contract": contract,
            "execution_contract_id": contract_id,
            "coupled_upstream_scale_choice_id": scale_choice_by_pair[
                (gate_bits, up_bits)
            ]["choice_id"],
            "solver": solver,
            "selection_rows_used": False,
            "holdout_used": False,
        }
        evidence = deferred_down_objective_evidence(evidence)
        evidence_id = pipeline_canonical_sha256(evidence)
        evidence["evidence_id"] = evidence_id
        binding = W4A8DerivedTensorBinding(
            official_bf16_parent=official_binding,
            execution_contract=contract,
            execution_contract_sha256=contract_id,
            fit_hb_evidence_id=evidence_id,
            fit_hb_evidence_sha256=evidence_id,
            fit_hessian_sha256=None,
            fit_cross_term_sha256=None,
            beta=beta,
            derived_tensor_sha256=None,
            candidate_hashes_deferred=True,
        )
        config = replace(
            base_config,
            bits=bits,
            source_binding=binding,
            anchored_input_residual_profile=anchor_profiles[bits],
        )
        dense = DenseHessian(
            matrix=encoder_h,
            evidence_id=evidence_id,
            construction=DERIVED_H2_CONSTRUCTION,
            split_id="fit",
            normalization_count=1,
            routed_sample_count=int(rows.size),
        )
        result = encode_uniform_sqg(target, dense, config, runtime=kquant_runtime)
        if not torch.equal(result.suh, anchor_profiles[bits].expected_stored_fp16()):
            raise RuntimeError(f"derived K{bits} down encode changed its anchored suh")
        if not torch.equal(result.svh, down_profile.expected_stored_fp16()):
            raise RuntimeError(f"derived K{bits} down encode changed shared svh")
        pair = (gate_bits, up_bits)
        encoded[pair][bits] = result
        native[pair][bits] = _native_projection(
            result, bits=bits, lut=lut_by_bits[bits], device=device
        )
        evidences[pair][bits] = evidence
        del (
            canonical_h_raw,
            canonical_h,
            encoder_h_unscaled,
            encoder_h_raw,
            cross_term,
            target,
        )
    return encoded, native, evidences


def _run_worker(args: argparse.Namespace) -> None:
    from kquant.sqg_e4m3 import sqg_xor_cheb_t12_bytes
    from scripts.encode_final_shard import _open_fast_sealed_runtime
    from src.fresh_pipeline_common import (
        atomic_json,
        canonical_sha256 as pipeline_canonical_sha256,
    )
    from src.fresh_pipeline_runner import (
        _load_bound_kquant_runtime,
        _load_permutation,
    )
    from src.glm52_fresh_sqg.reference import normalized_hadamard
    import src.glm52_fresh_sqg.codec as codec

    if (
        args.start is None
        or args.end is None
        or not 0 <= args.start < args.end <= NUM_EXPERTS
    ):
        raise ValueError("worker mode requires 0 <= start < end <= 256")
    runtime = _open_fast_sealed_runtime(
        args.preflight, layer=args.layer, device=args.device
    )
    device = torch.device(args.device)
    gate_profile, down_profile, selection_evidence = _profile_from_selection(
        runtime, args.profile_selection.resolve()
    )
    global_h13, global_evidence = _load_global_h13(
        args.output_root.resolve(), args.layer, device=device
    )
    if global_evidence.get("profile_selection") != selection_evidence:
        raise ValueError("global H13 profile selection differs from worker")
    hadamard = normalized_hadamard(
        device=device, dtype=torch.float32, size=HADAMARD_BLOCK
    )
    # The 6144^2 layer covariance was loaded directly on this worker's CUDA
    # device.  Never bounce it through host memory between experts.
    kquant_runtime = _load_bound_kquant_runtime(runtime)
    lut_by_bits = {
        bits: sqg_xor_cheb_t12_bytes(bits).to(device).contiguous() for bits in (3, 4)
    }
    codec.PRODUCTION_H13_CONSTRUCTION = H13_CONSTRUCTION
    codec.PRODUCTION_H2_CONSTRUCTION = PRELIMINARY_H2_CONSTRUCTION

    for expert in range(args.start, args.end):
        output = _expert_path(args.output_root.resolve(), args.layer, expert)
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            _validate_expert_record(existing, layer=args.layer, expert=expert)
            continue
        weights = runtime.source.load_expert_bf16(args.layer, expert, device=device)
        permutation = _load_permutation(runtime, expert)
        h13, h13_evidence, routed, calibration_mask = _build_expert_h13(
            runtime,
            expert=expert,
            global_h13=global_h13,
            global_evidence=global_evidence,
            gate_profile=gate_profile,
            hadamard=hadamard,
            chunk_rows=args.chunk_rows,
            device=device,
        )
        coupled_upstream = build_coupled_upstream_candidates(
            runtime,
            weights,
            permutation,
            gate_profile,
            down_profile,
            h13,
            h13_evidence,
            routed,
            calibration_mask,
            hadamard=hadamard,
            device=device,
            chunk_rows=args.chunk_rows,
            kquant_runtime=kquant_runtime,
            lut_by_bits=lut_by_bits,
        )
        scale_choice_by_pair = {
            pair: coupled_upstream.choice_receipt(pair)
            for pair in ((3, 3), (3, 4), (4, 3), (4, 4))
        }

        source_gpu = {
            "gate_proj": weights.gate_hf.T.float().to(device),
            "up_proj": weights.up_hf.T.float().to(device),
            "down_proj": weights.down_hf.T.float().to(device),
        }
        preliminary_down, preliminary_evidence = _preliminary_down_anchors(
            runtime,
            weights,
            permutation,
            gate_profile,
            down_profile,
            routed,
            calibration_mask,
            source_gpu,
            device=device,
            chunk_rows=args.chunk_rows,
            kquant_runtime=kquant_runtime,
        )
        allocation_mask = fit_subfold_mask(
            routed.document_epochs, layer=args.layer, subfold="allocation"
        )
        allocation_rows = routed.row_indices[allocation_mask]
        allocation_gates = routed.applied_gates[torch.from_numpy(allocation_mask)]
        if allocation_rows.size == 0:
            raise ValueError(f"expert {expert}: fit/allocation has no routed rows")
        validate_route_gates_once(allocation_gates, rows=int(allocation_rows.size))
        encoded_down_grid, native_down_grid, down_evidence_grid = (
            _fit_and_encode_down_candidate_grid(
                runtime,
                weights,
                permutation,
                down_profile,
                routed,
                calibration_mask,
                coupled_upstream.native_by_pair,
                scale_choice_by_pair,
                preliminary_down,
                source_gpu,
                beta=args.beta,
                device=device,
                hadamard=hadamard,
                chunk_rows=args.chunk_rows,
                kquant_runtime=kquant_runtime,
                lut_by_bits=lut_by_bits,
            )
        )

        # Score the complete triplet grid in one pass over fit/allocation.
        # h-A8 is common to all four upstream rate pairs, and the BF16 teacher
        # is common to all eight triplets.  Candidate-specific act-A8/down
        # execution remains separate, which preserves the exact unary loss.
        losses = {
            (gate_bits, up_bits, down_bits): torch.zeros(
                (), dtype=torch.float64, device=device
            )
            for gate_bits in (3, 4)
            for up_bits in (3, 4)
            for down_bits in (3, 4)
        }
        teacher_energy = torch.zeros((), dtype=torch.float64, device=device)
        for begin, end, hidden in runtime.capture.iter_hidden_prefetch(
            allocation_rows,
            chunk_rows=args.chunk_rows,
            device=device,
            dtype=torch.float32,
        ):
            teacher = _teacher_output(hidden, source_gpu)
            route_gates = allocation_gates[begin:end].to(device)
            teacher_energy.add_(
                torch.sum(
                    teacher.square().sum(dim=1) * route_gates.square(),
                    dtype=torch.float64,
                )
            )
            activations = execute_full_w4a8_upstream_pairs(
                hidden,
                coupled_upstream.native_by_pair,
                hadamard,
            )
            for pair, activation in activations.items():
                for down_bits in (3, 4):
                    candidate_output = execute_full_w4a8_down(
                        activation,
                        native_down_grid[pair][down_bits],
                        hadamard,
                    )
                    cell = (*pair, down_bits)
                    losses[cell].add_(
                        gate_square_weighted_complete_expert_sse(
                            candidate_output, teacher, route_gates
                        )
                    )
                    del candidate_output
            del hidden, teacher, route_gates, activations

        down_payload_sha256 = {
            pair: {
                bits: _projection_payload_sha256(encoded_down_grid[pair][bits])
                for bits in (3, 4)
            }
            for pair in encoded_down_grid
        }
        ordered_cells = tuple(
            (gate_bits, up_bits, down_bits)
            for gate_bits in (3, 4)
            for up_bits in (3, 4)
            for down_bits in (3, 4)
        )
        frozen_score_vector = torch.stack(
            [losses[cell] for cell in ordered_cells] + [teacher_energy]
        ).detach().cpu().tolist()
        validate_frozen_sse_vector(frozen_score_vector, expected=9)
        frozen_losses = dict(zip(ordered_cells, frozen_score_vector[:-1], strict=True))
        frozen_teacher_energy = frozen_score_vector[-1]
        allocation_gate_square_sum = float(allocation_gates.double().square().sum())
        candidates: list[dict[str, Any]] = []
        for gate_bits in (3, 4):
            for up_bits in (3, 4):
                pair = (gate_bits, up_bits)
                down_evidence = down_evidence_grid[pair]
                scale_choice = scale_choice_by_pair[pair]
                upstream_evidence = {
                    "layer": args.layer,
                    "expert": expert,
                    "gate_bits": gate_bits,
                    "up_bits": up_bits,
                    "gate_payload_sha256": scale_choice["gate"]["payload_sha256"],
                    "up_payload_sha256": scale_choice["up"]["payload_sha256"],
                    "h13_evidence_id": h13.evidence_id,
                    "coupled_scale_choice_id": scale_choice["choice_id"],
                    "coupled_scale_evidence_id": scale_choice[
                        "scale_evidence_id"
                    ],
                }
                upstream_id = pipeline_canonical_sha256(upstream_evidence)
                for down_bits in (3, 4):
                    rates = {
                        "gate_proj": gate_bits,
                        "up_proj": up_bits,
                        "down_proj": down_bits,
                    }
                    payloads = {
                        "gate_proj": upstream_evidence["gate_payload_sha256"],
                        "up_proj": upstream_evidence["up_payload_sha256"],
                        "down_proj": down_payload_sha256[pair][down_bits],
                    }
                    candidate_material = {
                        "layer": args.layer,
                        "expert": expert,
                        "rates": rates,
                        "payload_sha256": payloads,
                        "execution_contract_id": down_evidence[down_bits][
                            "execution_contract_id"
                        ],
                        "h13_evidence_id": h13.evidence_id,
                        "upstream_candidate_id": upstream_id,
                        "coupled_scale_choice_id": scale_choice["choice_id"],
                        "down_objective_evidence_id": down_evidence[down_bits][
                            "evidence_id"
                        ],
                    }
                    candidate_id = pipeline_canonical_sha256(candidate_material)
                    candidates.append(
                        {
                            "candidate_id": candidate_id,
                            "rates": rates,
                            "k4_count": sum(rate == 4 for rate in rates.values()),
                            "realized_w4a8_fit_sse": frozen_losses[
                                (*pair, down_bits)
                            ],
                            "realized_w4a8_fit_teacher_energy": frozen_teacher_energy,
                            "full_w4a8_realized": True,
                            "payload_sha256": payloads,
                            "execution_contract_id": down_evidence[down_bits][
                                "execution_contract_id"
                            ],
                            "h13_evidence_id": h13.evidence_id,
                            "upstream_candidate_id": upstream_id,
                            "coupled_scale_choice": scale_choice,
                            "down_objective_evidence_id": down_evidence[down_bits][
                                "evidence_id"
                            ],
                            "fit_subfold": "allocation",
                            "allocation_rows": int(allocation_rows.size),
                            "allocation_gate_square_sum": allocation_gate_square_sum,
                            "selection_used": False,
                            "holdout_used": False,
                        }
                    )

        record: dict[str, Any] = {
            "schema": EXPERT_SCHEMA,
            "complete": True,
            "layer": args.layer,
            "expert": expert,
            "role": "fit",
            "calibration_subfold": "calibration",
            "allocation_subfold": "allocation",
            "subfold_contract": subfold_contract(args.layer),
            "loss_definition": TRIPLET_LOSS_DEFINITION,
            "activation_endpoint": FULL_W4A8_ENDPOINT,
            "h13_evidence": h13_evidence,
            "coupled_scale_selection": coupled_upstream.scale_evidence,
            "coupled_raw_scale_centers": coupled_upstream.raw_center_receipts,
            "preliminary_down_anchor_evidence": preliminary_evidence,
            "profile_selection": selection_evidence,
            "beta": args.beta,
            "candidates": candidates,
            "selection_used": False,
            "holdout_used": False,
            "mcg_payloads_transforms_scales_or_rate_map_used": False,
        }
        record["score_id"] = canonical_sha256(record)
        _validate_expert_record(record, layer=args.layer, expert=expert)
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, record)
        del (
            weights,
            h13,
            coupled_upstream,
            preliminary_down,
            encoded_down_grid,
            native_down_grid,
            down_evidence_grid,
            source_gpu,
            losses,
            scale_choice_by_pair,
            down_payload_sha256,
        )
        print(f"layer {args.layer} triplet scores: expert {expert + 1}/256", flush=True)


def _validate_expert_record(
    value: Mapping[str, Any], *, layer: int, expert: int
) -> None:
    if (
        value.get("schema") != EXPERT_SCHEMA
        or value.get("complete") is not True
        or int(value.get("layer", -1)) != layer
        or int(value.get("expert", -1)) != expert
        or value.get("role") != "fit"
        or value.get("calibration_subfold") != "calibration"
        or value.get("allocation_subfold") != "allocation"
        or value.get("loss_definition") != TRIPLET_LOSS_DEFINITION
        or value.get("activation_endpoint") != FULL_W4A8_ENDPOINT
        or value.get("selection_used") is not False
        or value.get("holdout_used") is not False
        or value.get("mcg_payloads_transforms_scales_or_rate_map_used") is not False
    ):
        raise ValueError(f"layer {layer} expert {expert}: triplet score record differs")
    candidates = value.get("candidates")
    scale_selection = value.get("coupled_scale_selection")
    if (
        not isinstance(scale_selection, Mapping)
        or scale_selection.get("complete") is not True
        or scale_selection.get("selection_used") is not False
        or scale_selection.get("holdout_used") is not False
        or scale_selection.get("fit_allocation_used") is not False
    ):
        raise ValueError(
            f"layer {layer} expert {expert}: coupled scale selection differs"
        )
    if not isinstance(candidates, list) or len(candidates) != 8:
        raise ValueError(
            f"layer {layer} expert {expert}: eight candidates are required"
        )
    candidate_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise TypeError(
                f"layer {layer} expert {expert}: candidate is not an object"
            )
        candidate_id = candidate.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or len(candidate_id) != 64
            or any(character not in "0123456789abcdef" for character in candidate_id)
            or candidate_id in candidate_ids
        ):
            raise ValueError(f"layer {layer} expert {expert}: candidate ID differs")
        candidate_ids.add(candidate_id)
        rates = candidate.get("rates")
        if not isinstance(rates, Mapping) or set(rates) != set(TRIPLET_PROJECTIONS):
            raise ValueError(f"layer {layer} expert {expert}: candidate rates differ")
        loss = float(candidate.get("realized_w4a8_fit_sse", math.nan))
        if (
            not math.isfinite(loss)
            or loss < 0
            or candidate.get("full_w4a8_realized") is not True
        ):
            raise ValueError(f"layer {layer} expert {expert}: realized score differs")
        payloads = candidate.get("payload_sha256")
        if not isinstance(payloads, Mapping) or set(payloads) != set(
            TRIPLET_PROJECTIONS
        ):
            raise ValueError(f"layer {layer} expert {expert}: payload grid differs")
        rates = candidate["rates"]
        scale_choice = validate_coupled_scale_choice(
            candidate.get("coupled_scale_choice", {}),
            gate_bits=int(rates["gate_proj"]),
            up_bits=int(rates["up_proj"]),
        )
        if (
            scale_choice["scale_evidence_id"]
            != scale_selection.get("evidence_id")
            or payloads["gate_proj"]
            != scale_choice["gate"]["payload_sha256"]
            or payloads["up_proj"] != scale_choice["up"]["payload_sha256"]
        ):
            raise ValueError(
                f"layer {layer} expert {expert}: pair-specific scale payload differs"
            )
        for evidence in (
            *payloads.values(),
            candidate.get("execution_contract_id"),
            candidate.get("h13_evidence_id"),
            candidate.get("upstream_candidate_id"),
            candidate.get("down_objective_evidence_id"),
        ):
            if (
                not isinstance(evidence, str)
                or len(evidence) != 64
                or any(character not in "0123456789abcdef" for character in evidence)
            ):
                raise ValueError(
                    f"layer {layer} expert {expert}: evidence hash differs"
                )
    observed = {
        tuple(int(candidate["rates"][projection]) for projection in TRIPLET_PROJECTIONS)
        for candidate in candidates
    }
    if observed != set(enumerate_rate_triplets()):
        raise ValueError(f"layer {layer} expert {expert}: triplet grid differs")
    expected_id = value.get("score_id")
    body = dict(value)
    body.pop("score_id", None)
    if expected_id != canonical_sha256(body):
        raise ValueError(f"layer {layer} expert {expert}: score ID differs")


def finalize_score_manifest(root: Path, *, layer: int) -> dict[str, Any]:
    experts: dict[str, Any] = {}
    receipts: list[dict[str, Any]] = []
    beta: float | None = None
    profile_selection: Mapping[str, Any] | None = None
    for expert in range(NUM_EXPERTS):
        path = _expert_path(root, layer, expert)
        value = json.loads(path.read_text(encoding="utf-8"))
        _validate_expert_record(value, layer=layer, expert=expert)
        observed_beta = float(value["beta"])
        if beta is None:
            beta = observed_beta
            profile_selection = value["profile_selection"]
        elif observed_beta != beta or value["profile_selection"] != profile_selection:
            raise ValueError("expert score beta/profile selection differs")
        experts[str(expert)] = {
            "expert": expert,
            "candidates": value["candidates"],
        }
        receipts.append(
            {
                "expert": expert,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "score_id": value["score_id"],
            }
        )
    result: dict[str, Any] = {
        "schema": TRIPLET_SCORE_SCHEMA,
        "complete": True,
        "layer": layer,
        "role": "fit",
        "selection_used": False,
        "holdout_used": False,
        "loss_definition": TRIPLET_LOSS_DEFINITION,
        "activation_endpoint": FULL_W4A8_ENDPOINT,
        "signed_top8_used_for_candidate_choice": False,
        "subfold_contract": subfold_contract(layer),
        "calibration_subfold": "fit/calibration",
        "allocation_subfold": "fit/allocation",
        "h13": {
            "profile_specific_exact_h_a8_qpre": True,
            "global_alpha": 1.0 - LOCAL_H13_ALPHA,
            "local_alpha": LOCAL_H13_ALPHA,
        },
        "down": {
            "candidate_specific_h_b": True,
            "act_a8_in_fit_path": True,
            "private_suh_anchored_by_down_rate": True,
            "beta": beta,
        },
        "profile_selection": profile_selection,
        "expert_receipts": receipts,
        "experts": experts,
        "candidates_per_expert": 8,
        "mcg_rate_map_used": False,
        "mcg_payloads_transforms_or_scales_used": False,
    }
    result["score_manifest_id"] = canonical_sha256(result)
    _validate_triplet_scores(layer, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight")
    parser.add_argument("--profile-selection", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--beta", type=float)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-h13", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.threads <= 0 or args.chunk_rows <= 0:
        raise ValueError("threads and chunk rows must be positive")
    if args.finalize:
        output = (
            args.output_root.resolve()
            / f"layer_{args.layer:03d}"
            / "triplet_scores.json"
        )
        if output.exists():
            raise ValueError("triplet score manifest already exists")
        from src.fresh_pipeline_common import atomic_json

        result = finalize_score_manifest(args.output_root.resolve(), layer=args.layer)
        atomic_json(output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.preflight is None or args.profile_selection is None:
        raise ValueError(
            "prepare/worker mode requires --preflight and --profile-selection"
        )
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise ValueError(
            "exact SQG candidate encoding requires an explicit CUDA device"
        )
    if args.prepare_h13:
        _prepare_global_h13(args)
        return
    if args.beta is None or not math.isfinite(args.beta) or not 0.0 <= args.beta <= 1.0:
        raise ValueError("worker mode requires an explicit finite --beta in [0,1]")
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(args.threads)
    torch.set_num_threads(args.threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    _run_worker(args)
    print(
        json.dumps(
            {"complete": True, "elapsed_seconds": time.monotonic() - started},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
