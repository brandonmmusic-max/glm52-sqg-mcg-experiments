#!/usr/bin/env python3
"""Fit-only activation-parametric gate/up global-scale selection for GLM W4A8.

The search is deliberately independent of profile selection and triplet
allocation.  It accepts already-frozen raw KQuant scale centers and chooses a
possibly partner-conditional scalar for each K3/K4 gate/up rate pair by
executing the exact routed W4A8 upstream function on ``fit/calibration`` rows.
Projection candidates are cached by ``(projection, rate, float.hex(scale))``;
Cartesian cross-scoring never re-encodes a projection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import torch


SCHEMA = "glm52-full-w4a8-activation-parametric-coupled-scale-v1"
OVERRIDE_POLICY = "activation_parametric_coupled_v1"
HARD_CEILING = 4.05
CEILING_TOLERANCE = 1e-6
LOWER_BOUND = 0.10
COARSE_RATIOS = (0.70, 1.0, 1.30)
FINE_RATIOS = (math.sqrt(0.70), 1.0, math.sqrt(1.30))
RATE_PAIRS = ((3, 3), (3, 4), (4, 3), (4, 4))
TIE_BREAK = (
    "raw_gate_square_weighted_sse",
    "sum_abs_log_distance_from_independent_raw_centers",
    "gate_abs_log_distance",
    "up_abs_log_distance",
    "gate_scale_float_hex",
    "up_scale_float_hex",
    "gate_payload_sha256",
    "up_payload_sha256",
)
EXACT_ARITHMETIC = (
    "captured BF16 h -> FP16 h boundary",
    "multiply caller-owned shared gate/up suh -> FP16",
    "normalized block-H128 -> FP16",
    "per-row/per-K32 UE8M0 scaled finite-E4M3 h QDQ",
    "native finite-E4M3 SQG labels used directly",
    "FP32 gate/up GEMM accumulation -> FP16",
    "normalized block-H128 -> FP16, then caller-owned svh",
    "torch.nn.functional.silu(gate) * up -> FP16",
)
_SHA256 = set("0123456789abcdef")
REQUIRED_BINDINGS = frozenset(
    {
        "capture",
        "h13",
        "profile",
        "source_gate",
        "source_up",
        "config",
        "codec",
        "kquant",
        "lut_k3",
        "lut_k4",
        "selector_code",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256 for character in value)
    )


def _finite_positive(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


@dataclass(frozen=True)
class FitCalibrationPartition:
    rows: int
    documents: int
    gate_square_sum: float
    row_binding_sha256: str
    subfold_contract_sha256: str
    role: str = "fit"
    subfold: str = "calibration"
    selection_used: bool = False
    holdout_used: bool = False
    fit_allocation_used: bool = False

    def __post_init__(self) -> None:
        if self.role != "fit" or self.subfold != "calibration":
            raise ValueError("coupled scale selection requires fit/calibration")
        if self.selection_used or self.holdout_used or self.fit_allocation_used:
            raise ValueError("coupled scale selection forbids non-calibration rows")
        if self.rows <= 0 or self.documents <= 0:
            raise ValueError("fit/calibration partition must contain rows and documents")
        _finite_positive(self.gate_square_sum, "gate_square_sum")
        for name in ("row_binding_sha256", "subfold_contract_sha256"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")

    def manifest(self) -> dict[str, object]:
        return {
            "role": self.role,
            "subfold": f"fit/{self.subfold}",
            "rows": self.rows,
            "documents": self.documents,
            "gate_square_sum": self.gate_square_sum,
            "row_binding_sha256": self.row_binding_sha256,
            "subfold_contract_sha256": self.subfold_contract_sha256,
            "selection_used": False,
            "holdout_used": False,
            "fit_allocation_used": False,
        }


@dataclass(frozen=True)
class CoupledScaleContext:
    layer: int
    expert: int
    partition: FitCalibrationPartition
    bindings: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.layer < 0 or not 0 <= self.expert < 256:
            raise ValueError("invalid layer/expert scope")
        if set(self.bindings) != REQUIRED_BINDINGS:
            raise ValueError("coupled scale search immutable binding domain differs")
        if any(not name or not _is_sha256(value) for name, value in self.bindings.items()):
            raise ValueError("every coupled scale binding must be a named SHA-256")

    def manifest(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "expert": self.expert,
            "partition": self.partition.manifest(),
            "bindings": dict(sorted(self.bindings.items())),
        }


@dataclass(frozen=True)
class ProjectionCandidate:
    projection: str
    bits: int
    scale: float
    candidate_id: str
    native: Any
    receipt: Mapping[str, object]
    override_policy: str
    override_evidence_id: str

    def __post_init__(self) -> None:
        if self.projection not in ("gate_proj", "up_proj") or self.bits not in (3, 4):
            raise ValueError("coupled projection candidate scope differs")
        _finite_positive(self.scale, "candidate scale")
        if self.scale >= HARD_CEILING - CEILING_TOLERANCE:
            raise ValueError("coupled projection candidate reaches the hard ceiling")
        if not _is_sha256(self.candidate_id):
            raise ValueError("candidate ID must be SHA-256")
        if self.override_policy != OVERRIDE_POLICY or not _is_sha256(
            self.override_evidence_id
        ):
            raise ValueError("candidate did not use the strict coupled scale override")
        required = {
            "payload_sha256",
            "trellis_sha256",
            "suh_sha256",
            "svh_sha256",
            "decoded_label_sha256",
            "source_relative_rmse",
            "full_decode_deferred",
            "hessian_proxy_error",
        }
        if set(self.receipt) != required:
            raise ValueError("projection candidate receipt fields differ")
        for key in (
            "payload_sha256",
            "trellis_sha256",
            "suh_sha256",
            "svh_sha256",
            "decoded_label_sha256",
        ):
            if not _is_sha256(self.receipt[key]):
                raise ValueError(f"candidate {key} must be SHA-256")
        deferred = self.receipt["full_decode_deferred"]
        if not isinstance(deferred, bool):
            raise ValueError("candidate closure deferral flag must be boolean")
        source_rmse = self.receipt["source_relative_rmse"]
        if deferred:
            if source_rmse is not None:
                raise ValueError("deferred candidate must not claim a full-decode RMSE")
        else:
            value = float(source_rmse)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "candidate source_relative_rmse must be finite and nonnegative"
                )
        proxy = float(self.receipt["hessian_proxy_error"])
        if not math.isfinite(proxy) or proxy < 0.0:
            raise ValueError(
                "candidate hessian_proxy_error must be finite and nonnegative"
            )

    def compact_manifest(self) -> dict[str, object]:
        return {
            "projection": self.projection,
            "bits": self.bits,
            "scale": self.scale,
            "scale_hex": self.scale.hex(),
            "candidate_id": self.candidate_id,
            "override_policy": self.override_policy,
            "override_evidence_id": self.override_evidence_id,
            **dict(self.receipt),
        }


@dataclass(frozen=True)
class CoupledObjective:
    weighted_sse: float
    teacher_energy: float
    rows: int
    documents: int
    gate_square_sum: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.weighted_sse) or self.weighted_sse < 0.0:
            raise ValueError("coupled weighted SSE must be finite and nonnegative")
        _finite_positive(self.teacher_energy, "teacher energy")
        _finite_positive(self.gate_square_sum, "objective gate-square sum")
        if self.rows <= 0 or self.documents <= 0:
            raise ValueError("coupled objective must cover rows and documents")

    def manifest(self) -> dict[str, object]:
        return {
            "weighted_sse": self.weighted_sse,
            "teacher_energy": self.teacher_energy,
            "nmse": self.weighted_sse / self.teacher_energy,
            "rows": self.rows,
            "documents": self.documents,
            "gate_square_sum": self.gate_square_sum,
        }


class EncodeCandidate(Protocol):
    def __call__(
        self,
        projection: str,
        bits: int,
        scale: float,
        override_evidence_id: str,
    ) -> ProjectionCandidate: ...


class EncodeCandidateBatch(Protocol):
    def __call__(
        self,
        requests: Sequence[tuple[str, int, float]],
        override_evidence_id: str,
    ) -> Sequence[ProjectionCandidate]: ...


class EvaluatePair(Protocol):
    def __call__(
        self,
        gate: ProjectionCandidate,
        up: ProjectionCandidate,
        partition: FitCalibrationPartition,
    ) -> CoupledObjective: ...


def with_activation_parametric_scale_override(
    config: Any,
    *,
    scale: float,
    search_contract_id: str,
) -> Any:
    """Apply the strict codec override fields without weakening its validation."""

    if not _is_sha256(search_contract_id):
        raise ValueError("coupled scale search contract ID must be SHA-256")
    return replace(
        config,
        global_scale_override=_finite_positive(scale, "coupled scale override"),
        global_scale_override_policy=OVERRIDE_POLICY,
        global_scale_override_evidence_id=search_contract_id,
    )


@dataclass(frozen=True)
class ScaleGrid:
    values: tuple[float, ...]
    center: float
    ratios: tuple[float, ...]
    upper_substituted: bool
    requested_upper: float
    upper_probe: float

    def manifest(self) -> dict[str, object]:
        return {
            "center": self.center,
            "center_hex": self.center.hex(),
            "ratios": list(self.ratios),
            "values": [
                {"value": value, "hex": value.hex()} for value in self.values
            ],
            "upper_substituted": self.upper_substituted,
            "requested_upper": self.requested_upper,
            "requested_upper_hex": self.requested_upper.hex(),
            "upper_probe": self.upper_probe,
            "upper_probe_hex": self.upper_probe.hex(),
            "substitution": (
                "geometric_mean(center, hard_ceiling_minus_2xtolerance)"
                if self.upper_substituted
                else None
            ),
        }


def _grid(center: float, ratios: Sequence[float]) -> ScaleGrid:
    center = _finite_positive(center, "scale-grid center")
    interior_limit = HARD_CEILING - 2.0 * CEILING_TOLERANCE
    if center >= interior_limit:
        raise RuntimeError("coupled scale center is not below the interior ceiling")
    values: dict[str, float] = {}
    upper_ratio = max(float(ratio) for ratio in ratios)
    requested_upper = center * upper_ratio
    upper_substituted = requested_upper >= HARD_CEILING - CEILING_TOLERANCE
    for ratio in ratios:
        value = max(LOWER_BOUND, center * float(ratio))
        if ratio == upper_ratio and upper_substituted:
            value = math.sqrt(center * interior_limit)
        if value >= HARD_CEILING - CEILING_TOLERANCE:
            raise RuntimeError("coupled scale grid produced a non-interior probe")
        values.setdefault(value.hex(), value)
    if len(values) != 3:
        raise RuntimeError("coupled multiplicative grid did not retain three scales")
    ordered = tuple(
        values[key] for key in sorted(values, key=lambda key: float.fromhex(key))
    )
    return ScaleGrid(
        values=ordered,
        center=center,
        ratios=tuple(float(ratio) for ratio in ratios),
        upper_substituted=upper_substituted,
        requested_upper=requested_upper,
        upper_probe=max(ordered),
    )


def _fail_if_unbracketed_upper(
    grid: ScaleGrid,
    winner: ProjectionCandidate,
    *,
    projection: str,
    stage: str,
) -> None:
    if grid.upper_substituted and winner.scale.hex() == grid.upper_probe.hex():
        raise RuntimeError(
            "coupled scale optimum remains unbracketed at the interior upper probe: "
            f"projection={projection}, stage={stage}, scale={winner.scale:.8g}"
        )


def _pair_tie_key(
    objective: CoupledObjective,
    gate: ProjectionCandidate,
    up: ProjectionCandidate,
    *,
    gate_raw: float,
    up_raw: float,
) -> tuple[object, ...]:
    gate_distance = abs(math.log(gate.scale / gate_raw))
    up_distance = abs(math.log(up.scale / up_raw))
    return (
        objective.weighted_sse,
        gate_distance + up_distance,
        gate_distance,
        up_distance,
        gate.scale.hex(),
        up.scale.hex(),
        gate.receipt["payload_sha256"],
        up.receipt["payload_sha256"],
    )


def select_coupled_gate_up_scales(
    *,
    context: CoupledScaleContext,
    raw_scales: Mapping[tuple[str, int], float],
    encode_candidate: EncodeCandidate,
    encode_candidate_batch: EncodeCandidateBatch | None = None,
    evaluate_pair: EvaluatePair,
    rate_pairs: Sequence[tuple[int, int]] = RATE_PAIRS,
) -> tuple[dict[tuple[int, int], tuple[ProjectionCandidate, ProjectionCandidate]], dict[str, object]]:
    """Select partner-conditional gate/up scales for all requested rate pairs."""

    expected_raw = {(projection, bits) for projection in ("gate_proj", "up_proj") for bits in (3, 4)}
    if set(raw_scales) != expected_raw:
        raise ValueError("raw scale centers must cover gate/up K3/K4 exactly")
    normalized_raw = {
        key: _finite_positive(value, f"raw scale {key}") for key, value in raw_scales.items()
    }
    normalized_pairs = tuple((int(gate), int(up)) for gate, up in rate_pairs)
    if len(set(normalized_pairs)) != len(normalized_pairs) or any(
        gate not in (3, 4) or up not in (3, 4) for gate, up in normalized_pairs
    ):
        raise ValueError("rate pairs must be unique K3/K4 pairs")
    if not normalized_pairs:
        raise ValueError("at least one coupled rate pair is required")

    contract_body = {
        "schema": f"{SCHEMA}-contract",
        **context.manifest(),
        "rate_pairs": [list(pair) for pair in normalized_pairs],
        "raw_scales": {
            f"{projection}/k{bits}": {"value": value, "hex": value.hex()}
            for (projection, bits), value in sorted(normalized_raw.items())
        },
        "grid": {
            "coarse_ratios": list(COARSE_RATIOS),
            "fine_ratios": list(FINE_RATIOS),
            "lower_bound": LOWER_BOUND,
            "hard_ceiling": HARD_CEILING,
            "ceiling_tolerance": CEILING_TOLERANCE,
            "upper_probe_substitution": (
                "if requested upper >= hard_ceiling-tolerance, use geometric "
                "mean(center, hard_ceiling-2*tolerance); fail only if it wins"
            ),
            "two_stage": True,
        },
        "override_policy": OVERRIDE_POLICY,
        "objective": "raw gate-square-weighted upstream activation SSE",
        "exact_arithmetic": list(EXACT_ARITHMETIC),
        "tie_break": list(TIE_BREAK),
    }
    search_contract_id = canonical_sha256(contract_body)
    candidate_cache: dict[tuple[str, int, str], ProjectionCandidate] = {}
    score_cache: dict[tuple[str, str], CoupledObjective] = {}
    batch_call_sizes: list[int] = []

    def candidate(projection: str, bits: int, scale: float) -> ProjectionCandidate:
        key = (projection, bits, scale.hex())
        if key not in candidate_cache:
            encoded = encode_candidate(
                projection, bits, scale, search_contract_id
            )
            if (
                encoded.projection != projection
                or encoded.bits != bits
                or encoded.scale.hex() != scale.hex()
                or encoded.override_policy != OVERRIDE_POLICY
                or encoded.override_evidence_id != search_contract_id
            ):
                raise RuntimeError(
                    "candidate encoder changed projection/rate/scale/override binding"
                )
            candidate_cache[key] = encoded
        return candidate_cache[key]

    def ensure_candidates(requests: Sequence[tuple[str, int, float]]) -> None:
        missing: list[tuple[str, int, float]] = []
        seen: set[tuple[str, int, str]] = set()
        for projection, bits, scale in requests:
            key = (projection, bits, scale.hex())
            if key not in candidate_cache and key not in seen:
                missing.append((projection, bits, scale))
                seen.add(key)
        if not missing:
            return
        if encode_candidate_batch is None:
            for projection, bits, scale in missing:
                candidate(projection, bits, scale)
            return
        batch_call_sizes.append(len(missing))
        encoded_batch = tuple(
            encode_candidate_batch(missing, search_contract_id)
        )
        if len(encoded_batch) != len(missing):
            raise RuntimeError("candidate batch encoder returned a partial batch")
        for request, encoded in zip(missing, encoded_batch, strict=True):
            projection, bits, scale = request
            if (
                encoded.projection != projection
                or encoded.bits != bits
                or encoded.scale.hex() != scale.hex()
                or encoded.override_policy != OVERRIDE_POLICY
                or encoded.override_evidence_id != search_contract_id
            ):
                raise RuntimeError(
                    "candidate batch encoder changed projection/rate/scale/override binding"
                )
            key = (projection, bits, scale.hex())
            if key in candidate_cache:
                raise RuntimeError("candidate batch encoder repeated a cached request")
            candidate_cache[key] = encoded

    def score(gate: ProjectionCandidate, up: ProjectionCandidate) -> CoupledObjective:
        key = (gate.candidate_id, up.candidate_id)
        if key not in score_cache:
            observed = evaluate_pair(gate, up, context.partition)
            partition = context.partition
            if (
                observed.rows != partition.rows
                or observed.documents != partition.documents
                or not math.isclose(
                    observed.gate_square_sum,
                    partition.gate_square_sum,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise RuntimeError("coupled evaluator used a different row partition")
            score_cache[key] = observed
        return score_cache[key]

    winners: dict[tuple[int, int], tuple[ProjectionCandidate, ProjectionCandidate]] = {}
    pair_manifests = []
    for gate_bits, up_bits in normalized_pairs:
        gate_raw = normalized_raw[("gate_proj", gate_bits)]
        up_raw = normalized_raw[("up_proj", up_bits)]
        stage_records: list[dict[str, object]] = []
        evaluated: dict[tuple[str, str], tuple[ProjectionCandidate, ProjectionCandidate, CoupledObjective]] = {}

        def evaluate_stage(
            stage: str, gate_scales: Sequence[float], up_scales: Sequence[float]
        ) -> tuple[ProjectionCandidate, ProjectionCandidate]:
            ensure_candidates(
                [
                    *(('gate_proj', gate_bits, scale) for scale in gate_scales),
                    *(('up_proj', up_bits, scale) for scale in up_scales),
                ]
            )
            local = []
            for gate_scale in gate_scales:
                gate = candidate("gate_proj", gate_bits, gate_scale)
                for up_scale in up_scales:
                    up = candidate("up_proj", up_bits, up_scale)
                    objective = score(gate, up)
                    evaluated[(gate.candidate_id, up.candidate_id)] = (
                        gate, up, objective
                    )
                    local.append((gate, up, objective))
                    stage_records.append(
                        {
                            "stage": stage,
                            "gate_scale": gate.scale,
                            "gate_scale_hex": gate.scale.hex(),
                            "up_scale": up.scale,
                            "up_scale_hex": up.scale.hex(),
                            "gate_candidate_id": gate.candidate_id,
                            "up_candidate_id": up.candidate_id,
                            **objective.manifest(),
                        }
                    )
            gate, up, _objective = min(
                local,
                key=lambda item: _pair_tie_key(
                    item[2], item[0], item[1], gate_raw=gate_raw, up_raw=up_raw
                ),
            )
            return gate, up

        coarse_gate = _grid(gate_raw, COARSE_RATIOS)
        coarse_up = _grid(up_raw, COARSE_RATIOS)
        stage1_gate, stage1_up = evaluate_stage(
            "coarse", coarse_gate.values, coarse_up.values
        )
        _fail_if_unbracketed_upper(
            coarse_gate, stage1_gate, projection="gate_proj", stage="coarse"
        )
        _fail_if_unbracketed_upper(
            coarse_up, stage1_up, projection="up_proj", stage="coarse"
        )
        fine_gate = _grid(stage1_gate.scale, FINE_RATIOS)
        fine_up = _grid(stage1_up.scale, FINE_RATIOS)
        stage2_gate, stage2_up = evaluate_stage(
            "fine", fine_gate.values, fine_up.values
        )
        _fail_if_unbracketed_upper(
            fine_gate, stage2_gate, projection="gate_proj", stage="fine"
        )
        _fail_if_unbracketed_upper(
            fine_up, stage2_up, projection="up_proj", stage="fine"
        )
        gate, up, objective = min(
            evaluated.values(),
            key=lambda item: _pair_tie_key(
                item[2], item[0], item[1], gate_raw=gate_raw, up_raw=up_raw
            ),
        )
        winners[(gate_bits, up_bits)] = (gate, up)
        pair_manifests.append(
            {
                "gate_bits": gate_bits,
                "up_bits": up_bits,
                "raw_gate_scale": gate_raw,
                "raw_up_scale": up_raw,
                "stage1_gate_center": stage1_gate.scale,
                "stage1_up_center": stage1_up.scale,
                "grids": {
                    "coarse_gate": coarse_gate.manifest(),
                    "coarse_up": coarse_up.manifest(),
                    "fine_gate": fine_gate.manifest(),
                    "fine_up": fine_up.manifest(),
                },
                "scores": stage_records,
                "selected": {
                    "gate_candidate_id": gate.candidate_id,
                    "up_candidate_id": up.candidate_id,
                    "gate_scale": gate.scale,
                    "gate_scale_hex": gate.scale.hex(),
                    "up_scale": up.scale,
                    "up_scale_hex": up.scale.hex(),
                    **objective.manifest(),
                },
            }
        )

    evidence: dict[str, object] = {
        "schema": SCHEMA,
        "complete": True,
        "search_contract": contract_body,
        "search_contract_id": search_contract_id,
        "projection_candidates": [
            candidate_cache[key].compact_manifest() for key in sorted(candidate_cache)
        ],
        "rate_pair_results": pair_manifests,
        "candidate_cache": {
            "unique_projection_encodes": len(candidate_cache),
            "unique_pair_evaluations": len(score_cache),
            "cartesian_stage_records": sum(
                len(pair["scores"]) for pair in pair_manifests
            ),
            "overlap_reused": True,
            "encode_mode": (
                "batch_by_search_stage_v1"
                if encode_candidate_batch is not None
                else "serial_diagnostic_fallback_v1"
            ),
            "batch_call_sizes": batch_call_sizes,
        },
        "selection_used": False,
        "holdout_used": False,
        "fit_allocation_used": False,
    }
    evidence["evidence_id"] = canonical_sha256(evidence)
    return winners, evidence


@dataclass(frozen=True)
class ExactEvaluationBatch:
    hidden: torch.Tensor
    applied_gates: torch.Tensor
    document_epochs: torch.Tensor


class ExactGLMW4A8UpstreamEvaluator:
    """Stream exact accepted-runtime upstream arithmetic over one fit partition."""

    def __init__(
        self,
        *,
        batch_factory: Callable[[], Iterable[ExactEvaluationBatch]],
        gate_hf: torch.Tensor,
        up_hf: torch.Tensor,
        new_to_old: torch.Tensor,
        hadamard: torch.Tensor,
        expected_partition: FitCalibrationPartition,
    ) -> None:
        self.batch_factory = batch_factory
        self.gate_hf = gate_hf
        self.up_hf = up_hf
        self.new_to_old = new_to_old
        self.hadamard = hadamard
        self.expected_partition = expected_partition
        # Candidate-invariant teacher tensors and h-A8 operands are retained
        # exactly once for the fixed calibration partition. Candidate GEMMs and
        # float64 reduction order below remain unchanged.
        self._teacher_source_gpu = {
            "gate_proj": self.gate_hf.T.float(),
            "up_proj": self.up_hf.T.float(),
        }
        self._prepared_batches: tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...
        ] | None = None
        self._prepared_suh: torch.Tensor | None = None

    def _prepare_batches(
        self, suh: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        if self._prepared_batches is not None:
            assert self._prepared_suh is not None
            if not torch.equal(suh, self._prepared_suh):
                raise RuntimeError(
                    "coupled candidates changed the topology-shared caller input profile"
                )
            return self._prepared_batches

        from scripts.score_glm52_w4a8_activation_quality import prepare_gate_up_operand
        from scripts.score_sqg_w4a8_triplet_candidates import (
            _permuted_teacher_activation,
        )

        prepared: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        for batch in self.batch_factory():
            hidden = batch.hidden.float()
            gates = batch.applied_gates.float()
            if hidden.ndim != 2 or gates.ndim != 1 or hidden.shape[0] != gates.numel():
                raise ValueError("exact evaluator batch geometry differs")
            if batch.document_epochs.ndim != 1 or batch.document_epochs.numel() != gates.numel():
                raise ValueError("exact evaluator document geometry differs")
            operand, observation, _ = prepare_gate_up_operand(
                hidden, suh, self.hadamard, quantize_a8=True
            )
            if observation is None:
                raise RuntimeError("h MXFP8 observation is absent")
            teacher = _permuted_teacher_activation(
                hidden, self._teacher_source_gpu, self.new_to_old
            )
            prepared.append(
                (operand, teacher, gates, observation.preclamp_overflow.any())
            )
        self._prepared_suh = suh.detach().clone()
        self._prepared_batches = tuple(prepared)
        return self._prepared_batches

    def __call__(
        self,
        gate: ProjectionCandidate,
        up: ProjectionCandidate,
        partition: FitCalibrationPartition,
    ) -> CoupledObjective:
        if partition != self.expected_partition:
            raise ValueError("exact evaluator partition differs")
        gate_native = gate.native
        up_native = up.native
        if gate.bits != int(gate_native.bits) or up.bits != int(up_native.bits):
            raise ValueError("native projection rate differs")
        if not torch.equal(gate_native.suh, up_native.suh):
            raise RuntimeError("gate/up candidates changed the shared caller input profile")
        for native in (gate_native, up_native):
            labels = native.weight.float()
            if not torch.equal(labels, labels.to(torch.float8_e4m3fn).float()):
                raise RuntimeError("coupled candidate contains non-E4M3 labels")

        from scripts.score_glm52_w4a8_activation_quality import (
            apply_gate_up_output_transform_silu,
            native_label_gemm,
        )

        score_device = gate_native.weight.device
        weighted_sse = torch.zeros((), dtype=torch.float64, device=score_device)
        teacher_energy = torch.zeros((), dtype=torch.float64, device=score_device)
        gate_square_sum = torch.zeros((), dtype=torch.float64, device=score_device)
        overflow = torch.zeros((), dtype=torch.bool, device=score_device)
        rows = 0
        for operand, teacher, gates, batch_overflow in self._prepare_batches(
            gate_native.suh
        ):
            overflow.logical_or_(batch_overflow)
            _gate_output, _up_output, activation = apply_gate_up_output_transform_silu(
                native_label_gemm(operand, gate_native.weight),
                native_label_gemm(operand, up_native.weight),
                gate_native.svh,
                up_native.svh,
                self.hadamard,
            )
            importance = gates.square()
            difference = activation.float() - teacher.float()
            weighted_sse.add_(
                torch.sum(difference.double().square().sum(dim=1) * importance.double())
            )
            teacher_energy.add_(
                torch.sum(teacher.double().square().sum(dim=1) * importance.double())
            )
            gate_square_sum.add_(importance.double().sum())
            rows += int(operand.shape[0])
        frozen = torch.stack(
            (
                weighted_sse,
                teacher_energy,
                gate_square_sum,
                overflow.to(torch.float64),
            )
        ).detach().cpu().tolist()
        frozen_sse, frozen_teacher, frozen_gate_mass, frozen_overflow = frozen
        if frozen_overflow != 0.0:
            raise RuntimeError("h MXFP8 overflowed during coupled scale selection")
        return CoupledObjective(
            weighted_sse=frozen_sse,
            teacher_energy=frozen_teacher,
            rows=rows,
            documents=self.expected_partition.documents,
            gate_square_sum=frozen_gate_mass,
        )
