"""Strict fresh uniform-K3/K4 SQG bridge for GLM 5.2 expert matrices."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Any, Literal, Protocol

import torch

from .manifest import (
    APPROVED_KQUANT_BACKEND_SHA256,
    APPROVED_KQUANT_STATUS_PORCELAIN,
    APPROVED_KQUANT_STATUS_SHA256,
    APPROVED_KQUANT_TRACKED_DIFF_SHA256,
    BF16TensorBinding,
    FORBIDDEN_MCG_READS,
    FROZEN_KQUANT_REVISION,
    SQG_CODEBOOK,
    SQG_MARKER,
    SQG_LUT_SHA256,
    TAILBITE_CONTEXT,
    TENSOR_SCHEMA,
    SourceBinding,
    SyntheticTensorBinding,
    validate_tensor_manifest,
)
from .permutation import FreshExpertPermutation
from .reference import (
    HADAMARD_BLOCK,
    decode_stored_fp16,
    normalized_hadamard,
    pack_trellis_states,
    payload_sha256,
    relative_rmse,
    tensor_sha256,
    unpack_trellis_states,
)


CODEBOOK_SCALE = 1.24371088
LUT_SHA256 = SQG_LUT_SHA256
MatrixRole = Literal["gate", "up", "down", "synthetic_exl"]
ResidualSide = Literal["input", "output"]
GLOBAL_SCALE_SEARCH_CEILING = 2.05
GLOBAL_SCALE_SEARCH_CEILING_TOLERANCE = 1e-6
GATE_UP_MAX_SOURCE_RELATIVE_RMSE = 0.5

# Production is intentionally pinned to the reviewed KQuant tree plus the sole
# caller-owned scale/sign interface patch.  The constants live in the manifest
# schema so both construction and independent artifact validation share them.
PRODUCTION_H13_CONSTRUCTION = "fit_gate_square_layer_global_dense_covariance_v1"
PRODUCTION_H2_CONSTRUCTION = (
    "fit_applied_gate_square_decoded_gate_up_candidate_conditional_"
    "weighted_oas_scaled_identity_cap_0p75_v1"
)


class EncoderBackend(Protocol):
    def finalize_capture_H(
        self,
        H_data: dict[str, Any],
        quant_args: dict[str, Any],
        verbose: bool,
    ) -> tuple[bool, torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor | None]: ...

    def quantize_qsrt(
        self,
        weight: torch.Tensor,
        H_data: dict[str, Any],
        quant_args: dict[str, Any],
        return_weight_q: bool,
        progress_str: str | None = None,
        verbose: bool = False,
        swap_to_device: torch.device | None = None,
        save_reg: str | None = None,
    ) -> tuple[torch.Tensor, float, dict[str, torch.Tensor]]: ...


def validate_gate_up_encode_smoke(
    *,
    global_scale: float,
    source_relative_rmse: float,
) -> None:
    """Reject a gate/up encode whose cheap matrix diagnostics are catastrophic."""

    for name, value in (
        ("global_scale", global_scale),
        ("source_relative_rmse", source_relative_rmse),
    ):
        if not math.isfinite(value):
            raise RuntimeError(f"gate/up SQG smoke has non-finite {name}")
    if global_scale >= (
        GLOBAL_SCALE_SEARCH_CEILING - GLOBAL_SCALE_SEARCH_CEILING_TOLERANCE
    ):
        raise RuntimeError(
            "gate/up SQG global-scale search saturated at its upper boundary: "
            f"global_scale={global_scale:.8g}"
        )
    if source_relative_rmse > GATE_UP_MAX_SOURCE_RELATIVE_RMSE:
        raise RuntimeError(
            "gate/up SQG source reconstruction is catastrophically inaccurate: "
            f"relative_rmse={source_relative_rmse:.8g}, "
            f"limit={GATE_UP_MAX_SOURCE_RELATIVE_RMSE:.8g}"
        )


@dataclass(frozen=True)
class KQuantRuntime:
    """Loaded KQuant backend plus its exact SQG LUT provider."""

    backend: EncoderBackend
    lut_bytes: Callable[[int], torch.Tensor]
    pack_states: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor]
    revision: str
    backend_sha256: str
    working_tree_dirty: bool
    tracked_diff_sha256: str
    status_sha256: str
    status_porcelain: str
    requires_cuda: bool = True


@dataclass(frozen=True)
class DenseHessian:
    """Dense covariance/sum and evidence required by BlockLDLQ."""

    matrix: torch.Tensor
    evidence_id: str
    construction: str
    split_id: str
    normalization_count: int = 1
    routed_sample_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, torch.Tensor) or self.matrix.ndim != 2:
            raise TypeError("dense Hessian matrix must be a rank-two torch.Tensor")
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("dense Hessian matrix must be square")
        for name in ("evidence_id", "construction", "split_id"):
            if not getattr(self, name):
                raise ValueError(f"dense Hessian {name} must not be empty")
        if (
            isinstance(self.normalization_count, bool)
            or not isinstance(self.normalization_count, int)
            or self.normalization_count <= 0
        ):
            raise ValueError("normalization_count must be a positive integer")
        if self.routed_sample_count is not None and self.routed_sample_count <= 0:
            raise ValueError("routed_sample_count must be positive when supplied")


@dataclass
class DenseHSession:
    """Caller-owned reusable KQuant Hessian/BlockLDL state.

    A session is bound to one exact dense-H payload, normalization, device,
    damping value, and input-transform identity.  It is deliberately neither
    global nor serializable.  The first encode finalizes KQuant's BlockLDL
    state; later compatible encodes reuse it under the session's private lock.
    """

    session_id: str
    in_features: int
    hessian_sha256: str
    evidence_id: str
    normalization_count: int
    device: str
    sigma_reg: float
    input_transform_binding: str
    h_data: dict[str, Any] = field(repr=False)
    _source_matrix: torch.Tensor = field(repr=False)
    _source_matrix_version: int = field(repr=False)
    factorization_sha256: str | None = None
    use_count: int = 0
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    def manifest(self, *, use_ordinal: int, reused_finalized_state: bool) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "hessian_sha256": self.hessian_sha256,
            "device": self.device,
            "sigma_reg": self.sigma_reg,
            "input_transform_binding": self.input_transform_binding,
            "factorization_sha256": self.factorization_sha256,
            "use_ordinal": use_ordinal,
            "reused_finalized_block_ldl": reused_finalized_state,
            "caller_owned": True,
            "module_global": False,
        }


@dataclass(frozen=True)
class SharedResidualProfile:
    """Caller-owned scale/sign profile for a topology-neutral shared side.

    The profile contains positive channel magnitudes and explicit signs.  For
    ``side='input'`` KQuant stores ``sign * magnitude / -codebook_scale`` in
    ``suh`` and moves the per-tensor searched G-scale to ``svh``.  For
    ``side='output'`` it stores ``sign * magnitude`` in ``svh`` and moves the
    G-scale to ``suh``.  No module-global first-tensor cache is involved.
    """

    side: ResidualSide
    signs: torch.Tensor
    channel_scales: torch.Tensor
    profile_id: str
    derivation: str
    stored_fp16_override: torch.Tensor | None = field(default=None, repr=False)
    stored_fp16_realization: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.side not in ("input", "output"):
            raise ValueError("shared residual profile side must be input or output")
        if not self.profile_id or not self.derivation:
            raise ValueError("shared residual profile identity must not be empty")
        if self.signs.ndim != 1 or self.channel_scales.ndim != 1:
            raise ValueError("shared profile signs and scales must be one-dimensional")
        if self.signs.shape != self.channel_scales.shape:
            raise ValueError("shared profile signs and scales must have the same shape")
        signs = self.signs.detach().float()
        scales = self.channel_scales.detach().float()
        if not bool(torch.isfinite(signs).all()) or not bool((signs.abs() == 1).all()):
            raise ValueError("shared profile signs must contain only finite -1/+1")
        if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
            raise ValueError("shared profile scales must be finite and positive")
        override = self.stored_fp16_override
        realization = self.stored_fp16_realization
        if (override is None) != (realization is None):
            raise ValueError(
                "stored FP16 override and realization evidence must be supplied together"
            )
        if override is not None:
            if (
                override.device.type != "cpu"
                or override.dtype != torch.float16
                or override.ndim != 1
                or override.shape != self.signs.shape
                or not override.is_contiguous()
                or not bool(torch.isfinite(override).all())
                or bool((override == 0).any())
            ):
                raise ValueError(
                    "stored FP16 override must be contiguous, finite, nonzero CPU FP16"
                )
            if not isinstance(realization, Mapping):  # pragma: no cover
                raise TypeError("stored FP16 realization evidence must be a mapping")
            expected_record = {
                "schema": "kquant-assigned-cuda-fp32-to-fp16-v1",
                "side": self.side,
                "conceptual_cpu_sha256": tensor_sha256(
                    self._conceptual_stored_fp16()
                ),
                "realized_sha256": tensor_sha256(override),
                "mismatch_count": int(
                    (override != self._conceptual_stored_fp16()).sum().item()
                ),
                "backend_expression_unchanged": True,
                "backend_output_overwritten": False,
                "caller_owned": True,
            }
            if dict(realization) != expected_record or expected_record[
                "mismatch_count"
            ] <= 0:
                raise ValueError("stored FP16 realization evidence differs")

    @classmethod
    def from_stored_vector(
        cls,
        vector: torch.Tensor,
        *,
        side: ResidualSide,
        profile_id: str,
        derivation: str,
    ) -> "SharedResidualProfile":
        """Construct a profile whose persisted FP16 side must match ``vector``."""

        if vector.ndim != 1 or not vector.is_floating_point():
            raise TypeError("stored residual vector must be one-dimensional floating point")
        value = vector.detach().float()
        if not bool(torch.isfinite(value).all()) or not bool((value != 0).all()):
            raise ValueError("stored residual vector must be finite and nonzero")
        if side == "input":
            signs = -value.sign()
            scales = value.abs() * CODEBOOK_SCALE
        elif side == "output":
            signs = value.sign()
            scales = value.abs()
        else:
            raise ValueError("shared residual profile side must be input or output")
        return cls(
            side=side,
            signs=signs.contiguous(),
            channel_scales=scales.contiguous(),
            profile_id=profile_id,
            derivation=derivation,
        )

    def _conceptual_stored_fp16(self) -> torch.Tensor:
        signs = self.signs.detach().float()
        scales = self.channel_scales.detach().float()
        if self.side == "input":
            value = signs * scales / (-CODEBOOK_SCALE) + 1e-10
        else:
            value = signs * scales + 1e-10
        return value.half().contiguous()

    def expected_stored_fp16(self) -> torch.Tensor:
        if self.stored_fp16_override is not None:
            return self.stored_fp16_override.detach().clone().contiguous()
        return self._conceptual_stored_fp16()

    def manifest(self) -> dict[str, object]:
        value = {
            "profile_id": self.profile_id,
            "derivation": self.derivation,
            "side": self.side,
            "signs_sha256": tensor_sha256(self.signs.float()),
            "channel_scales_sha256": tensor_sha256(self.channel_scales.float()),
            "expected_stored_fp16_sha256": tensor_sha256(self.expected_stored_fp16()),
            "caller_owned": True,
            "module_global_first_tensor_state": False,
        }
        if self.stored_fp16_realization is not None:
            value["stored_fp16_realization"] = dict(
                self.stored_fp16_realization
            )
        return value


def realize_shared_residual_profile(
    profile: SharedResidualProfile,
    device: str | torch.device,
) -> SharedResidualProfile:
    """Bind the persisted shared vector to KQuant's assigned-device arithmetic.

    KQuant evaluates the explicit sign/scale expression in float32 on the
    assigned CUDA device and only then stores FP16.  A CPU recomputation can
    land on the adjacent FP16 value at an exact rounding boundary.  Realize
    that expression once per profile, retain KQuant's bytes as the oracle, and
    carry an override only when the CPU and CUDA realizations differ.
    """

    target = torch.device(device)
    if target.type != "cuda":
        if profile.stored_fp16_override is not None:
            raise ValueError("a CUDA-realized profile cannot be rebound on CPU")
        return profile
    signs = profile.signs.detach().to(
        device=target, dtype=torch.float32, copy=True
    )
    scales = profile.channel_scales.detach().to(
        device=target, dtype=torch.float32, copy=True
    )
    if profile.side == "input":
        value = signs * scales / (-CODEBOOK_SCALE) + 1e-10
    else:
        value = signs * scales + 1e-10
    realized = value.half().cpu().contiguous()
    conceptual = profile._conceptual_stored_fp16()
    if profile.stored_fp16_override is not None:
        if not torch.equal(realized, profile.stored_fp16_override):
            raise RuntimeError(
                "assigned CUDA realization differs from the sealed profile override"
            )
        return profile
    mismatch_count = int((realized != conceptual).sum().item())
    if mismatch_count == 0:
        return profile
    record: dict[str, object] = {
        "schema": "kquant-assigned-cuda-fp32-to-fp16-v1",
        "side": profile.side,
        "conceptual_cpu_sha256": tensor_sha256(conceptual),
        "realized_sha256": tensor_sha256(realized),
        "mismatch_count": mismatch_count,
        "backend_expression_unchanged": True,
        "backend_output_overwritten": False,
        "caller_owned": True,
    }
    return replace(
        profile,
        stored_fp16_override=realized,
        stored_fp16_realization=record,
    )


@dataclass(frozen=True)
class UniformSQGConfig:
    """Immutable settings for one fresh uniform-rate treatment tensor."""

    tensor_id: str
    bits: int
    matrix_role: MatrixRole
    transform_seed: int
    output_sign_seed: int
    source_binding: SourceBinding
    physical_permutation: FreshExpertPermutation | None = None
    shared_residual_profile: SharedResidualProfile | None = None
    sigma_reg: float = 0.025
    apply_out_scales: bool | None = None
    global_scale_into: ResidualSide = "input"
    device: str = "cuda:0"
    production: bool = True
    closure_max_relative_rmse: float = 0.01
    kquant_root: str | None = None
    exllamav3_root: str | None = None

    def __post_init__(self) -> None:
        if not self.tensor_id:
            raise ValueError("tensor_id must not be empty")
        if isinstance(self.bits, bool) or not isinstance(self.bits, int) or self.bits not in (3, 4):
            raise ValueError("fresh uniform SQG permits only integer K3 or K4")
        if self.matrix_role not in ("gate", "up", "down", "synthetic_exl"):
            raise ValueError("unknown GLM matrix role")
        for name in ("transform_seed", "output_sign_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not math.isfinite(self.sigma_reg) or self.sigma_reg <= 0:
            raise ValueError("sigma_reg must be positive and finite")
        if self.global_scale_into not in ("input", "output"):
            raise ValueError("global_scale_into must be input or output")
        if not math.isfinite(self.closure_max_relative_rmse) or self.closure_max_relative_rmse <= 0:
            raise ValueError("closure threshold must be positive and finite")
        if self.production:
            if not isinstance(self.source_binding, BF16TensorBinding):
                raise TypeError("production encoding requires an immutable official BF16 binding")
            if self.matrix_role == "synthetic_exl":
                raise ValueError("synthetic_exl is forbidden for production encoding")
            if self.physical_permutation is None:
                raise ValueError("production encoding requires a fresh physical permutation")
            if not self.physical_permutation.production_qualified:
                raise ValueError(
                    "production encoding requires a calibration-derived "
                    "GLM h2_reverse permutation"
                )
            source_match = re.fullmatch(
                r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
                r"(gate_proj|up_proj|down_proj)\.weight",
                self.source_binding.tensor_name,
            )
            if source_match is None:
                raise ValueError("production source is not a routed GLM expert tensor")
            projection_by_role = {
                "gate": "gate_proj",
                "up": "up_proj",
                "down": "down_proj",
            }
            if source_match.group(3) != projection_by_role[self.matrix_role]:
                raise ValueError("matrix role disagrees with official BF16 tensor name")
            expected_tensor_id = self.source_binding.tensor_name.removesuffix(".weight")
            if self.tensor_id != expected_tensor_id:
                raise ValueError(
                    "production tensor ID disagrees with official BF16 tensor binding"
                )
            expected_scope = (
                f"layer-{int(source_match.group(1)):03d}/"
                f"expert-{int(source_match.group(2)):03d}"
            )
            if self.physical_permutation.scope != expected_scope:
                raise ValueError(
                    "calibration-derived permutation scope does not match "
                    "the official BF16 tensor binding"
                )
        elif not isinstance(self.source_binding, (BF16TensorBinding, SyntheticTensorBinding)):
            raise TypeError("unsupported source binding")


@dataclass(frozen=True)
class EncodedSQGMatrix:
    """SQG-only tensors, authoritative stored decode, and sealed lineage."""

    bits: int
    trellis: torch.Tensor
    suh: torch.Tensor
    svh: torch.Tensor
    sqg: torch.Tensor
    reconstructed_exl: torch.Tensor
    proxy_error: float
    manifest: Mapping[str, object]

    def named_tensors(self, base: str) -> dict[str, torch.Tensor]:
        if not base:
            raise ValueError("tensor base name must not be empty")
        result = {
            f"{base}.trellis": self.trellis,
            f"{base}.suh": self.suh,
            f"{base}.svh": self.svh,
            f"{base}.sqg": self.sqg,
        }
        if any(name.endswith((".mcg", ".mul1")) for name in result):
            raise RuntimeError("fresh SQG payload unexpectedly contains a legacy marker")
        return result


def _input_transform_binding(config: UniformSQGConfig) -> str:
    profile = config.shared_residual_profile
    if profile is not None and profile.side == "input":
        identity: dict[str, object] = {
            "kind": "explicit_input_profile",
            "profile": profile.manifest(),
        }
    else:
        identity = {
            "kind": "fresh_seeded_input_signs",
            "transform_seed": config.transform_seed,
        }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def prepare_dense_h_session(
    dense_h: DenseHessian,
    config: UniformSQGConfig,
) -> DenseHSession:
    """Allocate caller-owned reusable dense-H state without factorizing it."""

    in_features = int(dense_h.matrix.shape[0])
    _validate_dense_hessian(dense_h, in_features)
    hessian_hash = tensor_sha256(dense_h.matrix)
    device = torch.device(config.device)
    canonical_h = dense_h.matrix.detach().to(
        device=device,
        dtype=torch.float32,
        copy=True,
    ).contiguous()
    input_binding = _input_transform_binding(config)
    session_material = {
        "hessian_sha256": hessian_hash,
        "evidence_id": dense_h.evidence_id,
        "normalization_count": dense_h.normalization_count,
        "device": str(device),
        "sigma_reg": config.sigma_reg,
        "input_transform_binding": input_binding,
    }
    session_id = hashlib.sha256(
        json.dumps(
            session_material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    h_data: dict[str, Any] = {
        "H": canonical_h.clone(),
        "error_H": canonical_h,
        "first_key": f"fresh-session:{session_id}",
        "count": dense_h.normalization_count,
        "finalized": False,
        "num_total": dense_h.normalization_count,
        "inf_nan": torch.zeros(2, dtype=torch.long, device=device),
        "device": device,
    }
    return DenseHSession(
        session_id=session_id,
        in_features=in_features,
        hessian_sha256=hessian_hash,
        evidence_id=dense_h.evidence_id,
        normalization_count=dense_h.normalization_count,
        device=str(device),
        sigma_reg=config.sigma_reg,
        input_transform_binding=input_binding,
        h_data=h_data,
        _source_matrix=dense_h.matrix,
        _source_matrix_version=dense_h.matrix._version,
    )


def _validate_dense_h_session(
    session: DenseHSession,
    dense_h: DenseHessian,
    config: UniformSQGConfig,
    in_features: int,
) -> None:
    if dense_h.matrix is not session._source_matrix:
        raise ValueError(
            "dense-H session must be reused with the same caller-owned matrix object"
        )
    if dense_h.matrix._version != session._source_matrix_version:
        raise ValueError("caller-owned dense-H matrix changed after session creation")
    expected = {
        "in_features": in_features,
        "hessian_sha256": session.hessian_sha256,
        "evidence_id": dense_h.evidence_id,
        "normalization_count": dense_h.normalization_count,
        "device": str(torch.device(config.device)),
        "sigma_reg": config.sigma_reg,
        "input_transform_binding": _input_transform_binding(config),
    }
    observed = {name: getattr(session, name) for name in expected}
    if observed != expected:
        drift = {
            name: {"session": observed[name], "requested": value}
            for name, value in expected.items()
            if observed[name] != value
        }
        raise ValueError(f"dense-H session binding mismatch: {drift}")
    if session.h_data.get("q_fallback") is True:
        raise RuntimeError("dense-H session was contaminated by a fallback result")


def resume_dense_h_session(
    session: DenseHSession,
    dense_h: DenseHessian,
    config: UniformSQGConfig,
    runtime: KQuantRuntime,
    *,
    prior_tensor_manifests: Sequence[Mapping[str, object]],
    expected_tensor_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Restore a reusable H13 session from a contiguous artifact prefix.

    BlockLDL state itself is intentionally not serialized.  A resumed worker
    re-factorizes the exact bound Hessian by invoking KQuant's frozen
    ``finalize_capture_H`` directly, without quantizing or reading a weight.
    The caller supplies the already sealed gate/up tensor manifests in their
    deterministic encode order.  Their session records must prove an exact
    contiguous ordinal prefix before ``use_count`` is restored.
    """

    if config.production:
        _validate_production_runtime(runtime)
        _validate_production_dense_hessian(dense_h, config)
    in_features = int(dense_h.matrix.shape[0])
    _validate_dense_h_session(session, dense_h, config, in_features)
    profile = config.shared_residual_profile
    if profile is None or profile.side != "input":
        raise ValueError(
            "production H13 resume requires the exact explicit shared input profile"
        )
    manifests = [dict(item) for item in prior_tensor_manifests]
    if not manifests:
        raise ValueError("dense-H resume requires a nonempty completed prefix")
    expected_binding = {
        "session_id": session.session_id,
        "hessian_sha256": session.hessian_sha256,
        "device": session.device,
        "sigma_reg": session.sigma_reg,
        "input_transform_binding": session.input_transform_binding,
        "caller_owned": True,
        "module_global": False,
    }
    prefix_records: list[dict[str, object]] = []
    observed_tensor_ids: list[str] = []
    prefix_factorization_sha256: str | None = None
    for ordinal, manifest in enumerate(manifests, start=1):
        validate_tensor_manifest(manifest)
        if manifest.get("matrix_role") not in ("gate", "up"):
            raise ValueError("dense-H resume prefix contains a non-gate/up tensor")
        tensor_id = manifest.get("tensor_id")
        if not isinstance(tensor_id, str) or not tensor_id:
            raise ValueError("dense-H resume prefix tensor ID is absent")
        observed_tensor_ids.append(tensor_id)
        if config.production:
            source_record = manifest.get("source")
            if not isinstance(source_record, Mapping) or source_record.get("kind") != "official_bf16":
                raise ValueError("production dense-H resume prefix is not official BF16")
            encoder_record = manifest.get("encoder")
            expected_encoder = {
                "kquant_revision": runtime.revision,
                "backend_sha256": runtime.backend_sha256,
                "working_tree_dirty": runtime.working_tree_dirty,
                "tracked_diff_sha256": runtime.tracked_diff_sha256,
                "status_sha256": runtime.status_sha256,
                "status_porcelain": runtime.status_porcelain,
            }
            if not isinstance(encoder_record, Mapping) or {
                key: encoder_record.get(key) for key in expected_encoder
            } != expected_encoder:
                raise ValueError("production dense-H resume encoder provenance differs")
        dense_record = manifest.get("dense_h")
        if not isinstance(dense_record, Mapping):
            raise ValueError("dense-H resume prefix lacks dense-H evidence")
        record = dense_record.get("session")
        if not isinstance(record, Mapping):
            raise ValueError("dense-H resume prefix lacks session evidence")
        observed_binding = {key: record.get(key) for key in expected_binding}
        if observed_binding != expected_binding:
            raise ValueError("dense-H resume prefix session binding differs")
        if record.get("use_ordinal") != ordinal:
            raise ValueError("dense-H resume prefix ordinals are not contiguous")
        if record.get("reused_finalized_block_ldl") is not (ordinal != 1):
            raise ValueError("dense-H resume prefix finalized-state evidence differs")
        factorization_sha256 = record.get("factorization_sha256")
        if (
            not isinstance(factorization_sha256, str)
            or len(factorization_sha256) != 64
            or any(char not in "0123456789abcdef" for char in factorization_sha256)
        ):
            raise ValueError("dense-H resume prefix lacks a BlockLDL fingerprint")
        if prefix_factorization_sha256 is None:
            prefix_factorization_sha256 = factorization_sha256
        elif factorization_sha256 != prefix_factorization_sha256:
            raise ValueError("dense-H resume prefix BlockLDL fingerprints differ")
        if (
            dense_record.get("matrix_sha256") != session.hessian_sha256
            or dense_record.get("evidence_id") != dense_h.evidence_id
            or dense_record.get("construction") != dense_h.construction
            or dense_record.get("split_id") != dense_h.split_id
            or dense_record.get("normalization_count")
            != dense_h.normalization_count
            or dense_record.get("fallback") is not False
        ):
            raise ValueError("dense-H resume prefix Hessian binding differs")
        prefix_records.append(
            {
                "tensor_id": tensor_id,
                "matrix_role": manifest.get("matrix_role"),
                "session": dict(record),
                "tensor_manifest_sha256": hashlib.sha256(
                    json.dumps(
                        manifest,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest(),
            }
        )
    if len(set(observed_tensor_ids)) != len(observed_tensor_ids):
        raise ValueError("dense-H resume prefix repeats a tensor ID")
    if expected_tensor_ids is None:
        if config.production:
            raise ValueError(
                "production dense-H resume requires an explicit deterministic prefix"
            )
    else:
        expected_ids = [str(value) for value in expected_tensor_ids]
        if not expected_ids or observed_tensor_ids != expected_ids:
            raise ValueError("dense-H resume tensor IDs differ from expected prefix")
    with session._lock:
        if session.use_count != 0 or session.h_data.get("finalized"):
            raise ValueError("dense-H resume requires a fresh unfinalized session")
        quant_args: dict[str, Any] = {
            "sigma_reg": config.sigma_reg,
            "input_signs": profile.signs,
        }
        q_fallback, _h, L, su, _diag = runtime.backend.finalize_capture_H(
            session.h_data,
            quant_args,
            False,
        )
        if (
            q_fallback is not False
            or session.h_data.get("q_fallback") is not False
            or session.h_data.get("finalized") is not True
            or L is None
        ):
            raise RuntimeError("resumed dense-H session failed no-fallback BlockLDL")
        expected_signs = profile.signs.detach().to(
            device=su.device, dtype=torch.float32, copy=True
        ).reshape(-1, 1)
        if not torch.equal(su, expected_signs):
            raise RuntimeError("resumed dense-H input signs differ from shared profile")
        observed_factorization_sha256 = tensor_sha256(L)
        if observed_factorization_sha256 != prefix_factorization_sha256:
            raise RuntimeError(
                "resumed BlockLDL fingerprint differs from the sealed prefix"
            )
        session.factorization_sha256 = observed_factorization_sha256
        session.use_count = len(manifests)
    evidence = {
        "schema": "glm52_fresh_sqg_dense_h_resume_v1",
        "session_id": session.session_id,
        "restored_use_count": session.use_count,
        "prefix_sha256": hashlib.sha256(
            json.dumps(
                prefix_records,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
        "direct_finalize_capture_H": True,
        "weight_encode_used_for_restore": False,
        "q_fallback": False,
        "finalized_block_ldl": True,
        "factorization_sha256": session.factorization_sha256,
    }
    return evidence


def _git_state(root: Path) -> tuple[str, str, str, str]:
    """Return exact, hashable KQuant revision/status/tracked-diff evidence."""

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_bytes = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        tracked_diff_bytes = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot establish KQuant revision under {root}") from exc
    try:
        status_porcelain = status_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:  # pragma: no cover - paths are controlled
        raise RuntimeError("KQuant porcelain status is not valid UTF-8") from exc
    return (
        revision,
        status_porcelain,
        hashlib.sha256(status_bytes).hexdigest(),
        hashlib.sha256(tracked_diff_bytes).hexdigest(),
    )


def _validate_production_provenance(
    *,
    revision: str,
    backend_sha256: str,
    working_tree_dirty: bool,
    tracked_diff_sha256: str,
    status_sha256: str,
    status_porcelain: str,
) -> None:
    """Reject every provenance tuple except the reviewed frozen encoder."""

    expected: dict[str, object] = {
        "revision": FROZEN_KQUANT_REVISION,
        "backend_sha256": APPROVED_KQUANT_BACKEND_SHA256,
        "working_tree_dirty": True,
        "tracked_diff_sha256": APPROVED_KQUANT_TRACKED_DIFF_SHA256,
        "status_sha256": APPROVED_KQUANT_STATUS_SHA256,
        "status_porcelain": APPROVED_KQUANT_STATUS_PORCELAIN,
    }
    observed: dict[str, object] = {
        "revision": revision,
        "backend_sha256": backend_sha256,
        "working_tree_dirty": working_tree_dirty,
        "tracked_diff_sha256": tracked_diff_sha256,
        "status_sha256": status_sha256,
        "status_porcelain": status_porcelain,
    }
    drift = {
        name: {"observed": observed[name], "approved": value}
        for name, value in expected.items()
        if observed[name] != value
    }
    if drift:
        raise RuntimeError(f"unapproved KQuant production runtime provenance: {drift}")


def _validate_production_runtime(runtime: KQuantRuntime) -> None:
    """Apply the frozen provenance gate to a loaded or injected runtime."""

    _validate_production_provenance(
        revision=runtime.revision,
        backend_sha256=runtime.backend_sha256,
        working_tree_dirty=runtime.working_tree_dirty,
        tracked_diff_sha256=runtime.tracked_diff_sha256,
        status_sha256=runtime.status_sha256,
        status_porcelain=runtime.status_porcelain,
    )


def load_kquant_runtime(kquant_root: str | Path, exllamav3_root: str | Path) -> KQuantRuntime:
    """Load the isolated KQuant backend and install its exact SQG encoder."""

    root = Path(kquant_root).resolve()
    if not (root / "kquant" / "sqg_e4m3.py").is_file():
        raise FileNotFoundError(f"invalid KQuant checkout: {root}")
    backend_path = root / "kquant" / "exl3_encoder_backend.py"
    if not backend_path.is_file():
        raise FileNotFoundError(f"missing KQuant encoder backend: {backend_path}")
    revision, status_porcelain, status_sha256, tracked_diff_sha256 = _git_state(root)
    backend_sha256 = hashlib.sha256(backend_path.read_bytes()).hexdigest()
    _validate_production_provenance(
        revision=revision,
        backend_sha256=backend_sha256,
        working_tree_dirty=bool(status_porcelain),
        tracked_diff_sha256=tracked_diff_sha256,
        status_sha256=status_sha256,
        status_porcelain=status_porcelain,
    )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    sqg_module = importlib.import_module("kquant.sqg_e4m3")
    loader_module = importlib.import_module("kquant.exl3_loader")
    quantizer_module = importlib.import_module("kquant.sqg_quantizer")
    loaded_path = Path(sqg_module.__file__).resolve()
    if root not in loaded_path.parents:
        raise ImportError(f"a different kquant package is already loaded from {loaded_path}")
    backend = loader_module.load_qsrt_encoder(exllamav3_root)
    quantizer_module.install_sqg_quantizer(backend)
    return KQuantRuntime(
        backend=backend,
        lut_bytes=lambda bits: sqg_module.sqg_xor_cheb_t12_bytes(bits),
        pack_states=backend.pack_trellis,
        revision=revision,
        backend_sha256=backend_sha256,
        working_tree_dirty=bool(status_porcelain),
        tracked_diff_sha256=tracked_diff_sha256,
        status_sha256=status_sha256,
        status_porcelain=status_porcelain,
        requires_cuda=True,
    )


def _validate_dense_hessian(dense_h: DenseHessian, in_features: int) -> None:
    matrix = dense_h.matrix.detach().float()
    if tuple(matrix.shape) != (in_features, in_features):
        raise ValueError(
            f"dense Hessian shape {tuple(matrix.shape)} does not match EXL K={in_features}"
        )
    if matrix.is_meta:
        raise ValueError("meta/identity fallback Hessians are forbidden")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("dense Hessian must contain only finite values")
    if not torch.allclose(matrix, matrix.T, rtol=1e-4, atol=1e-6):
        raise ValueError("dense Hessian must be symmetric")
    diag_mean = float(matrix.diagonal().mean()) / dense_h.normalization_count
    if not math.isfinite(diag_mean) or diag_mean <= 1e-20:
        raise ValueError("dense Hessian would trigger KQuant's fallback path")


def _validate_production_dense_hessian(
    dense_h: DenseHessian,
    config: UniformSQGConfig,
) -> None:
    """Bind each production projection to its fit-only Hessian recipe."""

    if dense_h.split_id != "fit":
        raise ValueError("production dense Hessians may consume only the fit split")
    expected = (
        PRODUCTION_H2_CONSTRUCTION
        if config.matrix_role == "down"
        else PRODUCTION_H13_CONSTRUCTION
    )
    if dense_h.construction != expected:
        raise ValueError(
            f"production {config.matrix_role} dense-H construction differs: "
            f"{dense_h.construction!r} != {expected!r}"
        )


def _prepare_exl_source(
    source: torch.Tensor,
    config: UniformSQGConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Clone, freshly permute, and orient one official HF matrix for EXL."""

    if source.ndim != 2 or not source.is_floating_point():
        raise TypeError("source must be a rank-two floating-point matrix")
    config.source_binding.validate_tensor(source)
    source_hash_before = tensor_sha256(source)
    role = config.matrix_role
    if role == "synthetic_exl":
        prepared = source.detach().clone().contiguous()
        transform = {"source_orientation": "exl", "operation": "clone_only"}
    else:
        permutation = config.physical_permutation
        if permutation is None:
            raise ValueError("GLM gate/up/down encoding requires a fresh permutation")
        order = permutation.new_to_old.to(device=source.device)
        if role in ("gate", "up"):
            if source.shape[0] != permutation.intermediate_features:
                raise ValueError("gate/up row count does not match fresh permutation")
            prepared = source.detach().index_select(0, order).T.clone().contiguous()
            operation = "fresh_P_rows_then_hf_to_exl_transpose"
        elif role == "down":
            if source.shape[1] != permutation.intermediate_features:
                raise ValueError("down column count does not match fresh permutation")
            prepared = source.detach().index_select(1, order).T.clone().contiguous()
            operation = "fresh_Pt_columns_then_hf_to_exl_transpose"
        else:  # pragma: no cover - guarded by config validation
            raise ValueError("unsupported matrix role")
        transform = {
            "source_orientation": "huggingface",
            "operation": operation,
            "physical_permutation": permutation.manifest(),
        }
    if tensor_sha256(source) != source_hash_before:
        raise RuntimeError("source matrix changed while preparing the private encode clone")
    if prepared.data_ptr() == source.data_ptr():
        raise RuntimeError("prepared source unexpectedly aliases the caller-owned tensor")
    transform["prepared_exl_sha256"] = tensor_sha256(prepared)
    transform["prepared_exl_dtype"] = str(prepared.dtype).removeprefix("torch.")
    transform["prepared_exl_shape"] = list(prepared.shape)
    return prepared, transform


def _transform_digest(transform: Mapping[str, object]) -> str:
    encoded = json.dumps(transform, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def encode_uniform_sqg(
    source: torch.Tensor,
    dense_h: DenseHessian,
    config: UniformSQGConfig,
    *,
    runtime: KQuantRuntime | None = None,
    dense_h_session: DenseHSession | None = None,
) -> EncodedSQGMatrix:
    """Encode one matrix with fresh KQuant regularization and dense-H SQG.

    This function has no API for legacy MCG tensors, scales, signs, seeds,
    permutations, decoded weights, or packed bytes.  The only legacy model
    decision allowed outside this call is the caller's frozen K3/K4 bit_map.
    """

    prepared_source, source_transform = _prepare_exl_source(source, config)
    k, n = prepared_source.shape
    if config.production:
        _validate_production_dense_hessian(dense_h, config)
    if k % HADAMARD_BLOCK or n % HADAMARD_BLOCK:
        raise ValueError("EXL K and N dimensions must both be divisible by 128")
    if not bool(torch.isfinite(prepared_source.float()).all()):
        raise ValueError("source matrix must contain only finite values")
    if runtime is None:
        if config.kquant_root is None or config.exllamav3_root is None:
            raise ValueError("kquant_root and exllamav3_root are required to load KQuant")
        runtime = load_kquant_runtime(config.kquant_root, config.exllamav3_root)
    if config.production:
        _validate_production_runtime(runtime)
    device = torch.device(config.device)
    if runtime.requires_cuda and device.type != "cuda":
        raise ValueError("the production SQG Viterbi encoder requires one CUDA device")

    lut = runtime.lut_bytes(config.bits).detach().to(device="cpu").contiguous()
    if lut.dtype != torch.uint8 or tuple(lut.shape) != (1 << 16,):
        raise RuntimeError("KQuant returned a malformed SQG E4M3 LUT")
    lut_hash = payload_sha256(lut)
    if lut_hash != LUT_SHA256[config.bits]:
        raise RuntimeError(
            f"SQG K{config.bits} LUT identity mismatch: {lut_hash}"
        )

    work = prepared_source.detach().to(
        device=device,
        dtype=torch.float32,
        copy=True,
    ).contiguous()
    if device == prepared_source.device and work.data_ptr() == prepared_source.data_ptr():
        raise RuntimeError("KQuant work matrix aliases the prepared source")
    if dense_h_session is None:
        if config.production and config.matrix_role in ("gate", "up"):
            raise ValueError(
                "production gate/up encoding requires an explicit reusable "
                "DenseHSession"
            )
        dense_h_session = prepare_dense_h_session(dense_h, config)
    _validate_dense_h_session(dense_h_session, dense_h, config, k)
    hessian_hash = dense_h_session.hessian_sha256
    h_data = dense_h_session.h_data

    captured_states: torch.Tensor | None = None

    def strict_pack(states: torch.Tensor, quant_args: dict[str, Any]) -> torch.Tensor:
        nonlocal captured_states
        if captured_states is not None:
            raise RuntimeError("KQuant invoked the uniform pack callback more than once")
        if quant_args.get("K") != config.bits:
            raise RuntimeError("KQuant changed the frozen per-tensor bit assignment")
        captured_states = states.detach().clone().contiguous()
        return runtime.pack_states(states, quant_args)

    quant_args: dict[str, Any] = {
        "K": config.bits,
        "seed": config.transform_seed,
        "sv_seed": config.output_sign_seed,
        "sigma_reg": config.sigma_reg,
        "devices": [device],
        "apply_out_scales": config.apply_out_scales,
        "g_scale_into_sv": config.global_scale_into == "output",
        "sqg_e4m3_lut": lut,
        "tailbite_context": TAILBITE_CONTEXT,
        "pack_trellis_fn": strict_pack,
    }
    profile = config.shared_residual_profile
    if profile is not None:
        expected_length = k if profile.side == "input" else n
        if profile.signs.numel() != expected_length:
            raise ValueError(
                f"shared {profile.side} profile length does not match its residual axis"
            )
        if profile.side == "input":
            quant_args["input_signs"] = profile.signs
            quant_args["input_channel_scale_profile"] = profile.channel_scales
            quant_args["g_scale_into_sv"] = True
        else:
            quant_args["output_signs"] = profile.signs
            quant_args["output_channel_scale_profile"] = profile.channel_scales
            quant_args["apply_out_scales"] = True
            quant_args["g_scale_into_sv"] = False
    forbidden_quant_args = {"mcg", "mul1", "shared_input_scales_key"}
    if forbidden_quant_args.intersection(quant_args):  # pragma: no cover
        raise RuntimeError("fresh SQG quant_args contain a forbidden legacy/shared-state key")

    with dense_h_session._lock:
        reused_finalized_state = bool(h_data.get("finalized"))
        session_use_ordinal = dense_h_session.use_count + 1
        encoder_weight, proxy_error, raw_tensors = runtime.backend.quantize_qsrt(
            work,
            h_data,
            quant_args,
            True,
            progress_str=None,
            verbose=False,
        )
        if quant_args.get("q_fallback") is not False:
            raise RuntimeError("KQuant attempted a forbidden non-dense-H fallback")
        factorization = h_data.get("L")
        if not isinstance(factorization, torch.Tensor):
            raise RuntimeError("KQuant did not retain the no-fallback BlockLDL state")
        if dense_h_session.factorization_sha256 is None:
            # The factor is caller-owned and KQuant reuses it read-only.  Hash
            # once; rehashing a 6144-square H13 for every expert would add a
            # large byte-neutral cost to the timed encode path.
            dense_h_session.factorization_sha256 = tensor_sha256(factorization)
        dense_h_session.use_count = session_use_ordinal
    if captured_states is None:
        raise RuntimeError("KQuant did not emit packed trellis states")
    if set(raw_tensors) != {"trellis", "suh", "svh"}:
        raise RuntimeError(
            f"KQuant emitted non-SQG or unexpected tensors: {sorted(raw_tensors)}"
        )
    trellis = raw_tensors["trellis"].detach().to(device="cpu", copy=True).contiguous()
    suh = raw_tensors["suh"].detach().to(device="cpu", copy=True).contiguous()
    svh = raw_tensors["svh"].detach().to(device="cpu", copy=True).contiguous()
    states_cpu = captured_states.detach().to(device="cpu", copy=True).contiguous()
    if trellis.dtype != torch.int16:
        raise TypeError("packed trellis must use torch.int16")
    if suh.dtype != torch.float16 or svh.dtype != torch.float16:
        raise TypeError("persisted SQG scale vectors must use torch.float16")
    expected_shape = (k // 16, n // 16, 16 * config.bits)
    if tuple(trellis.shape) != expected_shape:
        raise ValueError(f"packed trellis shape {tuple(trellis.shape)} != {expected_shape}")
    if tuple(suh.shape) != (k,) or tuple(svh.shape) != (n,):
        raise ValueError("stored SQG scale vectors do not match the EXL matrix shape")

    unpacked = unpack_trellis_states(trellis, config.bits)
    states_equal = torch.equal(unpacked, states_cpu)
    repacked_equal = torch.equal(pack_trellis_states(unpacked, config.bits), trellis)
    if not states_equal or not repacked_equal:
        raise RuntimeError("independent packed-state/tail-biting closure failed")
    reconstructed = decode_stored_fp16(
        trellis,
        suh,
        svh,
        bits=config.bits,
        codebook_e4m3=lut,
    )
    if not bool(torch.isfinite(reconstructed).all()):
        raise RuntimeError("independent stored-FP16 SQG decode is non-finite")
    encoder_cpu = encoder_weight.detach().to(device="cpu", dtype=torch.float32)
    encoder_relative_rmse = relative_rmse(reconstructed, encoder_cpu)
    if encoder_relative_rmse > config.closure_max_relative_rmse:
        raise RuntimeError(
            "stored-FP16 decode does not close against KQuant reconstruction: "
            f"relative_rmse={encoder_relative_rmse:.8g}"
        )
    source_relative_rmse = relative_rmse(
        reconstructed,
        prepared_source.detach().to(device="cpu", dtype=torch.float32),
    )
    if (
        config.production
        and config.matrix_role in ("gate", "up")
        and profile is not None
        and profile.side == "input"
    ):
        validate_gate_up_encode_smoke(
            global_scale=float(quant_args["g_scale"]),
            source_relative_rmse=source_relative_rmse,
        )

    if profile is not None:
        actual_shared = suh if profile.side == "input" else svh
        expected_shared = profile.expected_stored_fp16().cpu()
        if not torch.equal(actual_shared, expected_shared):
            raise RuntimeError(
                f"persisted shared {profile.side} residual vector changed during encode"
            )

    input_signs = (-suh.sign()).to(torch.int8)
    output_signs = svh.sign().to(torch.int8)
    hadamard_hash = tensor_sha256(
        normalized_hadamard(device="cpu", dtype=torch.float32)
    )
    transform_manifest: dict[str, object] = {
        **source_transform,
        "kquant_input_sign_seed": config.transform_seed,
        "kquant_output_sign_seed": config.output_sign_seed,
        "input_signs_sha256": tensor_sha256(input_signs),
        "output_signs_sha256": tensor_sha256(output_signs),
        "hadamard_block": HADAMARD_BLOCK,
        "normalized_hadamard_sha256": hadamard_hash,
        "global_scale": float(quant_args["g_scale"]),
        "global_scale_stored_into": (
            "svh" if bool(quant_args.get("g_scale_into_sv")) else "suh"
        ),
        "apply_out_scales": bool(quant_args["apply_out_scales"]),
        "legacy_transform_input": False,
    }
    if profile is not None:
        transform_manifest["shared_residual_profile"] = profile.manifest()
    transform_manifest["transform_sha256"] = _transform_digest(transform_manifest)

    closure_manifest = {
        "implementation": "independent_pytorch_packed_fp16_v1",
        "packed_states_exact": states_equal,
        "repacked_words_exact": repacked_equal,
        "finite": True,
        "encoder_relative_rmse": encoder_relative_rmse,
        "encoder_relative_rmse_limit": config.closure_max_relative_rmse,
        "source_relative_rmse": source_relative_rmse,
        "decoded_exl_sha256": tensor_sha256(reconstructed),
        "passed": True,
    }
    manifest: dict[str, object] = {
        "schema": TENSOR_SCHEMA,
        "tensor_id": config.tensor_id,
        "matrix_role": config.matrix_role,
        "source": config.source_binding.manifest(),
        "bits": config.bits,
        "codebook": SQG_CODEBOOK,
        "codebook_lut_sha256": lut_hash,
        "tailbite_context": TAILBITE_CONTEXT,
        "transform": transform_manifest,
        "scales": {
            "suh_sha256": tensor_sha256(suh),
            "svh_sha256": tensor_sha256(svh),
            "dtype": "float16",
            "freshly_generated": True,
            "legacy_scale_input": False,
        },
        "dense_h": {
            "evidence_id": dense_h.evidence_id,
            "construction": dense_h.construction,
            "split_id": dense_h.split_id,
            "matrix_sha256": hessian_hash,
            "normalization_count": dense_h.normalization_count,
            "routed_sample_count": dense_h.routed_sample_count,
            "sigma_reg": config.sigma_reg,
            "block_ldlq": True,
            "fallback": False,
            "session": dense_h_session.manifest(
                use_ordinal=session_use_ordinal,
                reused_finalized_state=reused_finalized_state,
            ),
        },
        "packed_trellis_sha256": tensor_sha256(trellis),
        "packed_trellis_payload_sha256": payload_sha256(trellis),
        "decoded_closure": closure_manifest,
        "marker": {"suffix": ".sqg", "int32": SQG_MARKER, "hex": "0x53514731"},
        "encoder": {
            "kquant_revision": runtime.revision,
            "backend_sha256": runtime.backend_sha256,
            "working_tree_dirty": runtime.working_tree_dirty,
            "tracked_diff_sha256": runtime.tracked_diff_sha256,
            "status_sha256": runtime.status_sha256,
            "status_porcelain": runtime.status_porcelain,
            "proxy_error": float(proxy_error),
        },
        "forbidden_input_reads": list(FORBIDDEN_MCG_READS),
    }
    validate_tensor_manifest(manifest)
    json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    marker = torch.tensor(SQG_MARKER, dtype=torch.int32)
    return EncodedSQGMatrix(
        bits=config.bits,
        trellis=trellis,
        suh=suh,
        svh=svh,
        sqg=marker,
        reconstructed_exl=reconstructed,
        proxy_error=float(proxy_error),
        manifest=manifest,
    )
