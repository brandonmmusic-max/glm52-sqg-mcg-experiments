"""Strict fresh uniform-K2--K6 SQG bridge for GLM 5.2 matrices."""

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
    W4A8DerivedTensorBinding,
    validate_tensor_manifest,
)
from .permutation import FreshExpertPermutation
from .reference import (
    HADAMARD_BLOCK,
    decode_regularized_states,
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
MatrixRole = Literal[
    "gate",
    "up",
    "down",
    "shared_gate",
    "shared_up",
    "shared_down",
    "q_b",
    "o_proj",
    "synthetic_exl",
]
ResidualSide = Literal["input", "output"]
GLOBAL_SCALE_SEARCH_INITIAL_MAX = 1.9
GLOBAL_SCALE_SEARCH_EXPANDED_MAX = 3.9
GLOBAL_SCALE_SEARCH_CEILING = 4.05
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
PRODUCTION_DENSE_K6_H_CONSTRUCTION = (
    "fit_exact_operand_topology_neutral_dense_covariance_gpu_fp32_v1"
)
PRODUCTION_SHARED_DOWN_HB_CONSTRUCTION = (
    "fit_candidate_specific_full_w4a8_shared_down_cross_term_"
    "qpre_encoder_h_gpu_fp32_v1"
)

_DENSE_K6_ROLE_BY_SUFFIX = {
    "mlp.shared_experts.gate_proj": "shared_gate",
    "mlp.shared_experts.up_proj": "shared_up",
    "mlp.shared_experts.down_proj": "shared_down",
    "self_attn.q_b_proj": "q_b",
    "self_attn.o_proj": "o_proj",
}
_DENSE_K6_SOURCE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\."
    r"(?P<suffix>mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj)|"
    r"self_attn\.(?:q_b_proj|o_proj))\.weight"
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

    def quantize_qsrt_batch(
        self,
        weights: list[torch.Tensor],
        H_datas: list[dict[str, Any]],
        quant_args_groups: list[list[dict[str, Any]]],
        *,
        return_weight_q: bool,
        verbose: bool,
    ) -> list[list[dict[str, object]]]: ...


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


def validate_global_scale_search_evidence(
    value: object,
    *,
    global_scale: float,
) -> dict[str, object]:
    """Validate KQuant's adaptive bracket receipt for a production tensor."""

    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
        raise RuntimeError("KQuant did not emit one global-scale search receipt")
    receipt = dict(value[0])
    if (
        receipt.get("policy") != "raw_sample_adaptive_coarse_then_fine_v2"
        or float(receipt.get("initial_max", math.nan))
        != GLOBAL_SCALE_SEARCH_INITIAL_MAX
        or float(receipt.get("expanded_max", math.nan))
        != GLOBAL_SCALE_SEARCH_EXPANDED_MAX
        or float(receipt.get("hard_ceiling", math.nan))
        != GLOBAL_SCALE_SEARCH_CEILING
        or receipt.get("selected_is_below_hard_ceiling") is not True
        or not isinstance(receipt.get("expanded"), bool)
        or not math.isclose(
            float(receipt.get("selected_scale", math.nan)),
            global_scale,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        if (
            receipt.get("policy") != "activation_parametric_coupled_v1"
            or receipt.get("raw_sample_search_performed") is not False
            or not isinstance(receipt.get("evidence_id"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("evidence_id"))) is None
            or float(receipt.get("hard_ceiling", math.nan))
            != GLOBAL_SCALE_SEARCH_CEILING
            or receipt.get("selected_is_below_hard_ceiling") is not True
            or not math.isclose(
                float(receipt.get("selected_scale", math.nan)),
                global_scale,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError("KQuant global-scale search receipt differs from codec policy")
    if global_scale >= (
        GLOBAL_SCALE_SEARCH_CEILING - GLOBAL_SCALE_SEARCH_CEILING_TOLERANCE
    ):
        raise RuntimeError("KQuant global-scale search receipt selected the hard ceiling")
    return receipt


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
    matrix_sha256: str | None = None

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
        if self.matrix_sha256 is not None and re.fullmatch(
            r"[0-9a-f]{64}", self.matrix_sha256
        ) is None:
            raise ValueError("dense Hessian matrix_sha256 must be SHA-256")


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
    hessian_sha256: str | None
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
    anchored_input_residual_profile: SharedResidualProfile | None = None
    sigma_reg: float = 0.025
    apply_out_scales: bool | None = None
    global_scale_into: ResidualSide = "input"
    global_scale_override: float | None = None
    global_scale_override_policy: str | None = None
    global_scale_override_evidence_id: str | None = None
    device: str = "cuda:0"
    production: bool = True
    closure_max_relative_rmse: float = 0.01
    candidate_sweep_fast: bool = False
    kquant_root: str | None = None
    exllamav3_root: str | None = None

    def __post_init__(self) -> None:
        if not self.tensor_id:
            raise ValueError("tensor_id must not be empty")
        if isinstance(self.bits, bool) or not isinstance(self.bits, int) or self.bits not in range(2, 7):
            raise ValueError("fresh uniform SQG permits integer K2 through K6")
        if self.matrix_role not in (
            "gate",
            "up",
            "down",
            "shared_gate",
            "shared_up",
            "shared_down",
            "q_b",
            "o_proj",
            "synthetic_exl",
        ):
            raise ValueError("unknown GLM matrix role")
        for name in ("transform_seed", "output_sign_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not math.isfinite(self.sigma_reg) or self.sigma_reg <= 0:
            raise ValueError("sigma_reg must be positive and finite")
        if self.global_scale_into not in ("input", "output"):
            raise ValueError("global_scale_into must be input or output")
        override_fields = (
            self.global_scale_override,
            self.global_scale_override_policy,
            self.global_scale_override_evidence_id,
        )
        if any(value is not None for value in override_fields):
            if any(value is None for value in override_fields):
                raise ValueError(
                    "global-scale override, policy, and evidence ID are atomic"
                )
            if (
                isinstance(self.global_scale_override, bool)
                or not isinstance(self.global_scale_override, (int, float))
                or not math.isfinite(float(self.global_scale_override))
                or float(self.global_scale_override) <= 0.0
                or float(self.global_scale_override)
                >= GLOBAL_SCALE_SEARCH_CEILING - GLOBAL_SCALE_SEARCH_CEILING_TOLERANCE
            ):
                raise ValueError(
                    "global-scale override must be positive, finite, and below ceiling"
                )
            if self.global_scale_override_policy != "activation_parametric_coupled_v1":
                raise ValueError(
                    "production global-scale override must be activation-parametric"
                )
            if re.fullmatch(
                r"[0-9a-f]{64}", str(self.global_scale_override_evidence_id)
            ) is None:
                raise ValueError("global-scale override evidence ID must be SHA-256")
        anchor = self.anchored_input_residual_profile
        if anchor is not None:
            if anchor.side != "input":
                raise ValueError("anchored input residual profile must use side='input'")
            if (
                self.shared_residual_profile is None
                or self.shared_residual_profile.side != "output"
            ):
                raise ValueError(
                    "an anchored input profile requires the topology-neutral "
                    "shared output profile"
                )
        if not math.isfinite(self.closure_max_relative_rmse) or self.closure_max_relative_rmse <= 0:
            raise ValueError("closure threshold must be positive and finite")
        if not isinstance(self.candidate_sweep_fast, bool):
            raise TypeError("candidate_sweep_fast must be boolean")
        if isinstance(self.source_binding, W4A8DerivedTensorBinding):
            if self.source_binding.candidate_hashes_deferred is not self.candidate_sweep_fast:
                raise ValueError(
                    "deferred W4A8 target hashes are permitted only for fast candidates"
                )
            if self.matrix_role not in ("down", "shared_down"):
                raise ValueError("W4A8-derived targets are valid only for down encoding")
            if self.matrix_role == "down" and self.physical_permutation is None:
                raise ValueError(
                    "W4A8-derived targets require their bound physical permutation"
                )
            if self.matrix_role == "shared_down" and self.physical_permutation is not None:
                raise ValueError(
                    "shared-down W4A8 targets must remain topology-neutral"
                )
        if self.production:
            if isinstance(self.source_binding, BF16TensorBinding):
                official_binding = self.source_binding
            elif isinstance(self.source_binding, W4A8DerivedTensorBinding):
                official_binding = self.source_binding.official_bf16_parent
            else:
                raise TypeError(
                    "production encoding requires an immutable official BF16 "
                    "or fit-only W4A8-derived binding"
                )
            if self.matrix_role == "synthetic_exl":
                raise ValueError("synthetic_exl is forbidden for production encoding")
            source_match = re.fullmatch(
                r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
                r"(gate_proj|up_proj|down_proj)\.weight",
                official_binding.tensor_name,
            )
            dense_match = _DENSE_K6_SOURCE.fullmatch(official_binding.tensor_name)
            if source_match is not None:
                if self.physical_permutation is None:
                    raise ValueError(
                        "routed production encoding requires a fresh physical permutation"
                    )
                if not self.physical_permutation.production_qualified:
                    raise ValueError(
                        "routed production encoding requires a calibration-derived "
                        "GLM h2_reverse permutation"
                    )
                projection_by_role = {
                    "gate": "gate_proj",
                    "up": "up_proj",
                    "down": "down_proj",
                }
                if source_match.group(3) != projection_by_role.get(self.matrix_role):
                    raise ValueError(
                        "matrix role disagrees with official routed BF16 tensor name"
                    )
            elif dense_match is not None:
                if self.physical_permutation is not None:
                    raise ValueError(
                        "dense K6 production encoding must remain topology-neutral"
                    )
                expected_role = _DENSE_K6_ROLE_BY_SUFFIX[dense_match.group("suffix")]
                if self.matrix_role != expected_role:
                    raise ValueError(
                        "matrix role disagrees with official dense-K6 BF16 tensor name"
                    )
                if not 3 <= int(dense_match.group("layer")) <= 78:
                    raise ValueError("dense K6 production layer lies outside 3..78")
            else:
                raise ValueError(
                    "production source is not a routed expert or dense-K6 GLM tensor"
                )
            expected_tensor_id = official_binding.tensor_name.removesuffix(".weight")
            if self.tensor_id != expected_tensor_id:
                raise ValueError(
                    "production tensor ID disagrees with official BF16 tensor binding"
                )
            if source_match is not None:
                expected_scope = (
                    f"layer-{int(source_match.group(1)):03d}/"
                    f"expert-{int(source_match.group(2)):03d}"
                )
                if self.physical_permutation.scope != expected_scope:
                    raise ValueError(
                        "calibration-derived permutation scope does not match "
                        "the official BF16 tensor binding"
                    )
        elif not isinstance(
            self.source_binding,
            (BF16TensorBinding, W4A8DerivedTensorBinding, SyntheticTensorBinding),
        ):
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


@dataclass(frozen=True)
class UniformSQGCandidateRequest:
    """One fixed-rate, fixed-scale member of a production candidate batch."""

    source: torch.Tensor = field(repr=False)
    config: UniformSQGConfig

    def __post_init__(self) -> None:
        if not isinstance(self.source, torch.Tensor):
            raise TypeError("candidate source must be a torch.Tensor")


def _input_transform_binding(config: UniformSQGConfig) -> str:
    profile = (
        config.anchored_input_residual_profile
        or config.shared_residual_profile
    )
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
    if config.candidate_sweep_fast:
        matrix = dense_h.matrix
        if tuple(matrix.shape) != (in_features, in_features) or matrix.is_meta:
            raise ValueError("fast-candidate dense Hessian geometry differs")
        hessian_hash = None
    else:
        _validate_dense_hessian(dense_h, in_features)
        hessian_hash = dense_h.matrix_sha256 or tensor_sha256(dense_h.matrix)
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
            "keep_hessian_on_device": True,
            "gpu_only_numerical_path": True,
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
        expected_path = (root / "kquant" / "sqg_e4m3.py").resolve()
        try:
            same_checkout_file = loaded_path.samefile(expected_path)
        except (FileNotFoundError, OSError):
            same_checkout_file = False
        if not same_checkout_file:
            raise ImportError(
                f"a different kquant package is already loaded from {loaded_path}"
            )
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
    if isinstance(config.source_binding, W4A8DerivedTensorBinding):
        binding = config.source_binding
        if dense_h.evidence_id != binding.fit_hb_evidence_id:
            raise ValueError("W4A8-derived dense-H evidence identity differs")
        if binding.candidate_hashes_deferred:
            if not config.candidate_sweep_fast:
                raise ValueError(
                    "deferred W4A8 Hessian hash is valid only for fast candidates"
                )
            return
        if tensor_sha256(dense_h.matrix) != binding.fit_hessian_sha256:
            raise ValueError("W4A8-derived dense-H payload hash differs")
        return
    if config.matrix_role == "down":
        expected = PRODUCTION_H2_CONSTRUCTION
    elif config.matrix_role in ("gate", "up"):
        expected = PRODUCTION_H13_CONSTRUCTION
    elif config.matrix_role == "shared_down":
        expected = PRODUCTION_SHARED_DOWN_HB_CONSTRUCTION
    else:
        expected = PRODUCTION_DENSE_K6_H_CONSTRUCTION
    if dense_h.construction != expected:
        raise ValueError(
            f"production {config.matrix_role} dense-H construction differs: "
            f"{dense_h.construction!r} != {expected!r}"
        )


def _prepare_exl_source(
    source: torch.Tensor,
    config: UniformSQGConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Prepare one bound official-HF or already-derived EXL target privately."""

    if source.ndim != 2 or not source.is_floating_point():
        raise TypeError("source must be a rank-two floating-point matrix")
    fast_candidate = config.production and config.candidate_sweep_fast
    if fast_candidate:
        # Candidate inputs are already bound by the sealed BF16 provider or by
        # the just-created W4A8DerivedTensorBinding.  Rehashing 50 MB-scale
        # CUDA tensors for every scale/cell is byte-neutral.  Keep dtype,
        # contract and mutation guards here; selected materialization performs
        # the authoritative full payload validation and hash.
        binding = config.source_binding
        if isinstance(binding, BF16TensorBinding):
            if source.dtype != torch.bfloat16:
                raise TypeError("production SQG source tensors must be official BF16")
        elif isinstance(binding, W4A8DerivedTensorBinding):
            binding._validated_execution_contract()
            if source.dtype != torch.float32:
                raise TypeError("W4A8-derived candidate targets must be float32")
        else:  # production config validation forbids synthetic bindings
            raise TypeError("fast candidates require an immutable production binding")
    else:
        config.source_binding.validate_tensor(source)
    source_data_ptr = source.data_ptr()
    source_version = source._version
    role = config.matrix_role
    if isinstance(config.source_binding, W4A8DerivedTensorBinding):
        permutation = config.physical_permutation
        if role == "shared_down":
            if permutation is not None:  # pragma: no cover - guarded by config
                raise ValueError("shared-down W4A8 target changed topology")
            prepared = source.detach().clone().contiguous()
            transform = {
                "source_orientation": "exl_input_output",
                "operation": "w4a8_derived_dense_exl_clone_only",
                "topology_neutral_dense_input": True,
                "physical_input_order": "identity",
                "official_bf16_parent_tensor_name": (
                    config.source_binding.official_bf16_parent.tensor_name
                ),
            }
        else:
            if permutation is None:  # pragma: no cover - guarded by config validation
                raise ValueError("W4A8-derived target lacks its physical permutation")
            if source.shape[0] != permutation.intermediate_features:
                raise ValueError(
                    "W4A8-derived EXL input axis does not match fresh permutation"
                )
            prepared = source.detach().clone().contiguous()
            transform = {
                "source_orientation": "exl_input_output",
                "operation": "w4a8_derived_exl_clone_only",
                "physical_permutation": permutation.manifest(),
                "official_bf16_parent_tensor_name": (
                    config.source_binding.official_bf16_parent.tensor_name
                ),
            }
    elif role == "synthetic_exl":
        prepared = source.detach().clone().contiguous()
        transform = {"source_orientation": "exl", "operation": "clone_only"}
    elif role in ("shared_gate", "shared_up", "shared_down", "q_b", "o_proj"):
        if config.physical_permutation is not None:
            raise ValueError("dense-K6 preparation must remain topology-neutral")
        prepared = source.detach().T.clone().contiguous()
        transform = {
            "source_orientation": "huggingface",
            "operation": "topology_neutral_hf_to_exl_transpose",
            "topology_neutral_dense_input": True,
            "physical_input_order": "identity",
        }
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
    if source.data_ptr() != source_data_ptr or source._version != source_version:
        raise RuntimeError("source matrix changed while preparing the private encode clone")
    if prepared.data_ptr() == source.data_ptr():
        raise RuntimeError("prepared source unexpectedly aliases the caller-owned tensor")
    if fast_candidate and isinstance(config.source_binding, W4A8DerivedTensorBinding):
        transform["prepared_exl_sha256"] = config.source_binding.derived_tensor_sha256
        transform["candidate_prepared_hash_deferred"] = True
    elif fast_candidate:
        transform["prepared_exl_sha256"] = None
        transform["candidate_prepared_hash_deferred"] = True
    else:
        transform["prepared_exl_sha256"] = tensor_sha256(prepared)
        transform["candidate_prepared_hash_deferred"] = False
    transform["prepared_exl_dtype"] = str(prepared.dtype).removeprefix("torch.")
    transform["prepared_exl_shape"] = list(prepared.shape)
    return prepared, transform


def _transform_digest(transform: Mapping[str, object]) -> str:
    encoded = json.dumps(transform, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _finalize_encoded_sqg_result(
    *,
    prepared_source: torch.Tensor,
    source_transform: Mapping[str, object],
    dense_h: DenseHessian,
    config: UniformSQGConfig,
    runtime: KQuantRuntime,
    dense_h_session: DenseHSession,
    session_use_ordinal: int,
    reused_finalized_state: bool,
    lut: torch.Tensor,
    lut_hash: str,
    quant_args: Mapping[str, Any],
    encoder_weight: torch.Tensor,
    proxy_error: float,
    raw_tensors: Mapping[str, torch.Tensor],
    captured_states: torch.Tensor,
    backend_execution: Mapping[str, object] | None = None,
) -> EncodedSQGMatrix:
    """Apply the one authoritative packed/decode/manifest closure.

    Serial and batched encodes both terminate here.  Keeping one closure is
    deliberate: batching may alter the optimizer trajectory, but it may not
    weaken the persisted-byte, direct-E4M3, Hessian, or provenance contract.
    """

    k, n = prepared_source.shape
    profile = config.shared_residual_profile
    input_anchor = config.anchored_input_residual_profile
    if set(raw_tensors) != {"trellis", "suh", "svh"}:
        raise RuntimeError(
            f"KQuant emitted non-SQG or unexpected tensors: {sorted(raw_tensors)}"
        )
    for name, value in raw_tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"KQuant {name} output is not a torch.Tensor")
    if not isinstance(encoder_weight, torch.Tensor):
        raise TypeError("KQuant reconstructed weight is not a torch.Tensor")
    if not isinstance(captured_states, torch.Tensor):
        raise TypeError("KQuant encoded state output is not a torch.Tensor")
    if not math.isfinite(float(proxy_error)):
        raise RuntimeError("KQuant proxy error is non-finite")

    numerical_device = torch.device(config.device)
    if config.production:
        numerical_tensors: dict[str, torch.Tensor] = {
            "prepared_source": prepared_source,
            "encoder_weight": encoder_weight,
            "captured_states": captured_states,
            **{f"raw_{name}": value for name, value in raw_tensors.items()},
        }
        for name in ("H", "L"):
            value = dense_h_session.h_data.get(name)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"production dense-H session lacks {name}")
            numerical_tensors[f"dense_h_{name}"] = value
        off_device = {
            name: str(value.device)
            for name, value in numerical_tensors.items()
            if value.device.type != "cuda"
        }
        if off_device:
            raise RuntimeError(
                "production SQG numerical tensor left CUDA before closure: "
                f"{off_device}"
            )
    trellis_device = raw_tensors["trellis"].detach().to(
        device=numerical_device, copy=True
    ).contiguous()
    suh_device = raw_tensors["suh"].detach().to(
        device=numerical_device, copy=True
    ).contiguous()
    svh_device = raw_tensors["svh"].detach().to(
        device=numerical_device, copy=True
    ).contiguous()
    states_device = captured_states.detach().to(
        device=numerical_device, copy=True
    ).contiguous()
    if trellis_device.dtype != torch.int16:
        raise TypeError("packed trellis must use torch.int16")
    if states_device.dtype != torch.int16:
        raise TypeError("unpacked trellis states must use torch.int16")
    if suh_device.dtype != torch.float16 or svh_device.dtype != torch.float16:
        raise TypeError("persisted SQG scale vectors must use torch.float16")
    expected_shape = (k // 16, n // 16, 16 * config.bits)
    if tuple(trellis_device.shape) != expected_shape:
        raise ValueError(
            f"packed trellis shape {tuple(trellis_device.shape)} != {expected_shape}"
        )
    if tuple(states_device.shape) != (k // 16, n // 16, 256):
        raise ValueError("unpacked trellis states do not match the EXL matrix shape")
    if tuple(suh_device.shape) != (k,) or tuple(svh_device.shape) != (n,):
        raise ValueError("stored SQG scale vectors do not match the EXL matrix shape")

    # Candidate searches may encode dozens of alternatives for one final
    # tensor.  Re-decoding and copying the full FP32 reconstruction for every
    # loser is byte-neutral and can dominate paid-node wall clock.  Preserve a
    # full independent closure in the separately sealed campaign preflight;
    # all sweep candidates perform structural CUDA checks only.  The selected
    # final encode never sets candidate_sweep_fast and therefore always
    # receives the complete persisted-byte closure below.
    full_decode_closure = not config.candidate_sweep_fast
    if full_decode_closure:
        unpacked = unpack_trellis_states(trellis_device, config.bits)
        states_equal: bool | None = torch.equal(unpacked, states_device)
        repacked_equal: bool | None = torch.equal(
            pack_trellis_states(unpacked, config.bits), trellis_device
        )
        if not states_equal or not repacked_equal:
            raise RuntimeError("independent packed-state/tail-biting closure failed")
        regularized_labels = decode_regularized_states(unpacked, lut)
        direct_e4m3_exact: bool | None = torch.equal(
            regularized_labels,
            regularized_labels.to(torch.float8_e4m3fn).float(),
        )
        if not direct_e4m3_exact:
            raise RuntimeError("decoded SQG labels are not exact finite E4M3 values")
        reconstructed_device = decode_stored_fp16(
            trellis_device,
            suh_device,
            svh_device,
            bits=config.bits,
            codebook_e4m3=lut,
        )
        if not bool(torch.isfinite(reconstructed_device).all()):
            raise RuntimeError("independent stored-FP16 SQG decode is non-finite")
        encoder_device = encoder_weight.detach().to(
            device=numerical_device, dtype=torch.float32
        )
        if tuple(encoder_device.shape) != (k, n) or not bool(
            torch.isfinite(encoder_device).all()
        ):
            raise RuntimeError("KQuant reconstruction is malformed or non-finite")
        encoder_relative_rmse: float | None = relative_rmse(
            reconstructed_device, encoder_device
        )
        if encoder_relative_rmse > config.closure_max_relative_rmse:
            raise RuntimeError(
                "stored-FP16 decode does not close against KQuant reconstruction: "
                f"relative_rmse={encoder_relative_rmse:.8g}"
            )
        source_relative_rmse: float | None = relative_rmse(
            reconstructed_device,
            prepared_source.detach().to(device=numerical_device, dtype=torch.float32),
        )
    else:
        states_equal = None
        repacked_equal = None
        direct_e4m3_exact = None
        reconstructed_device = None
        encoder_relative_rmse = None
        source_relative_rmse = None

    # Structural/full numerical checks are complete on CUDA.  Copy only the
    # compact persisted payload for candidate sweeps; the ~50 MB FP32 decoded
    # audit tensor crosses to CPU solely for full closures.
    trellis = trellis_device.to(device="cpu", copy=True).contiguous()
    suh = suh_device.to(device="cpu", copy=True).contiguous()
    svh = svh_device.to(device="cpu", copy=True).contiguous()
    reconstructed = (
        reconstructed_device.to(device="cpu", copy=True).contiguous()
        if reconstructed_device is not None
        else torch.empty(0, dtype=torch.float32)
    )
    if (
        config.production
        and config.matrix_role in ("gate", "up", "shared_gate", "shared_up")
        and profile is not None
        and profile.side == "input"
    ):
        if source_relative_rmse is not None:
            validate_gate_up_encode_smoke(
                global_scale=float(quant_args["g_scale"]),
                source_relative_rmse=source_relative_rmse,
            )
        global_scale_search = validate_global_scale_search_evidence(
            quant_args.get("g_scale_search_evidence"),
            global_scale=float(quant_args["g_scale"]),
        )
    else:
        global_scale_search = None

    if profile is not None:
        actual_shared = suh if profile.side == "input" else svh
        expected_shared = profile.expected_stored_fp16().cpu()
        if not torch.equal(actual_shared, expected_shared):
            raise RuntimeError(
                f"persisted shared {profile.side} residual vector changed during encode"
            )
    if input_anchor is not None:
        expected_anchor = input_anchor.expected_stored_fp16().cpu()
        if not torch.equal(suh, expected_anchor):
            raise RuntimeError(
                "persisted anchored input residual vector changed during encode"
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
    if global_scale_search is not None:
        transform_manifest["global_scale_search"] = global_scale_search
    if profile is not None:
        transform_manifest["shared_residual_profile"] = profile.manifest()
    if input_anchor is not None:
        transform_manifest["anchored_input_residual_profile"] = input_anchor.manifest()
    transform_manifest["transform_sha256"] = _transform_digest(transform_manifest)

    closure_manifest = {
        "implementation": (
            "independent_pytorch_packed_fp16_v1"
            if full_decode_closure
            else "candidate_sweep_structural_cuda_v1"
        ),
        "mode": (
            "selected_full"
            if not config.candidate_sweep_fast
            else "candidate_sweep_fast"
        ),
        "packed_states_exact": states_equal,
        "repacked_words_exact": repacked_equal,
        "direct_e4m3_labels_exact": direct_e4m3_exact,
        "finite": True if full_decode_closure else None,
        "encoder_relative_rmse": encoder_relative_rmse,
        "encoder_relative_rmse_limit": config.closure_max_relative_rmse,
        "source_relative_rmse": source_relative_rmse,
        "decoded_exl_sha256": (
            tensor_sha256(reconstructed) if full_decode_closure else None
        ),
        "full_decode_deferred": not full_decode_closure,
        "selected_model_eligible": not config.candidate_sweep_fast,
        "passed": True,
    }
    encoder_manifest: dict[str, object] = {
        "production": config.production,
        "kquant_revision": runtime.revision,
        "backend_sha256": runtime.backend_sha256,
        "working_tree_dirty": runtime.working_tree_dirty,
        "tracked_diff_sha256": runtime.tracked_diff_sha256,
        "status_sha256": runtime.status_sha256,
        "status_porcelain": runtime.status_porcelain,
        "proxy_error": float(proxy_error),
        "numerical_device": str(numerical_device),
        "cpu_numerical_work": numerical_device.type != "cuda",
        "bf16_source_on_cuda": prepared_source.device.type == "cuda",
        "hessian_on_cuda": all(
            isinstance(dense_h_session.h_data.get(name), torch.Tensor)
            and dense_h_session.h_data[name].device.type == "cuda"
            for name in ("H", "L")
        ),
        "closure_on_cuda": numerical_device.type == "cuda",
        "candidate_sweep_fast_requested": config.candidate_sweep_fast,
        "candidate_source_hash_deferred": (
            config.production and config.candidate_sweep_fast
        ),
        "full_decode_closure_performed": full_decode_closure,
        "fp32_accumulate": True,
        "cpu_roles_after_numerical_closure": ["hashing", "serialization"],
    }
    if backend_execution is not None:
        encoder_manifest["backend_execution"] = dict(backend_execution)
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
            "matrix_sha256": dense_h_session.hessian_sha256,
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
        "encoder": encoder_manifest,
        "forbidden_input_reads": list(FORBIDDEN_MCG_READS),
    }
    validate_tensor_manifest(manifest)
    json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return EncodedSQGMatrix(
        bits=config.bits,
        trellis=trellis,
        suh=suh,
        svh=svh,
        sqg=torch.tensor(SQG_MARKER, dtype=torch.int32),
        reconstructed_exl=reconstructed,
        proxy_error=float(proxy_error),
        manifest=manifest,
    )


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
    decision allowed outside this call is the caller's frozen per-tensor bit map.
    """

    prepared_source, source_transform = _prepare_exl_source(source, config)
    k, n = prepared_source.shape
    if config.production:
        _validate_production_dense_hessian(dense_h, config)
    if k % HADAMARD_BLOCK or n % HADAMARD_BLOCK:
        raise ValueError("EXL K and N dimensions must both be divisible by 128")
    if not config.candidate_sweep_fast and not bool(
        torch.isfinite(prepared_source.float()).all()
    ):
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
        if config.production and config.matrix_role in (
            "gate",
            "up",
            "shared_gate",
            "shared_up",
        ):
            raise ValueError(
                "production gate/up encoding requires an explicit reusable "
                "DenseHSession"
            )
        dense_h_session = prepare_dense_h_session(dense_h, config)
    _validate_dense_h_session(dense_h_session, dense_h, config, k)
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
        "keep_hessian_on_device": True,
        "gpu_only_numerical_path": True,
    }
    if config.global_scale_override is not None:
        quant_args.update(
            {
                "g_scale_override": float(config.global_scale_override),
                "g_scale_override_policy": config.global_scale_override_policy,
                "g_scale_override_evidence_id": (
                    config.global_scale_override_evidence_id
                ),
            }
        )
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
    input_anchor = config.anchored_input_residual_profile
    if input_anchor is not None:
        if input_anchor.signs.numel() != k:
            raise ValueError(
                "anchored input residual profile length does not match its residual axis"
            )
        quant_args["input_signs"] = input_anchor.signs
        quant_args["input_channel_scale_profile"] = input_anchor.channel_scales
        # Both persisted residual sides are caller-owned in this mode.  The
        # searched scalar has nowhere to move without changing one of those
        # exact FP16 vectors, so the regularized weight is encoded at g=1.
        if config.global_scale_override is not None:
            raise ValueError(
                "activation-parametric global-scale override cannot be combined "
                "with dual caller-owned residual anchors"
            )
        quant_args["g_scale_override"] = 1.0
    forbidden_quant_args = {"mcg", "mul1", "shared_input_scales_key"}
    if forbidden_quant_args.intersection(quant_args):  # pragma: no cover
        raise RuntimeError("fresh SQG quant_args contain a forbidden legacy/shared-state key")

    with dense_h_session._lock:
        reused_finalized_state = bool(h_data.get("finalized"))
        session_use_ordinal = dense_h_session.use_count + 1
        if input_anchor is None:
            encoder_weight, proxy_error, raw_tensors = runtime.backend.quantize_qsrt(
                work,
                h_data,
                quant_args,
                True,
                progress_str=None,
                verbose=False,
            )
        else:
            # The serial KQuant entry point predates ``g_scale_override`` and
            # silently performs a scale search.  With caller-owned input and
            # output residual vectors there is nowhere to store that scalar
            # without changing one of the exact FP16 vectors.  The existing
            # batched mixed-rate entry point honors the override, so execute
            # one constant-rate member through it and pack its states here.
            batch_api = getattr(runtime.backend, "quantize_qsrt_batch", None)
            if not callable(batch_api):
                raise RuntimeError(
                    "anchored input/output encoding requires KQuant's "
                    "g-scale-aware batch API"
                )
            quant_args["mixed_rate_axis"] = "n"
            quant_args["mixed_tile_bits"] = (config.bits,) * (n // 16)
            quant_args["sqg_e4m3_luts_by_bits"] = {config.bits: lut}
            quant_args.pop("sqg_e4m3_lut", None)
            batch_results = batch_api(
                [work],
                [h_data],
                [[quant_args]],
                return_weight_q=True,
                verbose=False,
            )
            if (
                not isinstance(batch_results, list)
                or len(batch_results) != 1
                or not isinstance(batch_results[0], list)
                or len(batch_results[0]) != 1
                or not isinstance(batch_results[0][0], dict)
            ):
                raise RuntimeError("KQuant returned malformed anchored batch output")
            batch_result = batch_results[0][0]
            if set(batch_result) != {
                "weight_q",
                "encoded",
                "suh",
                "svh",
                "proxy",
                "g_scale",
            }:
                raise RuntimeError(
                    "KQuant anchored batch output contains unexpected fields"
                )
            encoder_weight = batch_result["weight_q"]
            proxy_error = batch_result["proxy"]
            packed = strict_pack(batch_result["encoded"], quant_args)
            raw_tensors = {
                "trellis": packed,
                "suh": batch_result["suh"],
                "svh": batch_result["svh"],
            }
            if float(batch_result["g_scale"]) != 1.0:
                raise RuntimeError("KQuant changed the anchored global scale")
        if quant_args.get("q_fallback") is not False:
            raise RuntimeError("KQuant attempted a forbidden non-dense-H fallback")
        factorization = h_data.get("L")
        if not isinstance(factorization, torch.Tensor):
            raise RuntimeError("KQuant did not retain the no-fallback BlockLDL state")
        if (
            dense_h_session.factorization_sha256 is None
            and not config.candidate_sweep_fast
        ):
            # The factor is caller-owned and KQuant reuses it read-only.  Hash
            # once; rehashing a 6144-square H13 for every expert would add a
            # large byte-neutral cost to the timed encode path.
            dense_h_session.factorization_sha256 = tensor_sha256(factorization)
        dense_h_session.use_count = session_use_ordinal
    if captured_states is None:
        raise RuntimeError("KQuant did not emit packed trellis states")
    return _finalize_encoded_sqg_result(
        prepared_source=prepared_source,
        source_transform=source_transform,
        dense_h=dense_h,
        config=config,
        runtime=runtime,
        dense_h_session=dense_h_session,
        session_use_ordinal=session_use_ordinal,
        reused_finalized_state=reused_finalized_state,
        lut=lut,
        lut_hash=lut_hash,
        quant_args=quant_args,
        encoder_weight=encoder_weight,
        proxy_error=float(proxy_error),
        raw_tensors=raw_tensors,
        captured_states=captured_states,
    )


def encode_uniform_sqg_candidate_batch(
    requests: Sequence[UniformSQGCandidateRequest],
    dense_h: DenseHessian,
    *,
    runtime: KQuantRuntime,
    dense_h_session: DenseHSession,
    materialize_indices: Sequence[int] | None = None,
    full_closure_indices: Sequence[int] = (),
) -> list[EncodedSQGMatrix]:
    """Encode fixed K3/K4, activation-parametric candidates in one LDLQ batch.

    Each source is cloned independently and has one frozen candidate.  The
    shared object is only the caller-owned finalized dense-H session.  The
    function makes no claim that bytes equal a serial run: its reproducibility
    contract is deterministic equality against an identical batch rerun.
    """

    members = list(requests)
    if not members:
        raise ValueError("candidate batch must not be empty")
    if materialize_indices is None:
        materialized = tuple(range(len(members)))
    else:
        materialized = tuple(int(index) for index in materialize_indices)
    full_closure = frozenset(int(index) for index in full_closure_indices)
    if (
        len(set(materialized)) != len(materialized)
        or any(not 0 <= index < len(members) for index in materialized)
        or any(not 0 <= index < len(members) for index in full_closure)
        or not full_closure.issubset(materialized)
    ):
        raise ValueError("candidate batch materialization indices differ")
    if not isinstance(runtime, KQuantRuntime):
        raise TypeError("candidate batch requires an explicit KQuantRuntime")
    batch_api = getattr(runtime.backend, "quantize_qsrt_batch", None)
    if not callable(batch_api):
        raise RuntimeError("KQuant runtime lacks the fixed-candidate batch API")

    prepared: list[torch.Tensor] = []
    transforms: list[dict[str, object]] = []
    works: list[torch.Tensor] = []
    configs: list[UniformSQGConfig] = []
    quant_args_groups: list[list[dict[str, Any]]] = []
    luts: list[torch.Tensor] = []
    lut_hashes: list[str] = []
    seen_candidates: set[tuple[str, int, str, str]] = set()
    expected_shape: tuple[int, int] | None = None
    device: torch.device | None = None

    for index, request in enumerate(members):
        if not isinstance(request, UniformSQGCandidateRequest):
            raise TypeError(f"candidate batch member {index} has the wrong type")
        config = request.config
        if config.production:
            _validate_production_runtime(runtime)
            _validate_production_dense_hessian(dense_h, config)
            if config.matrix_role not in ("gate", "up"):
                raise ValueError("production candidate batching is gate/up-only")
        elif config.matrix_role not in ("gate", "up", "synthetic_exl"):
            raise ValueError("candidate batching supports only gate/up-shaped matrices")
        profile = config.shared_residual_profile
        if profile is None or profile.side != "input":
            raise ValueError(
                "candidate batching requires an explicit shared input residual profile"
            )
        if config.anchored_input_residual_profile is not None:
            raise ValueError("candidate batching forbids dual residual anchors")
        if (
            config.global_scale_override is None
            or config.global_scale_override_policy != "activation_parametric_coupled_v1"
            or config.global_scale_override_evidence_id is None
        ):
            raise ValueError(
                "candidate batching requires a fixed activation-parametric scale receipt"
            )
        candidate_key = (
            config.tensor_id,
            config.bits,
            float(config.global_scale_override).hex(),
            config.global_scale_override_evidence_id,
        )
        if candidate_key in seen_candidates:
            raise ValueError("candidate batch repeats an identical tensor/rate/scale member")
        seen_candidates.add(candidate_key)

        prepared_source, source_transform = _prepare_exl_source(request.source, config)
        k, n = prepared_source.shape
        if profile.signs.numel() != k:
            raise ValueError(
                "shared input profile length does not match the EXL input axis"
            )
        if k % HADAMARD_BLOCK or n % HADAMARD_BLOCK:
            raise ValueError("EXL K and N dimensions must both be divisible by 128")
        if not config.candidate_sweep_fast and not bool(
            torch.isfinite(prepared_source.float()).all()
        ):
            raise ValueError("source matrix must contain only finite values")
        if expected_shape is None:
            expected_shape = (k, n)
        elif expected_shape != (k, n):
            raise ValueError("candidate batch sources must have one EXL matrix shape")
        member_device = torch.device(config.device)
        if device is None:
            device = member_device
        elif device != member_device:
            raise ValueError("candidate batch members must use one encode device")
        if runtime.requires_cuda and member_device.type != "cuda":
            raise ValueError("the production SQG Viterbi encoder requires one CUDA device")
        _validate_dense_h_session(dense_h_session, dense_h, config, k)

        lut = runtime.lut_bytes(config.bits).detach().to(device="cpu").contiguous()
        if lut.dtype != torch.uint8 or tuple(lut.shape) != (1 << 16,):
            raise RuntimeError("KQuant returned a malformed SQG E4M3 LUT")
        lut_hash = payload_sha256(lut)
        if lut_hash != LUT_SHA256[config.bits]:
            raise RuntimeError(f"SQG K{config.bits} LUT identity mismatch: {lut_hash}")
        work = prepared_source.detach().to(
            device=member_device,
            dtype=torch.float32,
            copy=True,
        ).contiguous()
        if work.data_ptr() == prepared_source.data_ptr():
            raise RuntimeError("KQuant work matrix aliases the prepared source")
        if any(work.data_ptr() == prior.data_ptr() for prior in works):
            raise RuntimeError("candidate batch work matrices unexpectedly alias")

        args: dict[str, Any] = {
            "K": config.bits,
            "seed": config.transform_seed,
            "sv_seed": config.output_sign_seed,
            "sigma_reg": config.sigma_reg,
            "devices": [member_device],
            "apply_out_scales": config.apply_out_scales,
            "g_scale_into_sv": True,
            # KQuant treats these as inputs, but every member still receives
            # private tensors so no backend mutation can couple candidates.
            "input_signs": profile.signs.detach().clone().contiguous(),
            "input_channel_scale_profile": (
                profile.channel_scales.detach().clone().contiguous()
            ),
            "g_scale_override": float(config.global_scale_override),
            "g_scale_override_policy": config.global_scale_override_policy,
            "g_scale_override_evidence_id": config.global_scale_override_evidence_id,
            "mixed_rate_axis": "n",
            "mixed_tile_bits": (config.bits,) * (n // 16),
            "sqg_e4m3_luts_by_bits": {config.bits: lut},
            "tailbite_context": TAILBITE_CONTEXT,
            "keep_hessian_on_device": True,
            "gpu_only_numerical_path": True,
        }
        forbidden_quant_args = {"mcg", "mul1", "shared_input_scales_key"}
        if forbidden_quant_args.intersection(args):  # pragma: no cover
            raise RuntimeError("candidate batch contains a forbidden legacy key")
        configs.append(config)
        prepared.append(prepared_source)
        transforms.append(source_transform)
        works.append(work)
        quant_args_groups.append([args])
        luts.append(lut)
        lut_hashes.append(lut_hash)

    # KQuant's mixed-rate batched LDLQ groups tiles by rate and seeds each
    # grouped quantizer call from quant_args_list[0].  Every member therefore
    # needs the complete canonical LUT domain represented by this physical
    # batch, even though its own fixed-rate tile map contains only one rate.
    # Keeping a one-entry mapping here makes a K4-first batch fail as soon as
    # the backend reaches its K3 slice (and vice versa).
    batch_luts_by_bits: dict[int, torch.Tensor] = {}
    for config, lut in zip(configs, luts, strict=True):
        prior = batch_luts_by_bits.get(config.bits)
        if prior is not None and not torch.equal(prior, lut):
            raise RuntimeError(f"candidate batch has conflicting SQG K{config.bits} LUTs")
        batch_luts_by_bits[config.bits] = lut
    for group in quant_args_groups:
        group[0]["sqg_e4m3_luts_by_bits"] = dict(batch_luts_by_bits)

    h_data = dense_h_session.h_data
    with dense_h_session._lock:
        start_use_count = dense_h_session.use_count
        initially_finalized = bool(h_data.get("finalized"))
        batch_results = batch_api(
            works,
            [h_data] * len(works),
            quant_args_groups,
            return_weight_q=True,
            verbose=False,
        )
        if not isinstance(batch_results, list) or len(batch_results) != len(members):
            raise RuntimeError("KQuant returned a partial/malformed candidate batch")
        expected_fields = {
            "weight_q",
            "encoded",
            "suh",
            "svh",
            "proxy",
            "g_scale",
        }
        normalized: list[dict[str, object]] = []
        for index, group in enumerate(batch_results):
            if (
                not isinstance(group, list)
                or len(group) != 1
                or not isinstance(group[0], dict)
                or set(group[0]) != expected_fields
            ):
                raise RuntimeError(
                    f"KQuant candidate batch member {index} is malformed or partial"
                )
            result = group[0]
            config = configs[index]
            scale = result["g_scale"]
            if (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or float(scale) != float(config.global_scale_override)
            ):
                raise RuntimeError(
                    f"KQuant changed fixed global scale for candidate {index}"
                )
            args = quant_args_groups[index][0]
            if args.get("q_fallback") is not False:
                raise RuntimeError(
                    f"KQuant attempted a forbidden fallback for candidate {index}"
                )
            args["g_scale_search_evidence"] = [
                {
                    "policy": config.global_scale_override_policy,
                    "evidence_id": config.global_scale_override_evidence_id,
                    "raw_sample_search_performed": False,
                    "hard_ceiling": GLOBAL_SCALE_SEARCH_CEILING,
                    "selected_scale": float(config.global_scale_override),
                    "selected_is_below_hard_ceiling": True,
                }
            ]
            normalized.append(result)

        factorization = h_data.get("L")
        if not isinstance(factorization, torch.Tensor):
            raise RuntimeError("KQuant did not retain the no-fallback BlockLDL state")
        if (
            dense_h_session.factorization_sha256 is None
            and (
                not configs[0].candidate_sweep_fast
                or bool(full_closure)
            )
        ):
            # As in the serial path, hash the caller-owned factor once.  A
            # repeated 6144-square CPU transfer would erase the batch win.
            dense_h_session.factorization_sha256 = tensor_sha256(factorization)

        if full_closure and dense_h_session.hessian_sha256 is None:
            raise RuntimeError(
                "selected closure replay requires a fully bound dense-H session"
            )
        outputs: list[EncodedSQGMatrix] = []
        for index in materialized:
            result = normalized[index]
            config = configs[index]
            if index in full_closure:
                if not config.candidate_sweep_fast:
                    raise ValueError(
                        "full-closure replay expects scorer-fast request configs"
                    )
                # This changes closure/provenance work only. Candidate-sweep
                # mode is not passed to KQuant and therefore cannot change the
                # already-computed batch member's states, scales, or proxy.
                config = replace(config, candidate_sweep_fast=False)
            states = result["encoded"]
            if not isinstance(states, torch.Tensor):
                raise TypeError(f"KQuant encoded states for candidate {index} are malformed")
            args = quant_args_groups[index][0]
            packed = runtime.pack_states(states, args)
            backend_execution = {
                "mode": "quantize_qsrt_batch_fixed_uniform_candidates_v1",
                "batch_size": len(members),
                "batch_index": index,
                "serial_byte_equality_claimed": False,
                "deterministic_batch_rerun_required": True,
                "fixed_rate": True,
                "fixed_activation_parametric_scale": True,
            }
            outputs.append(
                _finalize_encoded_sqg_result(
                    prepared_source=prepared[index],
                    source_transform=transforms[index],
                    dense_h=dense_h,
                    config=config,
                    runtime=runtime,
                    dense_h_session=dense_h_session,
                    session_use_ordinal=start_use_count + index + 1,
                    reused_finalized_state=initially_finalized or index > 0,
                    lut=luts[index],
                    lut_hash=lut_hashes[index],
                    quant_args=args,
                    encoder_weight=result["weight_q"],
                    proxy_error=float(result["proxy"]),
                    raw_tensors={
                        "trellis": packed,
                        "suh": result["suh"],
                        "svh": result["svh"],
                    },
                    captured_states=states,
                    backend_execution=backend_execution,
                )
            )
        # Every member executed in KQuant even when only selected members are
        # materialized. Preserve the physical batch's session ordinal ledger.
        dense_h_session.use_count = start_use_count + len(members)
    return outputs
