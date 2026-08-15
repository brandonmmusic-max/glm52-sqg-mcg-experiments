"""Fit-only activation-parametric input anchors for GLM full-W4A8 down GEMMs.

The runtime already consumes an arbitrary, expert-private FP16 ``down.suh``
vector before its fixed H128 and K32 MXFP8 quantizer.  This module changes
only how that existing vector is calibrated.  It does not add metadata, alter
the SQG payload, fold the rotation into the weights, or change the runtime
format.

The treatment is a deliberately small 2x2 factorial panel:

* keep the preliminary down-anchor magnitude or replace it with a
  gate-square-weighted activation-RMS equalizer; and
* keep the preliminary stored signs or use one deterministic sign draw chosen
  on fit/calibration by exact post-H128 MXFP8 operand SSE.

All returned anchors are exact, finite, nonzero, contiguous CPU FP16 vectors.
They can therefore be passed to ``SharedResidualProfile.from_stored_vector``
and anchored through the existing selected-triplet or triplet-grid encoders.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from .reference import tensor_sha256


STATISTICS_SCHEMA = "glm52-w4a8-down-act-rms-statistics-v1"
SIGN_SCREEN_SCHEMA = "glm52-w4a8-down-act-sign-screen-v1"
ANCHOR_SCHEMA = "glm52-w4a8-activation-parametric-down-suh-v1"
PANEL_SCHEMA = "glm52-w4a8-activation-parametric-down-suh-panel-v1"
SIGN_NAMESPACE = "glm52/w4a8/down-suh/act-screened-rademacher/v1"
PANEL_ARM_IDS = (
    "baseline_current_signs",
    "baseline_screened_signs",
    "act_rms_current_signs",
    "act_rms_screened_signs",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_vector(value: torch.Tensor, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise TypeError(f"{label} must be a rank-one torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{label} must use a floating dtype")
    result = value.detach().float()
    if result.numel() == 0 or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must be nonempty and finite")
    return result


@dataclass
class GateSquareActRMSAccumulator:
    """Ordered accumulator for routed, gate-square-weighted activation RMS."""

    width: int
    device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("activation width must be a positive integer")
        self.device = torch.device(self.device)
        self.weighted_sumsq = torch.zeros(
            self.width, dtype=torch.float64, device=self.device
        )
        self.gate_square_sum = 0.0
        self.rows = 0

    def update(self, activation: torch.Tensor, route_gates: torch.Tensor) -> None:
        if activation.ndim != 2 or activation.shape[1] != self.width:
            raise ValueError("activation chunk shape differs from the RMS accumulator")
        if route_gates.ndim != 1 or route_gates.shape[0] != activation.shape[0]:
            raise ValueError("route gates must provide one value per activation row")
        if activation.shape[0] == 0:
            return
        values = activation.detach().to(device=self.device, dtype=torch.float32)
        gates = route_gates.detach().to(device=self.device, dtype=torch.float32)
        if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(gates).all()):
            raise ValueError("activation statistics inputs must be finite")
        if bool((gates < 0).any()):
            raise ValueError("route gates must be nonnegative")
        weights = gates.square()
        self.weighted_sumsq.add_(
            torch.sum(values.double().square() * weights.double()[:, None], dim=0)
        )
        self.gate_square_sum += float(weights.double().sum())
        self.rows += int(values.shape[0])

    def finish(
        self,
        *,
        layer: int,
        expert: int,
        documents: int,
        source_binding: Mapping[str, Any],
        upstream_binding: Mapping[str, Any],
        subfold_binding: Mapping[str, Any],
    ) -> "ActivationRMSStatistics":
        if self.rows <= 0 or not math.isfinite(self.gate_square_sum) or self.gate_square_sum <= 0:
            raise ValueError("activation RMS accumulator has no positive routed mass")
        if documents <= 0:
            raise ValueError("activation RMS statistics require at least one document")
        sumsq = self.weighted_sumsq.detach().cpu().contiguous()
        rms = torch.sqrt(sumsq / self.gate_square_sum).float().contiguous()
        positive = rms[rms > 0]
        if positive.numel() == 0:
            raise ValueError("all activation channels have zero RMS")
        floor = max(
            float(positive.double().median()) * 1.0e-6,
            float(torch.finfo(torch.float32).tiny),
        )
        rms = rms.clamp_min(floor).contiguous()
        evidence: dict[str, Any] = {
            "schema": STATISTICS_SCHEMA,
            "layer": int(layer),
            "expert": int(expert),
            "role": "fit",
            "subfold": "calibration",
            "rows": self.rows,
            "documents": int(documents),
            "gate_square_sum": self.gate_square_sum,
            "construction": (
                "exact_selected_full_w4a8_gate_up_candidate_swiglu_fp16_"
                "gate_square_weighted_channel_rms_v1"
            ),
            "weighted_sumsq_dtype": "float64",
            "weighted_sumsq_sha256": tensor_sha256(sumsq),
            "weighted_rms_dtype": "float32",
            "weighted_rms_sha256": tensor_sha256(rms),
            "zero_rms_floor": floor,
            "source_binding": dict(source_binding),
            "upstream_binding": dict(upstream_binding),
            "subfold_binding": dict(subfold_binding),
            "selection_rows_used": False,
            "holdout_rows_used": False,
        }
        evidence_id = canonical_sha256(evidence)
        return ActivationRMSStatistics(
            weighted_sumsq=sumsq,
            weighted_rms=rms,
            gate_square_sum=self.gate_square_sum,
            rows=self.rows,
            documents=int(documents),
            evidence=MappingProxyType({**evidence, "evidence_id": evidence_id}),
        )


@dataclass(frozen=True)
class ActivationRMSStatistics:
    weighted_sumsq: torch.Tensor
    weighted_rms: torch.Tensor
    gate_square_sum: float
    rows: int
    documents: int
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        sumsq = _require_vector(self.weighted_sumsq, "weighted sumsq")
        rms = _require_vector(self.weighted_rms, "weighted RMS")
        if sumsq.shape != rms.shape or bool((sumsq < 0).any()) or bool((rms <= 0).any()):
            raise ValueError("activation RMS tensors are inconsistent")
        if self.weighted_sumsq.device.type != "cpu" or self.weighted_sumsq.dtype != torch.float64:
            raise TypeError("weighted sumsq must be CPU float64")
        if self.weighted_rms.device.type != "cpu" or self.weighted_rms.dtype != torch.float32:
            raise TypeError("weighted RMS must be CPU float32")
        if self.evidence.get("schema") != STATISTICS_SCHEMA:
            raise ValueError("activation RMS evidence schema differs")
        body = dict(self.evidence)
        evidence_id = body.pop("evidence_id", None)
        if evidence_id != canonical_sha256(body):
            raise ValueError("activation RMS evidence ID differs")


def _bounded_unit_geomean_multiplier(
    raw_multiplier: torch.Tensor,
    *,
    lower: float,
    upper: float,
    iterations: int = 96,
) -> torch.Tensor:
    """Clamp a positive multiplier while preserving geometric mean one.

    A scalar shift is solved in log space.  This avoids the usual
    clamp-then-renormalize bug, where the renormalization can move values back
    outside the declared bounds.
    """

    raw = _require_vector(raw_multiplier, "raw RMS multiplier").double()
    if bool((raw <= 0).any()):
        raise ValueError("raw RMS multiplier must be positive")
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or not 0 < lower <= 1.0 <= upper
        or lower >= upper
    ):
        raise ValueError("multiplier bounds must straddle one")
    if iterations < 32:
        raise ValueError("bounded normalization requires at least 32 iterations")
    logs = raw.log()
    low_bound = math.log(lower)
    high_bound = math.log(upper)
    low = low_bound - float(logs.max()) - 1.0
    high = high_bound - float(logs.min()) + 1.0
    for _ in range(iterations):
        middle = (low + high) * 0.5
        mean = float(torch.clamp(logs + middle, low_bound, high_bound).mean())
        if mean < 0.0:
            low = middle
        else:
            high = middle
    shift = (low + high) * 0.5
    result = torch.exp(torch.clamp(logs + shift, low_bound, high_bound)).float()
    if float(result.min()) < lower - 2e-7 or float(result.max()) > upper + 2e-7:
        raise RuntimeError("bounded RMS normalization escaped its declared limits")
    return result.contiguous()


def act_rms_equalized_magnitude(
    baseline_stored_suh: torch.Tensor,
    weighted_rms: torch.Tensor,
    *,
    relative_min: float = 0.5,
    relative_max: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return a deterministic RMS-equalized magnitude at fixed global scale."""

    baseline = _require_vector(baseline_stored_suh, "baseline stored suh").cpu()
    rms = _require_vector(weighted_rms, "weighted RMS").cpu()
    if baseline.shape != rms.shape or bool((baseline == 0).any()) or bool((rms <= 0).any()):
        raise ValueError("baseline suh and weighted RMS are inconsistent")
    magnitude = baseline.abs().double()
    rms64 = rms.double()
    target_log = magnitude.log().mean() + rms64.log().mean() - rms64.log()
    raw_multiplier = torch.exp(target_log - magnitude.log())
    multiplier = _bounded_unit_geomean_multiplier(
        raw_multiplier.float(), lower=relative_min, upper=relative_max
    ).double()
    result = (magnitude * multiplier).float().contiguous()
    diagnostics = {
        "relative_min_bound": float(relative_min),
        "relative_max_bound": float(relative_max),
        "relative_multiplier_min": float(multiplier.min()),
        "relative_multiplier_max": float(multiplier.max()),
        "relative_multiplier_geomean": float(multiplier.log().mean().exp()),
        "baseline_magnitude_geomean": float(magnitude.log().mean().exp()),
        "candidate_magnitude_geomean": float(result.double().log().mean().exp()),
    }
    return result, diagnostics


def deterministic_stored_sign_draw(
    width: int,
    *,
    layer: int,
    expert: int,
    draw: int,
    namespace: str = SIGN_NAMESPACE,
) -> torch.Tensor:
    """Generate a cross-platform Rademacher draw without an RNG dependency."""

    for label, value in (("width", width), ("layer", layer), ("expert", expert), ("draw", draw)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a nonnegative integer")
    if width == 0 or not namespace:
        raise ValueError("sign draw width and namespace must be nonempty")
    material = f"{namespace}/{layer}/{expert}/{draw}".encode("ascii")
    payload = hashlib.shake_256(material).digest(width)
    signs = torch.tensor(
        [1.0 if byte & 1 else -1.0 for byte in payload], dtype=torch.float32
    )
    return signs.contiguous()


@dataclass
class SignScreenAccumulator:
    """Accumulate exact post-H/QDQ operand loss for a fixed sign-draw pool."""

    draws: Sequence[int]

    def __post_init__(self) -> None:
        values = tuple(int(value) for value in self.draws)
        if not values or len(set(values)) != len(values) or any(value < 0 for value in values):
            raise ValueError("sign-screen draws must be unique nonnegative integers")
        self.draws = values
        self.sse = {draw: 0.0 for draw in values}
        self.source_energy = {draw: 0.0 for draw in values}
        self.rows = 0
        self.gate_square_sum = 0.0

    def update(
        self,
        *,
        draw: int,
        source_operand: torch.Tensor,
        quantized_operand: torch.Tensor,
        route_gates: torch.Tensor,
    ) -> None:
        if draw not in self.sse:
            raise ValueError("sign-screen update used an undeclared draw")
        if source_operand.ndim != 2 or quantized_operand.shape != source_operand.shape:
            raise ValueError("sign-screen operands must be aligned rank-two tensors")
        if route_gates.ndim != 1 or route_gates.shape[0] != source_operand.shape[0]:
            raise ValueError("sign-screen route gates do not match operand rows")
        source = source_operand.float()
        quantized = quantized_operand.float()
        gates = route_gates.float()
        if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(quantized).all()):
            raise ValueError("sign-screen operands must be finite")
        if not bool(torch.isfinite(gates).all()) or bool((gates < 0).any()):
            raise ValueError("sign-screen route gates must be finite and nonnegative")
        importance = gates.square()[:, None]
        self.sse[draw] += float(
            torch.sum((quantized - source).double().square() * importance.double())
        )
        self.source_energy[draw] += float(
            torch.sum(source.double().square() * importance.double())
        )

    def finish_chunk(self, route_gates: torch.Tensor) -> None:
        if route_gates.ndim != 1:
            raise ValueError("sign-screen chunk gates must be a vector")
        gates = route_gates.float()
        if not bool(torch.isfinite(gates).all()) or bool((gates < 0).any()):
            raise ValueError("sign-screen chunk gates must be finite and nonnegative")
        self.rows += int(gates.numel())
        self.gate_square_sum += float(gates.double().square().sum())

    def finish(
        self,
        *,
        layer: int,
        expert: int,
        statistics_evidence_id: str,
        magnitude_sha256: str,
    ) -> dict[str, Any]:
        if self.rows <= 0 or self.gate_square_sum <= 0:
            raise ValueError("sign screen has no positive routed mass")
        scores = []
        for draw in self.draws:
            energy = self.source_energy[draw]
            sse = self.sse[draw]
            if not math.isfinite(energy) or energy <= 0 or not math.isfinite(sse) or sse < 0:
                raise ValueError("sign screen accumulated an invalid score")
            scores.append(
                {
                    "draw": draw,
                    "weighted_operand_sse": sse,
                    "weighted_operand_energy": energy,
                    "weighted_operand_nmse": sse / energy,
                }
            )
        winner = min(scores, key=lambda item: (item["weighted_operand_nmse"], item["draw"]))
        evidence: dict[str, Any] = {
            "schema": SIGN_SCREEN_SCHEMA,
            "layer": int(layer),
            "expert": int(expert),
            "role": "fit",
            "subfold": "calibration",
            "objective": "gate_square_weighted_exact_post_h128_mxfp8_operand_nmse",
            "draw_namespace": SIGN_NAMESPACE,
            "draws": list(self.draws),
            "scores": scores,
            "selected_draw": int(winner["draw"]),
            "rows": self.rows,
            "gate_square_sum": self.gate_square_sum,
            "activation_statistics_evidence_id": statistics_evidence_id,
            "screening_magnitude_sha256": magnitude_sha256,
            "selection_rows_used": False,
            "holdout_rows_used": False,
        }
        evidence["evidence_id"] = canonical_sha256(evidence)
        return evidence


@dataclass(frozen=True)
class ActivationParametricDownAnchor:
    arm_id: str
    stored_suh: torch.Tensor
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.arm_id not in PANEL_ARM_IDS:
            raise ValueError("unknown activation-parametric panel arm")
        vector = self.stored_suh
        if (
            vector.device.type != "cpu"
            or vector.dtype != torch.float16
            or vector.ndim != 1
            or not vector.is_contiguous()
            or not bool(torch.isfinite(vector).all())
            or bool((vector == 0).any())
        ):
            raise ValueError("stored down suh must be exact finite nonzero CPU FP16")
        if self.evidence.get("schema") != ANCHOR_SCHEMA:
            raise ValueError("activation-parametric anchor evidence schema differs")
        body = dict(self.evidence)
        evidence_id = body.pop("evidence_id", None)
        if evidence_id != canonical_sha256(body):
            raise ValueError("activation-parametric anchor evidence ID differs")
        if self.evidence.get("stored_suh_sha256") != tensor_sha256(vector):
            raise ValueError("activation-parametric anchor payload hash differs")

    @property
    def evidence_id(self) -> str:
        return str(self.evidence["evidence_id"])

    def profile(self, *, layer: int, expert: int, bits: int) -> Any:
        """Return the existing codec profile; callers may realize it on CUDA."""

        from .codec import SharedResidualProfile

        return SharedResidualProfile.from_stored_vector(
            self.stored_suh,
            side="input",
            profile_id=(
                f"layer-{layer:03d}/expert-{expert:03d}/down-k{bits}/"
                f"{self.arm_id}/{self.evidence_id[:16]}"
            ),
            derivation=(
                "fit_calibration_exact_full_w4a8_activation_parametric_down_suh_v1"
            ),
        )


def build_factorial_anchors(
    baseline_stored_suh: torch.Tensor,
    statistics: ActivationRMSStatistics,
    sign_screen_evidence: Mapping[str, Any],
    *,
    layer: int,
    expert: int,
    relative_min: float = 0.5,
    relative_max: float = 2.0,
) -> dict[str, ActivationParametricDownAnchor]:
    """Construct the exact FP16 2x2 panel from frozen fit-only evidence."""

    baseline = baseline_stored_suh.detach().cpu()
    if (
        baseline.dtype != torch.float16
        or baseline.ndim != 1
        or not baseline.is_contiguous()
        or not bool(torch.isfinite(baseline).all())
        or bool((baseline == 0).any())
    ):
        raise ValueError("baseline stored suh must be exact finite nonzero CPU FP16")
    if baseline.shape != statistics.weighted_rms.shape:
        raise ValueError("baseline stored suh width differs from activation RMS")
    screen = dict(sign_screen_evidence)
    body = dict(screen)
    screen_id = body.pop("evidence_id", None)
    if screen.get("schema") != SIGN_SCREEN_SCHEMA or screen_id != canonical_sha256(body):
        raise ValueError("sign-screen evidence is not sealed")
    if screen.get("activation_statistics_evidence_id") != statistics.evidence["evidence_id"]:
        raise ValueError("sign-screen evidence uses different activation statistics")
    upstream = statistics.evidence.get("upstream_binding")
    rates = upstream.get("rates") if isinstance(upstream, Mapping) else None
    if (
        not isinstance(rates, Mapping)
        or set(rates) != {"gate_proj", "up_proj", "down_proj"}
        or any(int(rates[name]) not in (3, 4) for name in rates)
    ):
        raise ValueError(
            "activation statistics must bind one exact selected K3/K4 triplet"
        )
    rate_triplet = {
        name: int(rates[name]) for name in ("gate_proj", "up_proj", "down_proj")
    }
    draw = int(screen["selected_draw"])
    current_signs = baseline.float().sign()
    screened_signs = deterministic_stored_sign_draw(
        baseline.numel(), layer=layer, expert=expert, draw=draw
    )
    baseline_magnitude = baseline.float().abs().contiguous()
    equalized_magnitude, normalization = act_rms_equalized_magnitude(
        baseline,
        statistics.weighted_rms,
        relative_min=relative_min,
        relative_max=relative_max,
    )
    if screen.get("screening_magnitude_sha256") != tensor_sha256(
        equalized_magnitude.half().contiguous()
    ):
        raise ValueError("sign screen was not run with the frozen RMS magnitude")

    specifications = {
        "baseline_current_signs": ("baseline", "current", baseline_magnitude, current_signs),
        "baseline_screened_signs": ("baseline", "screened", baseline_magnitude, screened_signs),
        "act_rms_current_signs": ("act_rms_equalized", "current", equalized_magnitude, current_signs),
        "act_rms_screened_signs": (
            "act_rms_equalized",
            "screened",
            equalized_magnitude,
            screened_signs,
        ),
    }
    result: dict[str, ActivationParametricDownAnchor] = {}
    baseline_sha = tensor_sha256(baseline)
    for arm_id in PANEL_ARM_IDS:
        magnitude_policy, sign_policy, magnitude, signs = specifications[arm_id]
        stored = (magnitude * signs).half().cpu().contiguous()
        if arm_id == "baseline_current_signs" and not torch.equal(stored, baseline):
            raise RuntimeError("baseline/current arm failed byte identity")
        actual_ratio = stored.float().abs() / baseline.float().abs()
        evidence: dict[str, Any] = {
            "schema": ANCHOR_SCHEMA,
            "arm_id": arm_id,
            "layer": int(layer),
            "expert": int(expert),
            "role": "fit",
            "subfold": "calibration",
            "magnitude_policy": magnitude_policy,
            "sign_policy": sign_policy,
            "baseline_stored_suh_sha256": baseline_sha,
            "stored_suh_sha256": tensor_sha256(stored),
            "activation_statistics_evidence_id": statistics.evidence["evidence_id"],
            "upstream_rate_pair": {
                "gate_proj": rate_triplet["gate_proj"],
                "up_proj": rate_triplet["up_proj"],
            },
            "down_rate": rate_triplet["down_proj"],
            "exact_selected_triplet": rate_triplet,
            "sign_screen_evidence_id": screen_id if sign_policy == "screened" else None,
            "selected_sign_draw": draw if sign_policy == "screened" else None,
            "normalization": normalization if magnitude_policy == "act_rms_equalized" else None,
            "realized_fp16_relative_multiplier": {
                "min": float(actual_ratio.min()),
                "max": float(actual_ratio.max()),
                "geomean": float(actual_ratio.double().log().mean().exp()),
            },
            "runtime_format_changed": False,
            "hadamard_block": 128,
            "mxfp8_block": 32,
            "selection_rows_used": False,
            "holdout_rows_used": False,
        }
        evidence_id = canonical_sha256(evidence)
        result[arm_id] = ActivationParametricDownAnchor(
            arm_id=arm_id,
            stored_suh=stored,
            evidence=MappingProxyType({**evidence, "evidence_id": evidence_id}),
        )
    return result


@dataclass(frozen=True)
class DownAnchorOverride:
    """Minimal duck-typed adapter accepted by the existing down fitters."""

    suh: torch.Tensor
    evidence_id: str
    arm_id: str


def selected_anchor_adapter(anchor: ActivationParametricDownAnchor) -> DownAnchorOverride:
    """Adapter for ``_fit_encode_selected_down(preliminary_down=...)``."""

    return DownAnchorOverride(
        suh=anchor.stored_suh,
        evidence_id=anchor.evidence_id,
        arm_id=anchor.arm_id,
    )


def triplet_anchor_adapters(
    anchors_by_triplet: Mapping[
        tuple[int, int, int], ActivationParametricDownAnchor
    ],
) -> dict[tuple[int, int, int], DownAnchorOverride]:
    """Return adapters without collapsing exact upstream-rate-pair geometry.

    The current shared triplet-grid fitter keys preliminary anchors only by
    down rate.  That API is insufficient for activation-parametric anchors:
    K3/K3, K3/K4, K4/K3, and K4/K4 create different exact SwiGLU
    activations.  Callers must retain all eight ``(gate, up, down)`` keys and
    update the shared grid orchestration before using this adapter there.
    """

    required = {
        (gate_bits, up_bits, down_bits)
        for gate_bits in (3, 4)
        for up_bits in (3, 4)
        for down_bits in (3, 4)
    }
    if set(anchors_by_triplet) != required:
        raise ValueError(
            "triplet anchor integration requires all eight exact rate triplets"
        )
    result: dict[tuple[int, int, int], DownAnchorOverride] = {}
    for triplet in sorted(required):
        anchor = anchors_by_triplet[triplet]
        bound = anchor.evidence.get("exact_selected_triplet")
        expected = {
            "gate_proj": triplet[0],
            "up_proj": triplet[1],
            "down_proj": triplet[2],
        }
        if bound != expected:
            raise ValueError(
                f"activation-parametric anchor binding differs for triplet {triplet}"
            )
        result[triplet] = selected_anchor_adapter(anchor)
    return result


def panel_contract_fragment(*, draw_count: int, relative_min: float, relative_max: float) -> dict[str, Any]:
    """Stable fragment for preregistration and full-encode source binding."""

    if draw_count <= 0:
        raise ValueError("draw count must be positive")
    return {
        "schema": PANEL_SCHEMA,
        "arms": list(PANEL_ARM_IDS),
        "factorial": {
            "magnitude": ["baseline", "act_rms_equalized"],
            "signs": ["current", "fit_calibration_screened_deterministic_draw"],
        },
        "rms_relative_bounds": [float(relative_min), float(relative_max)],
        "sign_draw_namespace": SIGN_NAMESPACE,
        "sign_draws": list(range(draw_count)),
        "sign_screen_objective": (
            "gate_square_weighted_exact_post_h128_mxfp8_operand_nmse"
        ),
        "choice_split": "fit/allocation",
        "report_only_splits": ["selection", "holdout"],
        "runtime_format_changed": False,
        "rate_policy": "preserve_selected_independent_per_tensor_k3_k4",
    }
