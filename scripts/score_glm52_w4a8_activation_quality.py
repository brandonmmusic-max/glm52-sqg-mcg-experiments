#!/usr/bin/env python3
"""Test 8b: isolate GLM-5.2 SQG activation quantization at layer 77.

This is a functional quality oracle, not a speed benchmark and not an
end-to-end logit-KLD run.  It decodes the selected SQG tensor labels directly
onto their native finite-E4M3 grid, keeps the stored Hadamard transforms and
``suh``/``svh`` scales on the activation/epilogue sides, and compares four
otherwise-identical arms:

* ``sqg_a16``: no activation E4M3 roundtrip;
* ``sqg_h_a8``: only the gate/up input is MXFP8-quantized;
* ``sqg_act_a8``: only the SwiGLU/down input is MXFP8-quantized; and
* ``sqg_w4a8``: both activation points are MXFP8-quantized.

MXFP8 uses one UE8M0 power-of-two scale per consecutive K32 block and a finite
E4M3 payload.  FP32 accumulation and the current FP16 inter-GEMM boundaries are
modelled explicitly.  Gate/up and down stay in the same expert-private
permuted intermediate basis through SwiGLU; no permutation or Hadamard is
folded into the native E4M3 weight labels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT / "kquant", PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.compare_hessian_weighted_nmse import (  # noqa: E402
    _load_hidden_chunk,
    _read_source_triplet,
)
from scripts.compare_raw_encoded_nmse import (  # noqa: E402
    DEFAULT_MCG_ROOT,
    _load_sqg_manifests,
)
from scripts.score_signed_top8_blends import (  # noqa: E402
    _reduce_rows,
    _sha256_array,
    tail_comparison,
)
from src.glm52_fresh_sqg.reference import (  # noqa: E402
    decode_regularized_states,
    normalized_hadamard,
    payload_sha256,
    tensor_sha256,
    unpack_trellis_states,
)


LAYER = 77
HIDDEN = 6144
INTERMEDIATE = 2048
TOPK = 8
HADAMARD_BLOCK = 128
MXFP8_BLOCK = 32
E4M3_MAX = 448.0
SQG_MARKER = 0x53514731
ROLES = {"selection": 1, "holdout": 2}
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ARMS = ("sqg_a16", "sqg_h_a8", "sqg_act_a8", "sqg_w4a8")
STAGES = ("gate", "up", "act", "expert_output")

DEFAULT_CANDIDATE_ROOT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-contig-late-final-a025-r1"
)
DEFAULT_PERMUTATION_ROOT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-contig-late-a025-r1"
)
DEFAULT_CAPTURE_ROOT = Path(
    "/home/brandonmusic/KLC_CAPTURE_RUNS/contig-late-capture-r1/contig-late-r1"
)
DEFAULT_BF16_ROOT = PROJECT_ROOT / "bf16_contiguous_late"
DEFAULT_BF16_MANIFEST = PROJECT_ROOT / "evidence/contiguous_late_bf16_manifest_r1.json"
DEFAULT_SELECTION_RECEIPT = (
    PROJECT_ROOT / "results/contiguous_late_h13_blend_selection_r1.json"
)
DEFAULT_HOLDOUT_RECEIPT = (
    PROJECT_ROOT / "results/contiguous_late_h13_blend_holdout_r1.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/glm52_w4a8_activation_quality_l077_r1.json"

EXECUTION_ORDER = (
    "capture BF16 h",
    "FP16 h boundary",
    "multiply stored gate/up suh and round FP16",
    "blockwise normalized H128",
    "A16: round FP16; A8: direct per-K32 UE8M0/E4M3 QDQ",
    "native finite-E4M3 SQG gate/up labels",
    "FP32 gate/up GEMM accumulation",
    "FP16 gate/up output boundary",
    "blockwise normalized H128 then stored gate/up svh",
    "SwiGLU in the unchanged expert-private permuted basis",
    "multiply stored down suh",
    "blockwise normalized H128",
    "A16: round FP16; A8: direct per-K32 UE8M0/E4M3 QDQ",
    "native finite-E4M3 SQG down labels",
    "FP32 down GEMM accumulation",
    "FP16 down output boundary",
    "blockwise normalized H128 then stored down svh",
    "captured router-probability signed top-8 FP32 sum",
)


def _sha256_file(path: Path, *, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fp16_boundary(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float16).to(torch.float32)


def _block_hadamard_right(
    value: torch.Tensor,
    hadamard: torch.Tensor,
) -> torch.Tensor:
    """Right-multiply independent blocks along the final dimension by H."""

    if value.ndim != 2:
        raise ValueError("block Hadamard input must be rank two")
    block = int(hadamard.shape[0])
    if tuple(hadamard.shape) != (block, block) or value.shape[1] % block:
        raise ValueError("Hadamard shape does not divide the activation width")
    rows, cols = value.shape
    blocks = value.float().reshape(rows, cols // block, block)
    return torch.matmul(blocks, hadamard.float()).reshape(rows, cols)


def pow2_ceil_ue8m0_torch(
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bit-exact Torch replica of the B12X UE8M0 ceil intrinsic."""

    if not scale.dtype.is_floating_point:
        raise TypeError("UE8M0 source scale must be floating point")
    values = scale.to(torch.float32).contiguous()
    if bool((values < 0).any()) or not bool(torch.isfinite(values).all()):
        raise ValueError("UE8M0 source scale must be finite and nonnegative")
    bits = values.view(torch.int32)
    mantissa = bits & 0x007FFFFF
    bumped = torch.where(
        mantissa != 0,
        (bits + 0x00800000) & 0x7F800000,
        bits,
    )
    rounded = bumped.view(torch.float32)
    byte = ((bumped >> 23) & 0xFF).to(torch.uint8)
    return rounded, byte


def _ue8m0_inverse_scale(byte: torch.Tensor) -> torch.Tensor:
    inverse_bits = (254 - byte.to(torch.int32)).clamp(min=0) << 23
    inverse = inverse_bits.view(torch.float32)
    return torch.where(byte == 0, torch.zeros_like(inverse), inverse)


@dataclass(frozen=True)
class QDQObservation:
    payload: torch.Tensor
    scale_bytes: torch.Tensor
    block_amax: torch.Tensor
    preclamp_overflow: torch.Tensor


def quant_dequant_mxfp8(
    value: torch.Tensor,
    *,
    block_size: int = MXFP8_BLOCK,
) -> tuple[torch.Tensor, QDQObservation]:
    """Roundtrip through finite E4M3 with one UE8M0 scale per K block."""

    if value.ndim != 2 or value.shape[1] % block_size:
        raise ValueError("MXFP8 input must be rank two with K divisible by block size")
    if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
        raise ValueError("MXFP8 input must be finite floating point")
    rows, cols = value.shape
    blocks = value.float().reshape(rows, cols // block_size, block_size)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    rounded, byte = pow2_ceil_ue8m0_torch(block_amax / E4M3_MAX)
    inverse = _ue8m0_inverse_scale(byte)
    normalized = blocks * inverse
    overflow = normalized.abs() > E4M3_MAX
    payload = normalized.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    output = payload.float().mul(rounded).reshape(rows, cols)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("MXFP8 roundtrip produced a non-finite value")
    return output, QDQObservation(
        payload=payload,
        scale_bytes=byte.squeeze(-1),
        block_amax=block_amax.squeeze(-1),
        preclamp_overflow=overflow,
    )


def _strided_sample(value: torch.Tensor, limit: int = 1024) -> np.ndarray:
    flat = value.detach().float().abs().flatten()
    if flat.numel() == 0:
        return np.empty(0, dtype=np.float32)
    stride = max(1, math.ceil(flat.numel() / limit))
    return flat[::stride][:limit].cpu().numpy().astype(np.float32, copy=False)


@dataclass
class OperandStats:
    elements: int = 0
    blocks: int = 0
    source_sse: float = 0.0
    quantization_sse: float = 0.0
    source_zeros: int = 0
    quantized_zeros: int = 0
    exact_values: int = 0
    saturated_payload_values: int = 0
    preclamp_overflow_values: int = 0
    zero_scale_blocks: int = 0
    min_nonzero_scale_byte: int = 255
    max_scale_byte: int = 0
    max_abs_source: float = 0.0
    max_abs_error: float = 0.0
    source_abs_samples: list[np.ndarray] = field(default_factory=list)
    error_abs_samples: list[np.ndarray] = field(default_factory=list)
    block_amax_samples: list[np.ndarray] = field(default_factory=list)

    def update(
        self,
        source: torch.Tensor,
        quantized: torch.Tensor,
        observation: QDQObservation,
    ) -> None:
        if source.shape != quantized.shape:
            raise ValueError("operand source and quantized shapes differ")
        difference = quantized.float() - source.float()
        self.elements += int(source.numel())
        self.blocks += int(observation.scale_bytes.numel())
        self.source_sse += float(source.double().square().sum().item())
        self.quantization_sse += float(difference.double().square().sum().item())
        self.source_zeros += int((source == 0).sum().item())
        self.quantized_zeros += int((quantized == 0).sum().item())
        self.exact_values += int((difference == 0).sum().item())
        self.saturated_payload_values += int(
            (observation.payload.float().abs() == E4M3_MAX).sum().item()
        )
        self.preclamp_overflow_values += int(observation.preclamp_overflow.sum().item())
        self.zero_scale_blocks += int((observation.scale_bytes == 0).sum().item())
        nonzero = observation.scale_bytes[observation.scale_bytes != 0]
        if nonzero.numel():
            self.min_nonzero_scale_byte = min(
                self.min_nonzero_scale_byte, int(nonzero.min().item())
            )
        self.max_scale_byte = max(
            self.max_scale_byte, int(observation.scale_bytes.max().item())
        )
        self.max_abs_source = max(self.max_abs_source, float(source.abs().max().item()))
        self.max_abs_error = max(self.max_abs_error, float(difference.abs().max().item()))
        self.source_abs_samples.append(_strided_sample(source))
        self.error_abs_samples.append(_strided_sample(difference))
        self.block_amax_samples.append(_strided_sample(observation.block_amax))

    @staticmethod
    def _sample_summary(chunks: list[np.ndarray]) -> dict[str, Any]:
        sample = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
        if sample.size == 0:
            return {"samples": 0}
        return {
            "samples": int(sample.size),
            "p50": float(np.quantile(sample, 0.50)),
            "p90": float(np.quantile(sample, 0.90)),
            "p99": float(np.quantile(sample, 0.99)),
            "p99_9": float(np.quantile(sample, 0.999)),
            "max_sampled": float(sample.max()),
        }

    def finish(self) -> dict[str, Any]:
        denominator = max(self.source_sse, np.finfo(np.float64).tiny)
        return {
            "elements": self.elements,
            "blocks_k32": self.blocks,
            "source_energy": self.source_sse,
            "quantization_sse": self.quantization_sse,
            "quantization_nmse": self.quantization_sse / denominator,
            "source_zero_fraction": self.source_zeros / max(1, self.elements),
            "quantized_zero_fraction": self.quantized_zeros / max(1, self.elements),
            "exact_value_fraction": self.exact_values / max(1, self.elements),
            "saturated_payload_fraction": self.saturated_payload_values
            / max(1, self.elements),
            "preclamp_overflow_values": self.preclamp_overflow_values,
            "zero_scale_blocks": self.zero_scale_blocks,
            "min_nonzero_ue8m0_byte": (
                None if self.min_nonzero_scale_byte == 255 else self.min_nonzero_scale_byte
            ),
            "max_ue8m0_byte": self.max_scale_byte,
            "max_abs_source": self.max_abs_source,
            "max_abs_error": self.max_abs_error,
            "sample_contract": (
                "deterministic evenly-strided absolute-value sample, at most "
                "1024 values per routed expert chunk; energy/error totals are exact"
            ),
            "source_abs_sample": self._sample_summary(self.source_abs_samples),
            "error_abs_sample": self._sample_summary(self.error_abs_samples),
            "block_amax_sample": self._sample_summary(self.block_amax_samples),
        }


@dataclass
class MagnitudeStats:
    """Exact moments plus an explicitly labelled strided magnitude sample."""

    elements: int = 0
    energy: float = 0.0
    absolute_sum: float = 0.0
    zeros: int = 0
    max_abs: float = 0.0
    abs_samples: list[np.ndarray] = field(default_factory=list)

    def update(self, value: torch.Tensor) -> None:
        if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
            raise ValueError("activation magnitude input must be finite floating point")
        values = value.float()
        self.elements += int(values.numel())
        self.energy += float(values.double().square().sum().item())
        self.absolute_sum += float(values.double().abs().sum().item())
        self.zeros += int((values == 0).sum().item())
        self.max_abs = max(self.max_abs, float(values.abs().max().item()))
        self.abs_samples.append(_strided_sample(values))

    def finish(self) -> dict[str, Any]:
        count = max(1, self.elements)
        return {
            "exact": {
                "elements": self.elements,
                "energy": self.energy,
                "rms": math.sqrt(self.energy / count),
                "mean_abs": self.absolute_sum / count,
                "zero_fraction": self.zeros / count,
                "max_abs": self.max_abs,
            },
            "strided_abs_distribution": OperandStats._sample_summary(
                self.abs_samples
            ),
            "sample_contract": (
                "deterministic evenly-strided absolute-value sample, at most "
                "1024 values per routed expert chunk; exact moments and maxima "
                "use every evaluated value"
            ),
        }


def gate_up_transform_points(
    hidden: torch.Tensor,
    suh: torch.Tensor,
    hadamard: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Expose h before scaling, after suh/pre-H, and after H128."""

    if suh.ndim != 1 or suh.numel() != hidden.shape[1]:
        raise ValueError("gate/up suh does not match hidden width")
    raw = hidden.float()
    source_fp16 = _fp16_boundary(raw)
    pre_h = _fp16_boundary(source_fp16 * suh.float())
    post_h = _block_hadamard_right(pre_h, hadamard)
    return {"raw": raw, "after_suh_pre_h": pre_h, "post_h_qdq_input": post_h}


def down_transform_points(
    activation: torch.Tensor,
    suh: torch.Tensor,
    hadamard: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Expose SwiGLU act before scaling, after suh/pre-H, and after H128."""

    if suh.ndim != 1 or suh.numel() != activation.shape[1]:
        raise ValueError("down suh does not match intermediate width")
    raw = activation.float()
    pre_h = raw * suh.float()
    post_h = _block_hadamard_right(pre_h, hadamard)
    return {"raw": raw, "after_suh_pre_h": pre_h, "post_h_qdq_input": post_h}


def prepare_gate_up_operand(
    hidden: torch.Tensor,
    suh: torch.Tensor,
    hadamard: torch.Tensor,
    *,
    quantize_a8: bool,
) -> tuple[torch.Tensor, QDQObservation | None, torch.Tensor]:
    """Apply the current full-rotation FC1 operand order."""

    points = gate_up_transform_points(hidden, suh, hadamard)
    rotated = points["post_h_qdq_input"]
    if not quantize_a8:
        return _fp16_boundary(rotated), None, rotated
    quantized, observation = quant_dequant_mxfp8(rotated)
    return quantized, observation, rotated


def prepare_down_operand(
    activation: torch.Tensor,
    suh: torch.Tensor,
    hadamard: torch.Tensor,
    *,
    quantize_a8: bool,
) -> tuple[torch.Tensor, QDQObservation | None, torch.Tensor]:
    """Apply down suh/H128 after SwiGLU, then the optional MXFP8 roundtrip."""

    points = down_transform_points(activation, suh, hadamard)
    rotated = points["post_h_qdq_input"]
    if not quantize_a8:
        return _fp16_boundary(rotated), None, rotated
    quantized, observation = quant_dequant_mxfp8(rotated)
    return quantized, observation, rotated


def native_label_gemm(
    activation: torch.Tensor,
    native_e4m3_weight: torch.Tensor,
) -> torch.Tensor:
    """FP32-accumulated GEMM followed by the serving path's FP16 boundary."""

    if activation.ndim != 2 or native_e4m3_weight.ndim != 2:
        raise ValueError("native-label GEMM requires rank-two operands")
    if activation.shape[1] != native_e4m3_weight.shape[0]:
        raise ValueError("native-label GEMM K dimensions differ")
    return _fp16_boundary(torch.matmul(activation.float(), native_e4m3_weight.float()))


def apply_output_transform(
    accumulated_fp16: torch.Tensor,
    svh: torch.Tensor,
    hadamard: torch.Tensor,
) -> torch.Tensor:
    if svh.ndim != 1 or svh.numel() != accumulated_fp16.shape[1]:
        raise ValueError("output svh does not match GEMM output width")
    return _block_hadamard_right(accumulated_fp16, hadamard) * svh.float()


@dataclass(frozen=True)
class NativeProjection:
    weight: torch.Tensor
    suh: torch.Tensor
    svh: torch.Tensor
    bits: int


def evaluate_sqg_arms(
    hidden: torch.Tensor,
    projections: Mapping[str, NativeProjection],
    hadamard: torch.Tensor,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, tuple[torch.Tensor, torch.Tensor, QDQObservation]],
    dict[str, dict[str, torch.Tensor]],
]:
    """Evaluate all four arms without changing weights, rates, or transforms."""

    if set(projections) != set(PROJECTIONS):
        raise ValueError("exactly gate/up/down native projections are required")
    gate = projections["gate_proj"]
    up = projections["up_proj"]
    down = projections["down_proj"]
    if not torch.equal(gate.suh, up.suh):
        raise RuntimeError("gate/up input suh must be byte-identical in this topology")

    magnitude_paths = {"h": gate_up_transform_points(hidden, gate.suh, hadamard)}
    h_source = magnitude_paths["h"]["post_h_qdq_input"]
    h_a16 = _fp16_boundary(h_source)
    h_a8, h_obs = quant_dequant_mxfp8(h_source)
    if h_obs is None:
        raise AssertionError("h-A8 observation was not produced")

    upstream: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for name, operand in (("a16", h_a16), ("h_a8", h_a8)):
        gate_acc = native_label_gemm(operand, gate.weight)
        up_acc = native_label_gemm(operand, up.weight)
        gate_out = apply_output_transform(gate_acc, gate.svh, hadamard)
        up_out = apply_output_transform(up_acc, up.svh, hadamard)
        activation = F.silu(gate_out.float()) * up_out.float()
        upstream[name] = (gate_out, up_out, activation)

    stages: dict[str, dict[str, torch.Tensor]] = {}
    qdq: dict[str, tuple[torch.Tensor, torch.Tensor, QDQObservation]] = {
        "h": (h_source, h_a8, h_obs)
    }
    arm_contract = {
        "sqg_a16": ("a16", False),
        "sqg_h_a8": ("h_a8", False),
        "sqg_act_a8": ("a16", True),
        "sqg_w4a8": ("h_a8", True),
    }
    for arm, (upstream_name, quantize_act) in arm_contract.items():
        gate_out, up_out, activation = upstream[upstream_name]
        magnitude_key = f"act_from_{upstream_name}"
        if magnitude_key not in magnitude_paths:
            magnitude_paths[magnitude_key] = down_transform_points(
                activation, down.suh, hadamard
            )
        down_source = magnitude_paths[magnitude_key]["post_h_qdq_input"]
        if quantize_act:
            down_operand, observation = quant_dequant_mxfp8(down_source)
        else:
            down_operand, observation = _fp16_boundary(down_source), None
        if quantize_act:
            if observation is None:
                raise AssertionError("act-A8 observation was not produced")
            qdq[magnitude_key] = (
                down_source,
                down_operand,
                observation,
            )
        down_acc = native_label_gemm(down_operand, down.weight)
        expert_output = apply_output_transform(down_acc, down.svh, hadamard)
        stages[arm] = {
            "gate": gate_out,
            "up": up_out,
            "act": activation,
            "expert_output": expert_output,
        }
    return stages, qdq, magnitude_paths


def _parse_sha256_sidecar(path: Path) -> tuple[str, str]:
    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or len(fields[0]) != 64:
        raise RuntimeError(f"malformed SHA-256 sidecar: {path}")
    return fields[0], fields[1]


def verify_sha256_sidecar(payload: Path, sidecar: Path) -> str:
    expected, filename = _parse_sha256_sidecar(sidecar)
    if filename != payload.name:
        raise RuntimeError(f"SHA-256 sidecar filename differs for {payload}")
    observed = _sha256_file(payload)
    if observed != expected:
        raise RuntimeError(f"SHA-256 sidecar mismatch for {payload}")
    return observed


def validate_tensor_contract(
    *,
    prefix: str,
    bits: int,
    tensor_manifest: Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> None:
    """Fail closed on the frozen per-tensor rate and official-source binding."""

    if bits not in (3, 4) or int(tensor_manifest.get("bits", -1)) != bits:
        raise RuntimeError(f"{prefix}: frozen per-tensor K3/K4 assignment differs")
    if tensor_manifest.get("codebook") != "sqg_xor_cheb_t12":
        raise RuntimeError(f"{prefix}: non-SQG codebook in candidate")
    forbidden = set(tensor_manifest.get("forbidden_input_reads", ()))
    required_forbidden = {
        "mcg.suh",
        "mcg.svh",
        "mcg.scales",
        "mcg.trellis",
        "mcg.packed_bytes",
    }
    if not required_forbidden.issubset(forbidden):
        raise RuntimeError(f"{prefix}: candidate does not forbid legacy MCG inputs")
    source = tensor_manifest.get("source", {})
    expected_pairs = (
        (source.get("kind"), "official_bf16", "source kind"),
        (source.get("mcg_source"), False, "MCG-source flag"),
        (source.get("tensor_name"), source_binding.get("source_name"), "tensor name"),
        (
            source.get("tensor_payload_sha256"),
            source_binding.get("bf16_sha256"),
            "BF16 tensor payload",
        ),
        (source.get("shard_name"), source_binding.get("source_shard"), "source shard"),
    )
    for observed, expected, label in expected_pairs:
        if observed != expected:
            raise RuntimeError(f"{prefix}: {label} binding differs")
    transform = tensor_manifest.get("transform", {})
    if int(transform.get("hadamard_block", -1)) != HADAMARD_BLOCK:
        raise RuntimeError(f"{prefix}: Hadamard block contract differs")
    if transform.get("legacy_transform_input") is not False:
        raise RuntimeError(f"{prefix}: legacy transform input is not forbidden")


def sealed_mcg_layer_scalar(receipt: Mapping[str, Any], layer: int) -> float:
    """Extract only the layer-matched MCG scalar, never the block aggregate."""

    try:
        value = float(
            receipt["layers"][str(layer)]["metrics"]["mcg"]["signed_top8_nmse"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("sealed receipt lacks the layer-matched MCG scalar") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("sealed layer-matched MCG scalar is invalid")
    return value


def _load_and_validate_projections(
    *,
    candidate_root: Path,
    permutation_root: Path,
    layer: int,
    expert: int,
    manifest: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    bf16_manifest: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, NativeProjection], torch.Tensor, dict[str, Any]]:
    permutation_path = (
        permutation_root
        / f"layer_{layer:03d}/preparation/permutations/expert-{expert:03d}.json"
    )
    permutation_json = json.loads(permutation_path.read_text(encoding="utf-8"))
    if not permutation_json.get("complete"):
        raise RuntimeError(f"layer {layer} expert {expert}: permutation incomplete")
    order = torch.tensor(permutation_json["new_to_old"], dtype=torch.long, device=device)
    if order.shape != (INTERMEDIATE,) or not torch.equal(
        torch.sort(order).values, torch.arange(INTERMEDIATE, device=device)
    ):
        raise RuntimeError(f"layer {layer} expert {expert}: malformed permutation")
    permutation_payload_sha256 = tensor_sha256(order.cpu())
    if permutation_payload_sha256 != permutation_json["manifest"].get(
        "new_to_old_sha256"
    ):
        raise RuntimeError(f"layer {layer} expert {expert}: permutation hash differs")

    shard = candidate_root / f"layer_{layer:03d}/experts" / str(manifest["shard"])
    expected_keys = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.{suffix}"
        for projection in PROJECTIONS
        for suffix in ("trellis", "suh", "svh", "sqg")
    }
    projections: dict[str, NativeProjection] = {}
    bit_receipt: dict[str, int] = {}
    with safe_open(shard, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != expected_keys:
            raise RuntimeError(f"layer {layer} expert {expert}: shard tensor census differs")
        for projection in PROJECTIONS:
            prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
            bits = int(sidecar["bit_map"][prefix])
            tensor_manifest = manifest["tensor_manifests"][prefix]
            source_binding = sidecar["tensor_provenance"][prefix]
            validate_tensor_contract(
                prefix=prefix,
                bits=bits,
                tensor_manifest=tensor_manifest,
                source_binding=source_binding,
            )
            source_shard = str(source_binding["source_shard"])
            sealed_shard = bf16_manifest["shards"].get(source_shard)
            if sealed_shard is None:
                raise RuntimeError(f"{prefix}: source shard is absent from BF16 seal")
            if tensor_manifest["source"].get("shard_sha256") != sealed_shard["sha256"]:
                raise RuntimeError(f"{prefix}: BF16 source shard seal differs")

            packed = handle.get_tensor(prefix + ".trellis")
            suh = handle.get_tensor(prefix + ".suh")
            svh = handle.get_tensor(prefix + ".svh")
            marker = handle.get_tensor(prefix + ".sqg")
            if marker.dtype != torch.int32 or marker.numel() != 1 or int(marker.item()) != SQG_MARKER:
                raise RuntimeError(f"{prefix}: SQG marker differs")
            if tensor_sha256(packed) != tensor_manifest["packed_trellis_sha256"]:
                raise RuntimeError(f"{prefix}: packed trellis hash differs")
            if tensor_sha256(suh) != tensor_manifest["scales"]["suh_sha256"]:
                raise RuntimeError(f"{prefix}: suh hash differs")
            if tensor_sha256(svh) != tensor_manifest["scales"]["svh_sha256"]:
                raise RuntimeError(f"{prefix}: svh hash differs")
            if suh.dtype != torch.float16 or svh.dtype != torch.float16:
                raise RuntimeError(f"{prefix}: scale dtype differs from FP16 contract")

            from kquant.sqg_e4m3 import sqg_xor_cheb_t12_bytes

            lut = sqg_xor_cheb_t12_bytes(bits)
            if payload_sha256(lut) != tensor_manifest["codebook_lut_sha256"]:
                raise RuntimeError(f"{prefix}: native E4M3 LUT hash differs")
            states = unpack_trellis_states(packed, bits)
            native = decode_regularized_states(states, lut).to(device)
            widened_raw = native.to(torch.float8_e4m3fn).view(torch.uint8)
            if not torch.equal(native, widened_raw.view(torch.float8_e4m3fn).float()):
                raise RuntimeError(f"{prefix}: decoded labels do not lie exactly on E4M3")
            projections[projection] = NativeProjection(
                weight=native,
                suh=suh.to(device),
                svh=svh.to(device),
                bits=bits,
            )
            bit_receipt[projection] = bits
            permutation_hash = tensor_manifest["transform"]["physical_permutation"][
                "new_to_old_sha256"
            ]
            if permutation_hash != permutation_json["manifest"]["new_to_old_sha256"]:
                raise RuntimeError(f"{prefix}: physical permutation binding differs")
    return projections, order, {
        "expert": expert,
        "bits": bit_receipt,
        "permutation_artifact_id": permutation_json["artifact_id"],
        "permutation_sha256": permutation_payload_sha256,
    }


def _validate_input_provenance(
    *,
    candidate_root: Path,
    capture_root: Path,
    permutation_root: Path,
    bf16_root: Path,
    bf16_manifest_path: Path,
    selection_receipt_path: Path,
    holdout_receipt_path: Path,
    mcg_root: Path,
    layer: int,
    verify_capture_payloads: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    for label, path in (
        ("candidate", candidate_root),
        ("capture", capture_root),
        ("permutation", permutation_root),
        ("BF16", bf16_root),
        ("MCG metadata", mcg_root),
    ):
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError(f"{label} root is absent, not a directory, or a symlink: {path}")
    for path in (
        bf16_manifest_path,
        selection_receipt_path,
        holdout_receipt_path,
    ):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"sealed manifest is absent or unsafe: {path}")

    run_seal_path = candidate_root / "run_seal.json"
    run_seal = json.loads(run_seal_path.read_text(encoding="utf-8"))
    layer_entry = run_seal.get("layers", {}).get(str(layer), {})
    if (
        run_seal.get("complete") is not True
        or run_seal.get("census", {}).get("mcg_tensors") != 0
        or layer_entry.get("mcg_tensors") != 0
        or layer_entry.get("sqg_tensors") != 768
        or layer_entry.get("K3") != 384
        or layer_entry.get("K4") != 384
    ):
        raise RuntimeError("candidate run seal does not close as zero-MCG 384-K3/384-K4")

    assembly_path = candidate_root / f"layer_{layer:03d}/assembly_result.json"
    assembly = json.loads(assembly_path.read_text(encoding="utf-8"))
    if (
        assembly.get("complete") is not True
        or assembly.get("mcg_tensors") != 0
        or assembly.get("sqg_tensors") != 768
        or assembly.get("bit_histogram") != {"3": 384, "4": 384}
    ):
        raise RuntimeError("candidate layer assembly census differs")

    selection_receipt = json.loads(selection_receipt_path.read_text(encoding="utf-8"))
    holdout_receipt = json.loads(holdout_receipt_path.read_text(encoding="utf-8"))
    candidate_resolved = str(candidate_root.resolve())
    if (
        selection_receipt.get("role") != "selection"
        or selection_receipt.get("aggregate", {}).get("selection_policy", {}).get("winner")
        != "mcg"
        or selection_receipt.get("aggregate", {}).get("selection_policy", {}).get(
            "diagnostic_fallback_nonbaseline"
        )
        != "sqg_a025"
        or selection_receipt.get("candidates", {}).get("sqg_a025")
        != candidate_resolved
    ):
        raise RuntimeError("selection receipt does not bind SQG alpha-0.25 diagnostic arm")
    if (
        holdout_receipt.get("role") != "holdout"
        or holdout_receipt.get("candidates", {}).get("sqg_a025")
        != candidate_resolved
    ):
        raise RuntimeError("holdout receipt does not bind SQG alpha-0.25 diagnostic arm")
    sealed_mcg_scalars = {
        "selection": sealed_mcg_layer_scalar(selection_receipt, layer),
        "holdout": sealed_mcg_layer_scalar(holdout_receipt, layer),
    }
    sealed_mcg_block_aggregate_scalars = {
        "selection": float(
            selection_receipt["aggregate"]["metrics"]["mcg"]["signed_top8_nmse"]
        ),
        "holdout": float(
            holdout_receipt["aggregate"]["metrics"]["mcg"]["signed_top8_nmse"]
        ),
    }

    manifests = _load_sqg_manifests(candidate_root, layer)
    manifest_hashes: list[tuple[str, str]] = []
    for expert in range(256):
        path = (
            candidate_root
            / f"layer_{layer:03d}/experts/layer-{layer:03d}-expert-{expert:03d}.json"
        )
        observed = verify_sha256_sidecar(path, path.with_suffix(".json.sha256"))
        manifest_hashes.append((path.name, observed))

    capture_manifest_path = capture_root / "capture_manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    capture_exclusions = capture_manifest.get("construction_exclusions", {})
    if capture_manifest.get("complete") is not True or any(
        int(capture_exclusions.get(name, -1)) != 0
        for name in (
            "mcg_encoder_seeds",
            "mcg_payload_bytes",
            "mcg_permutations",
            "mcg_scale_vectors",
            "mcg_transform_vectors",
        )
    ):
        raise RuntimeError("capture manifest does not close as a zero-MCG capture")
    capture_layer_ref = capture_manifest["layer_manifests"].get(str(layer))
    layer_manifest_path = capture_root / f"layer_{layer:03d}/layer_manifest.json"
    layer_manifest_sha256 = _sha256_file(layer_manifest_path)
    if capture_layer_ref is None or capture_layer_ref["sha256"] != layer_manifest_sha256:
        raise RuntimeError("capture layer manifest binding differs")
    capture_layer = json.loads(layer_manifest_path.read_text(encoding="utf-8"))
    if (
        capture_layer.get("schema") != "glm52-fresh-sqg-layer-capture-v1"
        or capture_layer.get("topk") != TOPK
        or capture_layer.get("hidden") != HIDDEN
        or capture_layer.get("num_experts") != 256
        or any(int(capture_layer["role_rows"].get(str(ROLES[r]), 0)) <= 0 for r in ROLES)
    ):
        raise RuntimeError("capture layer dimensions or requested roles differ")
    verified_capture_files: dict[str, str] = {}
    for filename, file_manifest in capture_layer["files"].items():
        path = capture_root / f"layer_{layer:03d}" / filename
        if not path.is_file() or path.is_symlink() or path.stat().st_size != int(
            file_manifest["bytes"]
        ):
            raise RuntimeError(f"capture file size or type differs: {path}")
        if verify_capture_payloads:
            observed = _sha256_file(path)
            if observed != file_manifest["sha256"]:
                raise RuntimeError(f"capture payload hash differs: {path}")
            verified_capture_files[filename] = observed

    bf16_manifest = json.loads(bf16_manifest_path.read_text(encoding="utf-8"))
    if (
        bf16_manifest.get("schema") != "glm52-fresh-sqg-bf16-shard-manifest-v1"
        or layer not in bf16_manifest.get("layers", [])
        or bf16_manifest.get("repo") != "zai-org/GLM-5.2"
    ):
        raise RuntimeError("BF16 shard manifest does not cover the requested official layer")
    for name, value in bf16_manifest["shards"].items():
        path = bf16_root / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != int(value["bytes"]):
            raise RuntimeError(f"BF16 sealed shard is absent or has the wrong size: {path}")

    sidecar_path = mcg_root / f"r7-experts-layer-{layer:03d}.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected_bit_histogram = {3: 0, 4: 0}
    for prefix, raw_bits in sidecar["bit_map"].items():
        if prefix.startswith(f"model.layers.{layer}.mlp.experts."):
            bits = int(raw_bits)
            if bits not in expected_bit_histogram:
                raise RuntimeError("protected per-tensor rate map contains a non-K3/K4 rate")
            expected_bit_histogram[bits] += 1
    if expected_bit_histogram != {3: 384, 4: 384}:
        raise RuntimeError("protected topology-neutral rate census differs")

    provenance = {
        "candidate": {
            "root": str(candidate_root.resolve()),
            "run_seal_sha256": _sha256_file(run_seal_path),
            "run_seal_id": run_seal["run_seal_id"],
            "layer_assembly_sha256": _sha256_file(assembly_path),
            "expert_manifest_set_sha256": _canonical_json_sha256(manifest_hashes),
            "expert_manifests": len(manifest_hashes),
            "sqg_tensors": 768,
            "mcg_tensors": 0,
            "selection_status": "MCG retained formal hard-gate winner",
            "selected_diagnostic_arm": "sqg_a025",
            "selection_receipt_sha256": _sha256_file(selection_receipt_path),
            "holdout_receipt_sha256": _sha256_file(holdout_receipt_path),
            "sealed_mcg_a16_scalar_comparators": sealed_mcg_scalars,
            "sealed_mcg_a16_comparator_scope": f"layer-{layer:03d}",
            "sealed_mcg_a16_four_layer_block_aggregate_not_used": (
                sealed_mcg_block_aggregate_scalars
            ),
            "mcg_comparison_scope": "cross-harness scalar only; not a Test 8b arm",
        },
        "capture": {
            "root": str(capture_root.resolve()),
            "capture_manifest_sha256": _sha256_file(capture_manifest_path),
            "layer_manifest_sha256": layer_manifest_sha256,
            "capture_run_uuid": capture_layer["capture_run_uuid"],
            "document_plan_fingerprint": capture_layer["document_plan_fingerprint"],
            "documents_verified": capture_layer["documents_verified"],
            "payload_hashes_recomputed": verify_capture_payloads,
            "verified_payloads": verified_capture_files,
        },
        "bf16": {
            "root": str(bf16_root.resolve()),
            "manifest": str(bf16_manifest_path.resolve()),
            "manifest_sha256": _sha256_file(bf16_manifest_path),
            "repository": bf16_manifest["repo"],
            "revision": bf16_manifest["revision"],
            "raw_shard_hashes_recomputed": False,
            "note": "all source tensors must match SQG and protected metadata payload bindings; sealed shard sizes are checked",
        },
        "frozen_rate_metadata": {
            "sidecar": str(sidecar_path.resolve()),
            "sidecar_sha256": _sha256_file(sidecar_path),
            "K3": expected_bit_histogram[3],
            "K4": expected_bit_histogram[4],
            "scope": "per tensor, never per expert",
            "payload_bytes_read": False,
        },
        "permutations": {"root": str(permutation_root.resolve())},
    }
    return provenance, sidecar, manifests


def _routing_index(
    capture_layer: Path,
) -> tuple[np.memmap, np.memmap, np.memmap, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]:
    role_path = capture_layer / "role_ids.u8.bin"
    total_rows = role_path.stat().st_size
    role_ids = np.memmap(role_path, mode="r", dtype="u1", shape=(total_rows,))
    topk_ids = np.memmap(
        capture_layer / "topk_ids.u8.bin", mode="r", dtype="u1", shape=(total_rows, TOPK)
    )
    topk_weights = np.memmap(
        capture_layer / "topk_weights.f32le.bin",
        mode="r",
        dtype="<f4",
        shape=(total_rows, TOPK),
    )
    routes: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for role, role_id in ROLES.items():
        selected = np.flatnonzero(np.asarray(role_ids) == role_id).astype(np.int64, copy=False)
        ids = np.asarray(topk_ids[selected]).reshape(-1)
        rows = np.repeat(selected, TOPK)
        slots = np.tile(np.arange(TOPK, dtype=np.int16), selected.size)
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        bounds = np.searchsorted(sorted_ids, np.arange(257), side="left")
        routes[role] = {
            expert: (
                rows[order[bounds[expert] : bounds[expert + 1]]],
                slots[order[bounds[expert] : bounds[expert + 1]]],
            )
            for expert in range(256)
        }
    hidden_words = np.memmap(
        capture_layer / "hidden.bf16.bin",
        mode="r",
        dtype="<u2",
        shape=(total_rows, HIDDEN),
    )
    return hidden_words, role_ids, topk_weights, routes


@dataclass
class RoleState:
    selected_rows: np.ndarray
    compact: np.ndarray
    reference_sum: torch.Tensor
    delta_sums: dict[str, torch.Tensor]
    stage_denominator: dict[str, float]
    stage_numerator: dict[str, dict[str, float]]
    operand_stats: dict[str, OperandStats]
    activation_magnitudes: dict[str, dict[str, MagnitudeStats]]
    individual_output_sse: dict[str, float]
    routes: int = 0


def _make_role_state(
    *, role_ids: np.memmap, role: str, device: torch.device
) -> RoleState:
    selected = np.flatnonzero(np.asarray(role_ids) == ROLES[role]).astype(np.int64, copy=False)
    compact = np.full(role_ids.shape[0], -1, dtype=np.int32)
    compact[selected] = np.arange(selected.size, dtype=np.int32)
    return RoleState(
        selected_rows=selected,
        compact=compact,
        reference_sum=torch.zeros((selected.size, HIDDEN), dtype=torch.float32, device=device),
        delta_sums={
            arm: torch.zeros((selected.size, HIDDEN), dtype=torch.float32, device=device)
            for arm in ARMS
        },
        stage_denominator={stage: 0.0 for stage in STAGES},
        stage_numerator={arm: {stage: 0.0 for stage in STAGES} for arm in ARMS},
        operand_stats={
            "h": OperandStats(),
            "act_from_a16": OperandStats(),
            "act_from_h_a8": OperandStats(),
        },
        activation_magnitudes={
            operand: {
                point: MagnitudeStats()
                for point in ("raw", "after_suh_pre_h", "post_h_qdq_input")
            }
            for operand in ("h", "act_from_a16", "act_from_h_a8")
        },
        individual_output_sse={arm: 0.0 for arm in ARMS},
    )


def _weighted_stage_update(
    state: RoleState,
    reference: Mapping[str, torch.Tensor],
    candidates: Mapping[str, Mapping[str, torch.Tensor]],
    gates: torch.Tensor,
) -> None:
    importance = gates.float().square().unsqueeze(1)
    for stage in STAGES:
        target = reference[stage].float()
        state.stage_denominator[stage] += float(
            (target.square() * importance).sum(dtype=torch.float64).item()
        )
        for arm in ARMS:
            error = candidates[arm][stage].float() - target
            state.stage_numerator[arm][stage] += float(
                (error.square() * importance).sum(dtype=torch.float64).item()
            )


def cross_harness_delta_percent(candidate_nmse: float, comparator_nmse: float) -> float:
    if (
        not math.isfinite(candidate_nmse)
        or candidate_nmse < 0.0
        or not math.isfinite(comparator_nmse)
        or comparator_nmse <= 0.0
    ):
        raise ValueError("cross-harness NMSE scalars must be finite and nonnegative")
    return (candidate_nmse / comparator_nmse - 1.0) * 100.0


def _finish_role(
    *,
    role: str,
    state: RoleState,
    capture_layer: Path,
    output: Path,
    sealed_mcg_a16_nmse: float,
    sealed_mcg_receipt_sha256: str,
) -> dict[str, Any]:
    reference_energy, errors = _reduce_rows(state.reference_sum, state.delta_sums)
    baseline_error = errors["sqg_a16"]
    metrics: dict[str, Any] = {}
    denominator = float(reference_energy.sum(dtype=np.float64))
    for arm in ARMS:
        sse = float(errors[arm].sum(dtype=np.float64))
        metrics[arm] = {
            "signed_top8_nmse": sse / denominator,
            "signed_top8_sse": sse,
            "reference_signed_top8_energy": denominator,
            "relative_to_sqg_a16_percent": (
                sse / float(baseline_error.sum(dtype=np.float64)) - 1.0
            )
            * 100.0,
            "cross_harness_delta_vs_sealed_mcg_a16_percent": (
                cross_harness_delta_percent(sse / denominator, sealed_mcg_a16_nmse)
            ),
            "sum_individual_route_sse": state.individual_output_sse[arm],
            "summed_over_individual_sse": sse / state.individual_output_sse[arm],
            "cross_expert_error_term": sse - state.individual_output_sse[arm],
            "tail_vs_sqg_a16": tail_comparison(errors[arm], baseline_error, reference_energy),
            "functional_stage_nmse": {
                stage: state.stage_numerator[arm][stage]
                / max(state.stage_denominator[stage], np.finfo(np.float64).tiny)
                for stage in STAGES
            },
        }

    total_rows = state.compact.size
    doc_epochs = np.memmap(
        capture_layer / "doc_epochs.u32le.bin", mode="r", dtype="<u4", shape=(total_rows,)
    )
    token_positions = np.memmap(
        capture_layer / "token_positions.u16le.bin", mode="r", dtype="<u2", shape=(total_rows,)
    )
    npz_path = output.with_name(f"{output.stem}-{role}.npz")
    np.savez_compressed(
        npz_path,
        global_rows=state.selected_rows,
        doc_epochs=np.asarray(doc_epochs[state.selected_rows]),
        token_positions=np.asarray(token_positions[state.selected_rows]),
        reference_energy=reference_energy,
        **{f"error__{arm}": values for arm, values in errors.items()},
    )
    return {
        "role": role,
        "positions": int(state.selected_rows.size),
        "routes": state.routes,
        "selected_global_rows_sha256": _sha256_array(state.selected_rows),
        "per_position_artifact": str(npz_path.resolve()),
        "per_position_artifact_sha256": _sha256_file(npz_path),
        "metrics": metrics,
        "external_mcg_a16_scalar_comparator": {
            "signed_top8_nmse": sealed_mcg_a16_nmse,
            "scope": "layer-077",
            "receipt_sha256": sealed_mcg_receipt_sha256,
            "comparison_kind": "cross_harness_scalar_only",
            "numeric_oracle_arm": False,
            "per_position_vector_available_in_test8b": False,
            "tail_comparison_permitted": False,
            "note": (
                "The scalar is bound from the sealed earlier signed-top8 receipt. "
                "It is not an arm executed by this direct-label numeric oracle."
            ),
        },
        "operand_quantization": {
            name: stats.finish() for name, stats in state.operand_stats.items()
        },
        "activation_magnitude_path": {
            operand: {
                point: stats.finish() for point, stats in points.items()
            }
            for operand, points in state.activation_magnitudes.items()
        },
    }


def validate_result_contract(result: Mapping[str, Any]) -> None:
    if result.get("schema") != "glm52-sqg-w4a8-activation-quality-v1":
        raise RuntimeError("Test 8b result schema differs")
    method = result.get("method", {})
    if tuple(method.get("arms", ())) != ARMS or tuple(method.get("execution_order", ())) != EXECUTION_ORDER:
        raise RuntimeError("Test 8b arm or execution-order receipt differs")
    required_true = (
        "native_e4m3_weight_labels",
        "per_k32_ue8m0_e4m3_activation_qdq",
        "fp32_gemm_accumulation",
        "fp16_inter_gemm_boundaries",
        "hadamard_kept_on_activation_side",
        "per_tensor_rate_map_frozen",
        "no_mcg_payload_bytes",
        "signed_router_weighted_top8_sum_before_square",
        "activation_magnitude_exact_moments_and_strided_quantiles",
        "sealed_mcg_a16_scalar_comparator_bound",
        "mcg_a16_is_not_numeric_oracle_arm",
    )
    if any(method.get(name) is not True for name in required_true):
        raise RuntimeError("Test 8b method receipt is missing a required invariant")
    if method.get("mcg_per_position_tail_comparison") is not False:
        raise RuntimeError("Test 8b must not claim a per-position MCG tail comparison")
    if set(result.get("roles", {})) != set(ROLES):
        raise RuntimeError("Test 8b must contain both selection and holdout")
    for role, payload in result["roles"].items():
        if payload.get("positions", 0) <= 0 or payload.get("routes", 0) <= 0:
            raise RuntimeError(f"Test 8b {role} contains no evaluated rows")
        if set(payload.get("metrics", {})) != set(ARMS):
            raise RuntimeError(f"Test 8b {role} arm census differs")
        comparator = payload.get("external_mcg_a16_scalar_comparator", {})
        if (
            comparator.get("comparison_kind") != "cross_harness_scalar_only"
            or comparator.get("scope") != "layer-077"
            or comparator.get("numeric_oracle_arm") is not False
            or comparator.get("tail_comparison_permitted") is not False
        ):
            raise RuntimeError(f"Test 8b {role} MCG scalar scope differs")
        magnitude = payload.get("activation_magnitude_path", {})
        if set(magnitude) != {"h", "act_from_a16", "act_from_h_a8"} or any(
            set(points) != {"raw", "after_suh_pre_h", "post_h_qdq_input"}
            for points in magnitude.values()
        ):
            raise RuntimeError(f"Test 8b {role} activation magnitude path differs")
    go_no_go = result.get("go_no_go_limit", {})
    if (
        go_no_go.get("can_kill_w4a8_for_activation_quality") is not True
        or go_no_go.get("can_greenlight_full_quant_by_itself") is not False
    ):
        raise RuntimeError("Test 8b go/no-go scope differs")


def _render(result: Mapping[str, Any]) -> str:
    lines = [
        "# Test 8b: GLM-5.2 SQG A8 activation quality",
        "",
        "This is a layer-77 functional oracle. It is not final-logit KLD and not a speed benchmark.",
        "",
    ]
    for role in ROLES:
        payload = result["roles"][role]
        lines.extend(
            [
                f"## {role.title()}",
                "",
                "| Arm | Signed top-8 NMSE | vs SQG A16 | vs sealed MCG A16 scalar | Improved positions vs A16 | Output-stage NMSE |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for arm in ARMS:
            metric = payload["metrics"][arm]
            tail = metric["tail_vs_sqg_a16"]
            lines.append(
                f"| {arm} | {metric['signed_top8_nmse']:.9e} | "
                f"{metric['relative_to_sqg_a16_percent']:+.4f}% | "
                f"{metric['cross_harness_delta_vs_sealed_mcg_a16_percent']:+.4f}% | "
                f"{tail['improved_fraction']:.3%} | "
                f"{metric['functional_stage_nmse']['expert_output']:.9e} |"
            )
        lines.extend(["", "### Activation operands", ""])
        for name, operand in payload["operand_quantization"].items():
            lines.append(
                f"- `{name}`: NMSE `{operand['quantization_nmse']:.9e}`, "
                f"max |x| `{operand['max_abs_source']:.6g}`, "
                f"pre-clamp overflows `{operand['preclamp_overflow_values']}`."
            )
        lines.append("")
        comparator = payload["external_mcg_a16_scalar_comparator"]
        lines.extend(
            [
                "The sealed MCG-A16 value "
                f"`{comparator['signed_top8_nmse']:.9e}` is cross-harness scalar "
                "context only. It has no Test 8b per-position arm, so no MCG tail "
                "comparison is made.",
                "",
                "### Magnitude path around H128",
                "",
            ]
        )
        for operand, points in payload["activation_magnitude_path"].items():
            lines.append(f"- `{operand}`")
            for point, distribution in points.items():
                exact = distribution["exact"]
                sampled = distribution["strided_abs_distribution"]
                lines.append(
                    f"  - `{point}`: exact RMS `{exact['rms']:.6g}`, exact max "
                    f"`{exact['max_abs']:.6g}`, sampled p99 "
                    f"`{sampled.get('p99', float('nan')):.6g}`."
                )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "The A16 arm is the matched SQG control. `h-A8` and `act-A8` isolate the two activation sites; `W4A8` composes them. Selection and holdout are reported separately. The holdout was not used by the encoder, but was already inspected in earlier analysis and is therefore secondary confirmation rather than a newly blind split.",
            "",
            "Test 8b can reject W4A8 for activation-quality damage. Passing Test 8b cannot by itself greenlight a full quant: Test 8c must still demonstrate the required GLM-shape prefill speed, and this oracle does not measure final-logit KLD.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--permutation-root", type=Path, default=DEFAULT_PERMUTATION_ROOT)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--bf16-root", type=Path, default=DEFAULT_BF16_ROOT)
    parser.add_argument("--bf16-manifest", type=Path, default=DEFAULT_BF16_MANIFEST)
    parser.add_argument(
        "--selection-receipt", type=Path, default=DEFAULT_SELECTION_RECEIPT
    )
    parser.add_argument("--holdout-receipt", type=Path, default=DEFAULT_HOLDOUT_RECEIPT)
    parser.add_argument("--mcg-root", type=Path, default=DEFAULT_MCG_ROOT)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--limit-experts", type=int)
    parser.add_argument("--skip-capture-payload-hashes", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.layer != LAYER:
        raise ValueError("preregistered Test 8b is restricted to selected layer 77")
    if args.chunk_rows <= 0:
        raise ValueError("chunk rows must be positive")
    if args.limit_experts is not None and not 1 <= args.limit_experts <= 256:
        raise ValueError("limit experts must be in [1,256]")
    output = args.output.resolve()
    derived = [
        output,
        output.with_suffix(".md"),
        output.with_suffix(output.suffix + ".sha256"),
        *(output.with_name(f"{output.stem}-{role}.npz") for role in ROLES),
    ]
    if any(path.exists() or path.is_symlink() for path in derived):
        raise RuntimeError("Test 8b refuses to overwrite an existing result artifact")
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("full Test 8b requires an explicit available CUDA device")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    provenance, sidecar, manifests = _validate_input_provenance(
        candidate_root=args.candidate_root.resolve(),
        capture_root=args.capture_root.resolve(),
        permutation_root=args.permutation_root.resolve(),
        bf16_root=args.bf16_root.resolve(),
        bf16_manifest_path=args.bf16_manifest.resolve(),
        selection_receipt_path=args.selection_receipt.resolve(),
        holdout_receipt_path=args.holdout_receipt.resolve(),
        mcg_root=args.mcg_root.resolve(),
        layer=args.layer,
        verify_capture_payloads=not args.skip_capture_payload_hashes,
    )
    bf16_manifest = json.loads(args.bf16_manifest.read_text(encoding="utf-8"))
    capture_layer = args.capture_root / f"layer_{args.layer:03d}"
    hidden_words, role_ids, topk_weights, routes = _routing_index(capture_layer)
    states = {
        role: _make_role_state(role_ids=role_ids, role=role, device=device)
        for role in ROLES
    }
    hadamard = normalized_hadamard(
        device=device, dtype=torch.float32, size=HADAMARD_BLOCK
    )
    expert_limit = 256 if args.limit_experts is None else args.limit_experts
    expert_receipts: list[dict[str, Any]] = []
    source_bindings_checked = 0

    with torch.inference_mode():
        for expert in range(expert_limit):
            projections, permutation, expert_receipt = _load_and_validate_projections(
                candidate_root=args.candidate_root.resolve(),
                permutation_root=args.permutation_root.resolve(),
                layer=args.layer,
                expert=expert,
                manifest=manifests[expert],
                sidecar=sidecar,
                bf16_manifest=bf16_manifest,
                device=device,
            )
            expert_receipts.append(expert_receipt)
            source_hf = _read_source_triplet(
                args.bf16_root.resolve(), sidecar, args.layer, expert
            )
            for projection in PROJECTIONS:
                prefix = (
                    f"model.layers.{args.layer}.mlp.experts.{expert}.{projection}"
                )
                observed_source_hash = payload_sha256(source_hf[projection])
                expected_source_hash = sidecar["tensor_provenance"][prefix][
                    "bf16_sha256"
                ]
                if observed_source_hash != expected_source_hash:
                    raise RuntimeError(f"{prefix}: official BF16 payload hash differs")
                source_bindings_checked += 1
            source = {
                projection: source_hf[projection].T.float().to(device)
                for projection in PROJECTIONS
            }

            for role, state in states.items():
                row_indices, slots = routes[role][expert]
                if row_indices.size == 0:
                    continue
                compact_rows = torch.from_numpy(
                    state.compact[row_indices].astype(np.int64)
                ).to(device)
                if bool((compact_rows < 0).any()):
                    raise RuntimeError("role route failed compact-row mapping")
                gates_all = torch.from_numpy(
                    np.array(topk_weights[row_indices, slots], dtype=np.float32, copy=True)
                ).to(device)
                for begin in range(0, row_indices.size, args.chunk_rows):
                    end = min(row_indices.size, begin + args.chunk_rows)
                    hidden = _load_hidden_chunk(hidden_words, row_indices[begin:end], device)
                    gates = gates_all[begin:end]
                    indices = compact_rows[begin:end]
                    arm_stages, qdq, magnitude_paths = evaluate_sqg_arms(
                        hidden, projections, hadamard
                    )

                    source_gate_original = torch.matmul(hidden, source["gate_proj"])
                    source_up_original = torch.matmul(hidden, source["up_proj"])
                    source_gate = source_gate_original.index_select(1, permutation)
                    source_up = source_up_original.index_select(1, permutation)
                    source_act = F.silu(source_gate.float()) * source_up.float()
                    source_output = torch.matmul(
                        F.silu(source_gate_original.float()) * source_up_original.float(),
                        source["down_proj"],
                    )
                    reference = {
                        "gate": source_gate,
                        "up": source_up,
                        "act": source_act,
                        "expert_output": source_output,
                    }
                    _weighted_stage_update(state, reference, arm_stages, gates)

                    weighted_reference = source_output * gates[:, None]
                    state.reference_sum.index_add_(0, indices, weighted_reference)
                    for arm in ARMS:
                        signed_delta = (
                            arm_stages[arm]["expert_output"] - source_output
                        ) * gates[:, None]
                        state.delta_sums[arm].index_add_(0, indices, signed_delta)
                        route_sse = float(signed_delta.double().square().sum().item())
                        state.individual_output_sse[arm] += route_sse
                    for operand, (source_operand, quantized, observation) in qdq.items():
                        state.operand_stats[operand].update(
                            source_operand, quantized, observation
                        )
                    for operand, points in magnitude_paths.items():
                        for point, values in points.items():
                            state.activation_magnitudes[operand][point].update(values)
                    state.routes += int(end - begin)
                del compact_rows, gates_all
            del projections, source_hf, source
            if (expert + 1) % 8 == 0 or expert + 1 == expert_limit:
                print(f"layer 77 Test 8b: {expert + 1}/{expert_limit} experts", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    comparator_receipt_hashes = {
        "selection": provenance["candidate"]["selection_receipt_sha256"],
        "holdout": provenance["candidate"]["holdout_receipt_sha256"],
    }
    sealed_mcg_scalars = provenance["candidate"][
        "sealed_mcg_a16_scalar_comparators"
    ]
    role_results = {
        role: _finish_role(
            role=role,
            state=state,
            capture_layer=capture_layer,
            output=output,
            sealed_mcg_a16_nmse=float(sealed_mcg_scalars[role]),
            sealed_mcg_receipt_sha256=comparator_receipt_hashes[role],
        )
        for role, state in states.items()
    }
    admissible = expert_limit == 256
    result = {
        "schema": "glm52-sqg-w4a8-activation-quality-v1",
        "test": "8b",
        "layer": args.layer,
        "device": str(device),
        "admissible_full_layer_result": admissible,
        "diagnostic_expert_limit": None if admissible else expert_limit,
        "method": {
            "arms": list(ARMS),
            "execution_order": list(EXECUTION_ORDER),
            "native_e4m3_weight_labels": True,
            "per_k32_ue8m0_e4m3_activation_qdq": True,
            "fp32_gemm_accumulation": True,
            "fp16_inter_gemm_boundaries": True,
            "hadamard_kept_on_activation_side": True,
            "permutation_kept_out_of_weight_labels": True,
            "swiglu_evaluated_in_shared_permuted_intermediate_basis": True,
            "per_tensor_rate_map_frozen": True,
            "no_mcg_payload_bytes": True,
            "signed_router_weighted_top8_sum_before_square": True,
            "activation_magnitude_points_reported": [
                "raw",
                "after_suh_pre_h",
                "post_h_qdq_input",
            ],
            "activation_magnitude_exact_moments_and_strided_quantiles": True,
            "sealed_mcg_a16_scalar_comparator_bound": True,
            "mcg_a16_is_not_numeric_oracle_arm": True,
            "mcg_per_position_tail_comparison": False,
            "bf16_reference_expert_functions": True,
            "selection_rows_used": True,
            "holdout_rows_used": True,
            "fit_rows_used": False,
            "holdout_encoder_unseen_but_analysis_seen": True,
            "fast_math_kernel_exp_not_emulated": True,
            "scope_limit": (
                "PyTorch numeric oracle; does not execute FP8 MMA, measure speed, "
                "recapture H2_A8, or estimate final-logit KLD"
            ),
        },
        "provenance": provenance,
        "receipt": {
            "program_sha256": _sha256_file(Path(__file__).resolve()),
            "runtime_qdq_reference": str(
                (
                    PROJECT_ROOT
                    / "evaluation/runtime_overlay/b12x_sqg/_lib/intrinsics.py"
                ).resolve()
            ),
            "runtime_qdq_reference_sha256": _sha256_file(
                PROJECT_ROOT / "evaluation/runtime_overlay/b12x_sqg/_lib/intrinsics.py"
            ),
            "experts": expert_limit,
            "source_bindings_checked": source_bindings_checked,
            "official_bf16_tensor_payloads_rehashed": True,
            "expert_contract_set_sha256": _canonical_json_sha256(expert_receipts),
            "topology": "256 experts, top-8 routing, frozen K3/K4 per tensor",
        },
        "go_no_go_limit": {
            "can_kill_w4a8_for_activation_quality": True,
            "can_greenlight_full_quant_by_itself": False,
            "required_additional_gate": "Test 8c measured GLM-shape W4A8 prefill speed",
            "quality_scope": (
                "A8 incremental layer-77 functional quality versus matched SQG-A16; "
                "final-logit KLD remains unmeasured here"
            ),
            "mcg_scope": (
                "sealed MCG-A16 scalar context only; no matched MCG per-position arm "
                "or MCG tail claim"
            ),
        },
        "roles": role_results,
    }
    validate_result_contract(result)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_render(result), encoding="utf-8")
    digest = _sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(_render(result), flush=True)
    print(f"json: {output}", flush=True)
    print(f"sha256: {digest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
