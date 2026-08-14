#!/usr/bin/env python3
"""Shared updated-QSRT coupled-W4A8 construction primitives.

This module is deliberately source-agnostic at the Python interface.  The
caller supplies a layer runtime whose ``source.load_expert_bf16`` method may
decode the frozen SQG checkpoint; no official BF16 shard is required.  All
candidate construction uses the updated QSRT H512/H128/H128 GLM-SiLU
reparameterization, profile-specific h-A8 H13, fit/calibration-only local H13
and candidate-conditioned down ``(H, B)``, and native direct-E4M3 execution.

The same functions are used by the no-shortcut profile/beta recipe and are
kept small enough to be adopted by the fleet scorer/encoder without a second
numerical implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "kquant"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
FINAL_BINDING_SCHEMA = "glm52-updated-qsrt-coupled-final-profile-binding-v1"
PROFILE_SELECTION_SCHEMA = "glm52-updated-qsrt-coupled-profile-selection-v1"
ALL_TRIPLETS = tuple(
    (gate, up, down)
    for gate in (3, 4)
    for up in (3, 4)
    for down in (3, 4)
)


def validate_triplets(
    triplets: Sequence[Sequence[int]],
) -> tuple[tuple[int, int, int], ...]:
    result = tuple(tuple(int(rate) for rate in item) for item in triplets)
    if not result or len(set(result)) != len(result):
        raise ValueError("coupled recipe requires one or more unique triplets")
    if any(len(item) != 3 or item not in ALL_TRIPLETS for item in result):
        raise ValueError("coupled recipe triplets must lie in the K3/K4 cube")
    return result


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def load_final_recipe_profile(
    runtime: Any,
    *,
    selection_path: Path,
    binding_path: Path,
    layer: int,
) -> tuple[Any, Any, dict[str, Any], float, dict[str, Any]]:
    """Validate and realize the selected no-shortcut layer profile."""

    selection_path = selection_path.resolve()
    binding_path = binding_path.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict) or not isinstance(binding, dict):
        raise TypeError("coupled final profile inputs must be JSON objects")
    unsigned_selection = dict(selection)
    selection_id = unsigned_selection.pop("selection_id", None)
    selected_cell = selection.get("selected_cell")
    arithmetic = selection.get("arithmetic")
    if (
        selection.get("schema") != PROFILE_SELECTION_SCHEMA
        or selection.get("complete") is not True
        or int(selection.get("layer", -1)) != layer
        or not isinstance(selected_cell, Mapping)
        or selection.get("selected_cell_id") != selected_cell.get("cell_id")
        or not isinstance(arithmetic, Mapping)
        or selection_id != _canonical_sha256(unsigned_selection)
        or selection.get("profile_selection_precedes_rate_allocation") is not True
        or selection.get("holdout_used_for_choice") is not False
        or selection.get("no_b300_identity_only_rescue") is not True
        or int(selection.get("mcg_inputs", -1)) != 0
    ):
        raise ValueError("no-shortcut coupled profile selection differs")
    required_arithmetic = {
        "candidate_specific_down_h_b": True,
        "caller_coordinate_corrected": True,
        "private_down_suh_anchored_by_rate": True,
        "shared_output_svh_anchored": True,
        "updated_qsrt_coupled_hadamard": True,
        "activation": "torch.nn.functional.silu(gate) * up",
    }
    if any(arithmetic.get(key) != value for key, value in required_arithmetic.items()):
        raise ValueError("updated-QSRT coupled profile arithmetic differs")
    beta = float(arithmetic.get("beta", np.nan))
    if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("selected coupled beta differs")
    unsigned_binding = dict(binding)
    binding_id = unsigned_binding.pop("binding_id", None)
    if (
        binding.get("schema") != FINAL_BINDING_SCHEMA
        or binding.get("complete") is not True
        or int(binding.get("layer", -1)) != layer
        or float(binding.get("selected_beta", np.nan)) != beta
        or binding.get("profile_selection_id") != selection_id
        or binding.get("profile_selection_sha256") != _sha256_file(selection_path)
        or binding.get("no_b300_owner_speed_rescue") is not True
        or binding_id != _canonical_sha256(unsigned_binding)
        or not _is_sha256(binding.get("beta_choice_id"))
        or not _is_sha256(binding.get("beta_choice_sha256"))
        or (binding.get("bootstrap_reused_byte_for_byte") is True)
        != (beta == 0.0625)
        or (binding.get("profile_reselected_exactly_once") is True)
        != (beta != 0.0625)
    ):
        raise ValueError("no-shortcut coupled final profile binding differs")
    gate_profile, down_profile = realize_profile_cell(runtime, selected_cell)
    evidence = {
        "selection_path": str(selection_path),
        "selection_sha256": _sha256_file(selection_path),
        "selection_id": selection_id,
        "selected_cell_id": str(selection["selected_cell_id"]),
        "draw": int(selected_cell["draw"]),
        "family": str(selected_cell["family"]),
        "final_binding_path": str(binding_path),
        "final_binding_sha256": _sha256_file(binding_path),
        "final_binding_id": binding_id,
        "profile_selection_precedes_rate_allocation": True,
    }
    return gate_profile, down_profile, evidence, beta, binding


def profile_cells(runtime: Any) -> tuple[dict[str, Any], ...]:
    """Reconstruct the complete four-draw/four-family profile panel."""

    from src.fresh_pipeline_runner import _load_scale_evidence, _profiles_for_cell

    scales = _load_scale_evidence(runtime)
    families = tuple(scales.families())
    expected = {
        "identity",
        "aggregate_rms",
        "quarter_rms",
        "inverse_quarter_rms",
    }
    if set(families) != expected:
        raise ValueError("profile scale family census differs")
    result: list[dict[str, Any]] = []
    for draw in range(4):
        for family in (
            "identity",
            "aggregate_rms",
            "quarter_rms",
            "inverse_quarter_rms",
        ):
            gate, down = _profiles_for_cell(
                runtime, scales, draw=draw, family=family
            )
            result.append(
                {
                    "cell_id": f"draw-{draw:02d}__{family}",
                    "draw": draw,
                    "family": family,
                    "gate_up_input_profile": gate.manifest(),
                    "down_output_profile": down.manifest(),
                }
            )
    return tuple(result)


def realize_profile_cell(runtime: Any, cell: Mapping[str, Any]) -> tuple[Any, Any]:
    from src.fresh_pipeline_runner import _load_scale_evidence, _profiles_for_cell
    from src.glm52_fresh_sqg.codec import realize_shared_residual_profile

    scales = _load_scale_evidence(runtime)
    gate, down = _profiles_for_cell(
        runtime,
        scales,
        draw=int(cell["draw"]),
        family=str(cell["family"]),
    )
    if gate.manifest() != cell.get("gate_up_input_profile"):
        raise ValueError("coupled recipe gate profile reconstruction differs")
    if down.manifest() != cell.get("down_output_profile"):
        raise ValueError("coupled recipe down profile reconstruction differs")
    return (
        realize_shared_residual_profile(gate, runtime.device),
        realize_shared_residual_profile(down, runtime.device),
    )


def build_profile_global_h13(
    runtime: Any,
    gate_profile: Any,
    *,
    cell_id: str,
    chunk_rows: int,
) -> tuple[Any, dict[str, Any]]:
    """Build the profile-specific h-A8 global H13 on fit/calibration only."""

    import torch
    from scripts.profile_search_full_w4a8_native import _profile_global_h13
    from src.glm52_fresh_sqg.reference import normalized_hadamard

    device = torch.device(runtime.device)
    hadamard = normalized_hadamard(device=device, dtype=torch.float32, size=128)
    return _profile_global_h13(
        runtime,
        gate_profile,
        hadamard,
        chunk_rows=chunk_rows,
        cell_id=cell_id,
    )


@dataclass
class PreparedCoupledExpert:
    runtime: Any
    expert: int
    triplets: tuple[tuple[int, int, int], ...]
    weights: Any
    permutation: Any
    spec: Any
    execution: Any
    coupled_h13: Any
    routed: Any
    calibration_mask: np.ndarray
    source_gpu: dict[str, Any]
    sources: dict[str, Any]
    encoded_upstream: dict[str, dict[int, Any]]
    native_pairs: dict[tuple[int, int], dict[str, Any]]
    preliminary_down: dict[tuple[int, int], dict[int, Any]]
    down_configs: dict[int, Any]
    preliminary_h2_evidence: dict[str, Any]
    coupled_h13_evidence: dict[str, Any]
    gate_profile: Any
    down_profile: Any
    device: Any
    hadamard: Any
    kquant_runtime: Any
    lut: dict[int, Any]
    chunk_rows: int

    def release(self) -> None:
        import torch

        for name in (
            "weights",
            "permutation",
            "spec",
            "execution",
            "coupled_h13",
            "routed",
            "source_gpu",
            "sources",
            "encoded_upstream",
            "native_pairs",
            "preliminary_down",
            "down_configs",
            "kquant_runtime",
            "lut",
        ):
            setattr(self, name, None)
        gc.collect()
        torch.cuda.empty_cache()


def prepare_coupled_expert(
    runtime: Any,
    *,
    expert: int,
    gate_profile: Any,
    down_profile: Any,
    global_h13: Any,
    global_evidence: Mapping[str, Any],
    triplets: Sequence[Sequence[int]],
    qsrt_root: Path,
    chunk_rows: int,
    intermediate_draw: int = 0,
) -> PreparedCoupledExpert:
    """Encode the beta-neutral upstream candidates and private down anchors."""

    triplets = validate_triplets(triplets)
    qsrt_root = qsrt_root.resolve()
    if str(qsrt_root) not in sys.path:
        sys.path.insert(0, str(qsrt_root))

    import torch
    from kquant.sqg_e4m3 import sqg_xor_cheb_t12_bytes
    from qsrt.qsrt_coupled import (
        CoupledHadamardSpec,
        coupled_execution,
        encode_coupled_weights,
    )
    from scripts.encode_coupled_mixed_rate_shard import (
        _build_coupled_h2,
        _synthetic_config,
        _transform_hessian,
    )
    from scripts.score_sqg_w4a8_triplet_candidates import (
        H13_CONSTRUCTION,
        _build_expert_h13,
        _native_projection,
    )
    from src.calibration_hessian import apply_frozen_h2_shrinkage
    from src.fresh_pipeline_common import canonical_sha256, derive_seed
    from src.fresh_pipeline_runner import _load_permutation
    from src.glm52_fresh_sqg import (
        DenseHessian,
        SyntheticTensorBinding,
        UniformSQGConfig,
        apply_glm_expert_permutation,
        encode_uniform_sqg,
        load_kquant_runtime,
        prepare_dense_h_session,
    )
    from src.glm52_fresh_sqg.reference import normalized_hadamard, tensor_sha256

    if not 0 <= expert < 256:
        raise ValueError("expert must lie in [0,256)")
    if not 0 <= intermediate_draw < 8:
        raise ValueError("intermediate draw must lie in [0,8)")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    device = torch.device(runtime.device)
    if device.type != "cuda":
        raise ValueError("coupled recipe construction requires CUDA")
    hadamard = normalized_hadamard(device=device, dtype=torch.float32, size=128)
    kquant_runtime = load_kquant_runtime(
        runtime.paths.kquant_root, runtime.paths.exllamav3_root
    )
    needed_bits = {
        projection: tuple(sorted({item[index] for item in triplets}))
        for index, projection in enumerate(PROJECTIONS)
    }
    lut = {
        bits: sqg_xor_cheb_t12_bytes(bits).to(device).contiguous()
        for bits in sorted({rate for item in triplets for rate in item})
    }

    weights = runtime.source.load_expert_bf16(
        runtime.layer, expert, device=device
    )
    permutation = _load_permutation(runtime, expert)
    expert_h13, h13_evidence, routed, calibration_mask = _build_expert_h13(
        runtime,
        expert=expert,
        global_h13=global_h13,
        global_evidence=global_evidence,
        gate_profile=gate_profile,
        hadamard=hadamard,
        chunk_rows=chunk_rows,
        device=device,
    )
    permuted = apply_glm_expert_permutation(
        weights.gate_hf, weights.up_hf, weights.down_hf, permutation
    )
    spec = CoupledHadamardSpec(
        residual_block_size=512,
        preactivation_block_size=128,
        postactivation_block_size=128,
        residual_draw=0,
        intermediate_draw=intermediate_draw,
        activation="silu",
    )
    coupled_gpu = encode_coupled_weights(
        tuple(value.to(device=device, dtype=torch.float32) for value in permuted),
        spec,
    )
    execution = coupled_execution(coupled_gpu, spec)
    sources = {
        "gate_proj": coupled_gpu[0].T.cpu().contiguous(),
        "up_proj": coupled_gpu[1].T.cpu().contiguous(),
        "down_proj": coupled_gpu[2].T.cpu().contiguous(),
    }
    coupled_h13 = _transform_hessian(
        expert_h13,
        execution,
        construction=H13_CONSTRUCTION,
        canonical_sha256=canonical_sha256,
        DenseHessian=DenseHessian,
        tensor_sha256=tensor_sha256,
        device=device,
    )
    configs: dict[str, dict[int, Any]] = {}
    for projection in PROJECTIONS:
        configs[projection] = {
            bits: _synthetic_config(
                runtime,
                projection,
                bits,
                gate_profile if projection != "down_proj" else down_profile,
                sources[projection],
                draw=intermediate_draw,
                expert=expert,
                derive_seed=derive_seed,
                tensor_sha256=tensor_sha256,
                SyntheticTensorBinding=SyntheticTensorBinding,
                UniformSQGConfig=UniformSQGConfig,
            )
            for bits in needed_bits[projection]
        }
    h13_session = prepare_dense_h_session(
        coupled_h13, configs["gate_proj"][needed_bits["gate_proj"][0]]
    )
    encoded_upstream: dict[str, dict[int, Any]] = {}
    native_upstream: dict[str, dict[int, Any]] = {}
    for projection in ("gate_proj", "up_proj"):
        encoded_upstream[projection] = {}
        native_upstream[projection] = {}
        for bits in needed_bits[projection]:
            encoded = encode_uniform_sqg(
                sources[projection],
                coupled_h13,
                configs[projection][bits],
                runtime=kquant_runtime,
                dense_h_session=h13_session,
            )
            encoded_upstream[projection][bits] = encoded
            native_upstream[projection][bits] = _native_projection(
                encoded, bits=bits, lut=lut[bits], device=device
            )
    pairs = tuple(sorted({item[:2] for item in triplets}))
    native_pairs = {
        pair: {
            "gate_proj": native_upstream["gate_proj"][pair[0]],
            "up_proj": native_upstream["up_proj"][pair[1]],
        }
        for pair in pairs
    }
    preliminary_down: dict[tuple[int, int], dict[int, Any]] = {}
    preliminary_h2_evidence: dict[str, Any] = {}
    # The shared down fitter accepts a rectangular pair x down-rate grid.
    # Materialize every requested down rate for every requested upstream pair;
    # callers still score and bind only their preregistered triplets.
    down_bit_domain = needed_bits["down_proj"]
    for pair in pairs:
        pair_down_bits = down_bit_domain
        h2, evidence = _build_coupled_h2(
            runtime,
            execution,
            encoded_upstream["gate_proj"][pair[0]].reconstructed_exl,
            encoded_upstream["up_proj"][pair[1]].reconstructed_exl,
            expert=expert,
            chunk_rows=chunk_rows,
            apply_frozen_h2_shrinkage=apply_frozen_h2_shrinkage,
            canonical_sha256=canonical_sha256,
            DenseHessian=DenseHessian,
            tensor_sha256=tensor_sha256,
            row_mask=calibration_mask,
            fit_scope="fit/calibration",
        )
        preliminary_h2_evidence[f"k{pair[0]}_k{pair[1]}"] = evidence
        session = prepare_dense_h_session(
            h2, configs["down_proj"][pair_down_bits[0]]
        )
        preliminary_down[pair] = {
            bits: encode_uniform_sqg(
                sources["down_proj"],
                h2,
                configs["down_proj"][bits],
                runtime=kquant_runtime,
                dense_h_session=session,
            )
            for bits in pair_down_bits
        }
    source_gpu = {
        "gate_proj": weights.gate_hf.T.float().to(device),
        "up_proj": weights.up_hf.T.float().to(device),
        "down_proj": weights.down_hf.T.float().to(device),
    }
    return PreparedCoupledExpert(
        runtime=runtime,
        expert=expert,
        triplets=triplets,
        weights=weights,
        permutation=permutation,
        spec=spec,
        execution=execution,
        coupled_h13=coupled_h13,
        routed=routed,
        calibration_mask=calibration_mask,
        source_gpu=source_gpu,
        sources=sources,
        encoded_upstream=encoded_upstream,
        native_pairs=native_pairs,
        preliminary_down=preliminary_down,
        down_configs=configs["down_proj"],
        preliminary_h2_evidence=preliminary_h2_evidence,
        coupled_h13_evidence={
            "canonical": h13_evidence,
            "transformed_evidence_id": coupled_h13.evidence_id,
            "updated_qsrt_coupled": True,
        },
        gate_profile=gate_profile,
        down_profile=down_profile,
        device=device,
        hadamard=hadamard,
        kquant_runtime=kquant_runtime,
        lut=lut,
        chunk_rows=chunk_rows,
    )


def fit_coupled_down(
    prepared: PreparedCoupledExpert,
    *,
    beta: float,
) -> tuple[dict[tuple[int, int], dict[int, Any]], dict[tuple[int, int], dict[int, Any]], dict[tuple[int, int], dict[int, dict[str, Any]]]]:
    """Fit and encode the requested candidate-conditioned down grid."""

    if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be finite and lie in [0,1]")
    from scripts.score_coupled_tail_triplet_candidates import (
        _fit_and_encode_coupled_down_grid,
    )
    from scripts.score_glm52_w4a8_activation_quality import (
        apply_output_transform,
        native_label_gemm,
        prepare_down_operand,
        prepare_gate_up_operand,
    )
    from scripts.score_sqg_w4a8_triplet_candidates import _native_projection
    from scripts.w4a8_cross_term import (
        CrossTermStatistics,
        effective_canonical_operand_from_quantized_transform,
        kquant_prefinalize_operand_from_label_operand,
        shrink_cross_term_objective,
    )
    from scripts.w4a8_stable_solve import solve_with_minimal_official_prior
    from src.calibration_hessian import apply_frozen_h2_shrinkage
    from src.fresh_pipeline_common import canonical_sha256
    from src.glm52_fresh_sqg import (
        DenseHessian,
        SharedResidualProfile,
        SyntheticTensorBinding,
        encode_uniform_sqg,
    )
    from src.glm52_fresh_sqg.codec import realize_shared_residual_profile
    from src.glm52_fresh_sqg.reference import tensor_sha256

    down_bits = tuple(sorted({item[2] for item in prepared.triplets}))
    return _fit_and_encode_coupled_down_grid(
        runtime=prepared.runtime,
        execution=prepared.execution,
        routed=prepared.routed,
        calibration_mask=prepared.calibration_mask,
        native_pairs=prepared.native_pairs,
        preliminary_down=prepared.preliminary_down,
        down_configs=prepared.down_configs,
        official_down_exl=prepared.sources["down_proj"],
        source_gpu=prepared.source_gpu,
        beta=float(beta),
        expert=prepared.expert,
        device=prepared.device,
        hadamard=prepared.hadamard,
        chunk_rows=prepared.chunk_rows,
        kquant_runtime=prepared.kquant_runtime,
        lut=prepared.lut,
        prepare_down_operand=prepare_down_operand,
        prepare_gate_up_operand=prepare_gate_up_operand,
        native_label_gemm=native_label_gemm,
        apply_output_transform=apply_output_transform,
        apply_frozen_h2_shrinkage=apply_frozen_h2_shrinkage,
        canonical_sha256=canonical_sha256,
        tensor_sha256=tensor_sha256,
        DenseHessian=DenseHessian,
        SharedResidualProfile=SharedResidualProfile,
        SyntheticTensorBinding=SyntheticTensorBinding,
        encode_uniform_sqg=encode_uniform_sqg,
        realize_shared_residual_profile=realize_shared_residual_profile,
        effective_canonical_operand_from_quantized_transform=(
            effective_canonical_operand_from_quantized_transform
        ),
        kquant_prefinalize_operand_from_label_operand=(
            kquant_prefinalize_operand_from_label_operand
        ),
        CrossTermStatistics=CrossTermStatistics,
        shrink_cross_term_objective=shrink_cross_term_objective,
        solve_with_minimal_official_prior=solve_with_minimal_official_prior,
        native_projection=_native_projection,
        down_bits_domain=down_bits,
    )


def score_coupled_candidates(
    prepared: PreparedCoupledExpert,
    native_down: Mapping[tuple[int, int], Mapping[int, Any]],
    *,
    score_role: str,
    return_rows: bool = False,
) -> dict[str, Any]:
    """Score requested triplets on selection, holdout, or fit/allocation."""

    import torch
    import torch.nn.functional as F
    from scripts.score_coupled_tail_triplet_candidates import (
        _execute_down,
        _execute_upstream,
    )
    from scripts.score_glm52_w4a8_activation_quality import (
        apply_output_transform,
        native_label_gemm,
        prepare_down_operand,
        prepare_gate_up_operand,
    )
    from scripts.score_sqg_w4a8_triplet_candidates import fit_subfold_mask

    if score_role == "fit/allocation":
        routed = prepared.routed
        mask = fit_subfold_mask(
            routed.document_epochs,
            layer=prepared.runtime.layer,
            subfold="allocation",
        )
        rows = routed.row_indices[mask]
        weights = routed.gate_square_weights[torch.from_numpy(mask)]
        documents = routed.document_epochs[torch.from_numpy(mask)]
    elif score_role in {"selection", "holdout"}:
        routed = prepared.runtime.capture.routed_rows(prepared.expert, score_role)
        rows = routed.row_indices
        weights = routed.gate_square_weights
        documents = routed.document_epochs
    else:
        raise ValueError("score_role must be selection, holdout, or fit/allocation")
    if rows.size == 0:
        raise ValueError(f"expert {prepared.expert}: {score_role} has no rows")
    row_sse = torch.empty(
        (len(prepared.triplets), rows.size), dtype=torch.float64, device="cpu"
    )
    row_reference = torch.empty(rows.size, dtype=torch.float64, device="cpu")
    for begin in range(0, rows.size, prepared.chunk_rows):
        end = min(rows.size, begin + prepared.chunk_rows)
        hidden = prepared.runtime.capture.load_hidden(
            rows[begin:end], device=prepared.device, dtype=torch.float32
        )
        gate = torch.matmul(hidden.float(), prepared.source_gpu["gate_proj"])
        up = torch.matmul(hidden.float(), prepared.source_gpu["up_proj"])
        teacher = torch.matmul(
            F.silu(gate) * up, prepared.source_gpu["down_proj"]
        )
        middle = _execute_upstream(
            hidden,
            prepared.execution,
            prepared.native_pairs,
            prepared.hadamard,
            prepare_gate_up_operand,
            native_label_gemm,
            apply_output_transform,
        )
        importance = weights[begin:end].to(
            device=prepared.device, dtype=torch.float32
        )
        row_reference[begin:end] = (
            teacher.float().square().sum(dim=1) * importance
        ).double().cpu()
        for index, (gate_bits, up_bits, down_bits) in enumerate(
            prepared.triplets
        ):
            pair = (gate_bits, up_bits)
            candidate = _execute_down(
                middle[pair],
                native_down[pair][down_bits],
                prepared.execution,
                prepared.hadamard,
                prepare_down_operand,
                native_label_gemm,
                apply_output_transform,
            )
            row_sse[index, begin:end] = (
                (candidate.float() - teacher.float()).square().sum(dim=1)
                * importance
            ).double().cpu()
        del hidden, gate, up, teacher, middle, importance
    scores = row_sse.numpy()
    reference = row_reference.numpy()
    triplet_sse = {
        str(item): float(scores[index].sum(dtype=np.float64))
        for index, item in enumerate(prepared.triplets)
    }
    total_sse = float(scores.mean(axis=0).sum(dtype=np.float64))
    total_reference = float(reference.sum(dtype=np.float64))
    result: dict[str, Any] = {
        "role": score_role,
        "rows": int(rows.size),
        "documents": int(torch.unique(documents).numel()),
        "triplet_sse": triplet_sse,
        "mean_balanced_triplet_sse": total_sse,
        "teacher_energy": total_reference,
        "relative_error": total_sse / total_reference,
    }
    if return_rows:
        result["row_indices"] = np.asarray(rows, dtype=np.int64)
        result["document_epochs"] = np.asarray(documents, dtype=np.int64)
        result["row_sse"] = scores
        result["row_reference_energy"] = reference
    return result


def candidate_payloads(
    prepared: PreparedCoupledExpert,
    encoded_down: Mapping[tuple[int, int], Mapping[int, Any]],
) -> dict[str, dict[str, str]]:
    from scripts.score_sqg_w4a8_triplet_candidates import (
        _projection_payload_sha256,
    )

    result: dict[str, dict[str, str]] = {}
    for gate_bits, up_bits, down_bits in prepared.triplets:
        result[str((gate_bits, up_bits, down_bits))] = {
            "gate_proj": _projection_payload_sha256(
                prepared.encoded_upstream["gate_proj"][gate_bits]
            ),
            "up_proj": _projection_payload_sha256(
                prepared.encoded_upstream["up_proj"][up_bits]
            ),
            "down_proj": _projection_payload_sha256(
                encoded_down[(gate_bits, up_bits)][down_bits]
            ),
        }
    return result
