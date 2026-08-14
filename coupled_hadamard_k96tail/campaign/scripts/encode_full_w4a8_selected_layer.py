#!/usr/bin/env python3
"""Re-encode and materialize one selected full-W4A8 GLM layer.

This is the production consumer of the v3 realized-W4A8 triplet allocation.
Candidate scoring deliberately discards candidate bytes after sealing their
hashes.  A worker therefore re-encodes exactly one selected K3/K4 gate, up and
down tensor per expert, closes the selected H13, execution and down-(H,B)
evidence identities (plus the independently encoded down payload), and writes
an ordinary resumable expert mini-shard.  Finalize mode publishes the disjoint
mini-shards and assembles the canonical layer artifact.

No selection or holdout row is used by this program.  Gate/up use the frozen
profile-specific exact-h-A8 H13.  Down is rebuilt from the selected upstream
pair with exact h-A8, GLM SiLU(gate)*up, act-A8 and the selected-rate anchored
private suh.  Its source is a production W4A8Derived binding to the official
BF16 parent plus fit-only (H,B) evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "kquant"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts.score_glm52_w4a8_activation_quality import (  # noqa: E402
    NativeProjection,
    apply_gate_up_output_transform_silu,
    prepare_down_operand,
    prepare_gate_up_operand,
    native_label_gemm,
)
from scripts.score_sqg_w4a8_triplet_candidates import (  # noqa: E402
    DERIVED_H2_CONSTRUCTION,
    HIDDEN,
    H13_CONSTRUCTION,
    INTERMEDIATE,
    NUM_EXPERTS,
    PRELIMINARY_H2_CONSTRUCTION,
    _build_expert_h13,
    _config_at_rate,
    _load_global_h13,
    _native_projection,
    _profile_from_selection,
    _projection_payload_sha256,
    _teacher_output,
    _validate_expert_record,
    validate_coupled_scale_choice,
)
from scripts.coupled_gate_up_scale_selector import (  # noqa: E402
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
    BIT_UNITS,
    FULL_W4A8_ENDPOINT,
    K3_COUNT,
    K4_COUNT,
    TRIPLET_ALLOCATION_SCHEMA,
    TRIPLET_LOSS_DEFINITION,
    TRIPLET_PROJECTIONS,
    TRIPLET_SCORE_SCHEMA,
    _validate_triplet_scores,
    canonical_json_bytes,
    sha256_file,
)


FINAL_CONTRACT_SCHEMA = "glm52-sqg-full-w4a8-selected-encode-contract-v1"
PROGRESS_SCHEMA = "glm52-sqg-full-w4a8-selected-shard-progress-v1"
FINALIZE_SCHEMA = "glm52-sqg-full-w4a8-selected-layer-finalize-v1"
PURPOSE = "final_treatment"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} is not a lowercase SHA256")
    return text


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def selected_candidate_for_expert(
    allocation: Mapping[str, Any],
    scores: Mapping[str, Any],
    *,
    layer: int,
    expert: int,
    allow_historical_pre_coupled_scale: bool = False,
) -> dict[str, Any]:
    """Return one selected record only after allocation/score exact closure."""

    if not 0 <= expert < NUM_EXPERTS:
        raise ValueError("expert must lie in [0,255]")
    assignment = allocation.get("expert_assignments", {}).get(str(expert))
    score_expert = scores.get("experts", {}).get(str(expert))
    if not isinstance(assignment, Mapping) or not isinstance(score_expert, Mapping):
        raise ValueError(f"layer {layer} expert {expert}: selected record is absent")
    candidates = score_expert.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"layer {layer} expert {expert}: candidate list is absent")
    candidate_id = _require_sha256(
        assignment.get("candidate_id"), f"layer {layer} expert {expert} candidate ID"
    )
    matches = [item for item in candidates if item.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"layer {layer} expert {expert}: selected candidate is not unique")
    selected = dict(matches[0])
    if assignment.get("candidate_record") != selected:
        raise ValueError(f"layer {layer} expert {expert}: embedded candidate record differs")
    if assignment.get("rates") != selected.get("rates"):
        raise ValueError(f"layer {layer} expert {expert}: selected rates differ")
    if float(assignment.get("realized_w4a8_fit_sse", math.nan)) != float(
        selected.get("realized_w4a8_fit_sse", math.nan)
    ):
        raise ValueError(f"layer {layer} expert {expert}: selected loss differs")
    rates = selected.get("rates")
    if not isinstance(rates, Mapping) or set(rates) != set(TRIPLET_PROJECTIONS):
        raise ValueError(f"layer {layer} expert {expert}: selected rate map differs")
    for projection in TRIPLET_PROJECTIONS:
        bits = int(rates[projection])
        prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        if bits not in (3, 4) or allocation.get("bit_map", {}).get(prefix) != bits:
            raise ValueError(f"layer {layer} expert {expert}: {projection} allocation differs")
    if "coupled_scale_choice" not in selected:
        if not allow_historical_pre_coupled_scale:
            raise ValueError(
                f"layer {layer} expert {expert}: coupled scale choice is absent"
            )
    else:
        choice = validate_coupled_scale_choice(
            selected["coupled_scale_choice"],
            gate_bits=int(rates["gate_proj"]),
            up_bits=int(rates["up_proj"]),
        )
        payloads = selected.get("payload_sha256", {})
        if (
            payloads.get("gate_proj") != choice["gate"]["payload_sha256"]
            or payloads.get("up_proj") != choice["up"]["payload_sha256"]
        ):
            raise ValueError(
                f"layer {layer} expert {expert}: selected scale payload differs"
            )
    return selected


@dataclass(frozen=True)
class FinalSelectedContract:
    layer: int
    allocation: Mapping[str, Any]
    scores: Mapping[str, Any]
    profile_selection: Mapping[str, Any]
    allocation_path: Path
    scores_path: Path
    profile_selection_path: Path
    contract_id: str
    run_id: str
    historical_pre_coupled_scale: bool = False

    @property
    def bit_map(self) -> dict[str, int]:
        return {str(name): int(bits) for name, bits in self.allocation["bit_map"].items()}

    @property
    def beta(self) -> float:
        return float(self.scores["down"]["beta"])

    def selected(self, expert: int) -> dict[str, Any]:
        return selected_candidate_for_expert(
            self.allocation,
            self.scores,
            layer=self.layer,
            expert=expert,
            allow_historical_pre_coupled_scale=self.historical_pre_coupled_scale,
        )


@dataclass(frozen=True)
class CoupledOriginBatchMember:
    """One scorer candidate in its original physical CUDA batch position."""

    projection: str
    bits: int
    scale: float
    scale_hex: str
    candidate_id: str
    receipt: Mapping[str, Any]


def load_scoring_expert_record(
    contract: FinalSelectedContract, expert: int
) -> dict[str, Any]:
    """Load the score producer's authenticated per-expert replay receipt."""

    raw_receipts = contract.scores.get("expert_receipts")
    if not isinstance(raw_receipts, list):
        raise ValueError("triplet score manifest lacks expert replay receipts")
    matches = [
        item
        for item in raw_receipts
        if isinstance(item, Mapping) and int(item.get("expert", -1)) == expert
    ]
    if len(matches) != 1:
        raise ValueError(f"expert {expert}: score replay receipt is not unique")
    receipt = matches[0]
    path = Path(str(receipt.get("path", "")))
    if not path.is_absolute():
        path = contract.scores_path.parent / path
    path = path.resolve()
    if sha256_file(path) != _require_sha256(
        receipt.get("sha256"), f"expert {expert} score receipt SHA256"
    ):
        raise ValueError(f"expert {expert}: score replay file hash differs")
    value = _read_json(path)
    _validate_expert_record(value, layer=contract.layer, expert=expert)
    if (
        value.get("score_id") != receipt.get("score_id")
        or value.get("candidates")
        != contract.scores.get("experts", {}).get(str(expert), {}).get("candidates")
        or value.get("profile_selection") != contract.scores.get("profile_selection")
    ):
        raise ValueError(f"expert {expert}: score replay receipt content differs")
    return value


def reconstruct_coupled_origin_batches(
    expert_record: Mapping[str, Any],
) -> tuple[tuple[CoupledOriginBatchMember, ...], ...]:
    """Rebuild the scorer's first-seen fixed-candidate batch schedule.

    The scorer seals every coarse/fine scale in iteration order plus the
    projection-candidate census and physical batch sizes.  Those fields are
    sufficient to recover each original KQuant batch without candidate bytes.
    """

    evidence = expert_record.get("coupled_scale_selection")
    if not isinstance(evidence, Mapping) or evidence.get("complete") is not True:
        raise ValueError("coupled scale replay evidence is absent")
    raw_candidates = evidence.get("projection_candidates")
    raw_pairs = evidence.get("rate_pair_results")
    cache = evidence.get("candidate_cache")
    if (
        not isinstance(raw_candidates, list)
        or not isinstance(raw_pairs, list)
        or not isinstance(cache, Mapping)
        or cache.get("encode_mode") != "batch_by_search_stage_v1"
        or cache.get("overlap_reused") is not True
    ):
        raise ValueError("coupled scale replay schedule differs")

    by_key: dict[tuple[str, int, str], CoupledOriginBatchMember] = {}
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            raise TypeError("coupled projection candidate is not an object")
        projection = str(raw.get("projection"))
        bits = int(raw.get("bits", -1))
        scale = float(raw.get("scale", math.nan))
        scale_hex = str(raw.get("scale_hex", ""))
        candidate_id = _require_sha256(
            raw.get("candidate_id"), "coupled projection candidate ID"
        )
        if (
            projection not in ("gate_proj", "up_proj")
            or bits not in (3, 4)
            or not math.isfinite(scale)
            or scale <= 0.0
            or scale.hex() != scale_hex
        ):
            raise ValueError("coupled projection candidate identity differs")
        key = (projection, bits, scale_hex)
        if key in by_key:
            raise ValueError("coupled projection candidate key is duplicated")
        by_key[key] = CoupledOriginBatchMember(
            projection=projection,
            bits=bits,
            scale=scale,
            scale_hex=scale_hex,
            candidate_id=candidate_id,
            receipt=dict(raw),
        )

    contract_pairs = evidence.get("search_contract", {}).get("rate_pairs")
    expected_pairs = [
        [int(item.get("gate_bits", -1)), int(item.get("up_bits", -1))]
        for item in raw_pairs
        if isinstance(item, Mapping)
    ]
    if contract_pairs != expected_pairs or len(expected_pairs) != 4:
        raise ValueError("coupled rate-pair replay order differs")

    seen: set[tuple[str, int, str]] = set()
    batches: list[tuple[CoupledOriginBatchMember, ...]] = []
    for pair in raw_pairs:
        gate_bits = int(pair["gate_bits"])
        up_bits = int(pair["up_bits"])
        grids = pair.get("grids")
        if not isinstance(grids, Mapping):
            raise ValueError("coupled replay grids are absent")
        for stage in ("coarse", "fine"):
            requests: list[tuple[str, int, str]] = []
            for projection, bits, name in (
                ("gate_proj", gate_bits, f"{stage}_gate"),
                ("up_proj", up_bits, f"{stage}_up"),
            ):
                grid = grids.get(name)
                values = grid.get("values") if isinstance(grid, Mapping) else None
                if not isinstance(values, list):
                    raise ValueError("coupled replay grid values are absent")
                for value in values:
                    if not isinstance(value, Mapping):
                        raise TypeError("coupled replay grid value is not an object")
                    scale = float(value.get("value", math.nan))
                    scale_hex = str(value.get("hex", ""))
                    if not math.isfinite(scale) or scale.hex() != scale_hex:
                        raise ValueError("coupled replay grid scale differs")
                    requests.append((projection, bits, scale_hex))
            missing = tuple(key for key in requests if key not in seen)
            seen.update(missing)
            if missing:
                try:
                    batches.append(tuple(by_key[key] for key in missing))
                except KeyError as exc:
                    raise ValueError(
                        "coupled replay grid lacks a projection candidate"
                    ) from exc

    recorded_sizes = cache.get("batch_call_sizes")
    if (
        recorded_sizes != [len(batch) for batch in batches]
        or int(cache.get("unique_projection_encodes", -1)) != len(by_key)
        or seen != set(by_key)
    ):
        raise ValueError("coupled replay batch census differs")
    return tuple(batches)


def selected_origin_batches(
    expert_record: Mapping[str, Any], selected: Mapping[str, Any]
) -> tuple[
    tuple[tuple[CoupledOriginBatchMember, ...], tuple[int, ...]], ...
]:
    """Return only origin batches containing the frozen selected gate/up."""

    rates = selected["rates"]
    choice = validate_coupled_scale_choice(
        selected["coupled_scale_choice"],
        gate_bits=int(rates["gate_proj"]),
        up_bits=int(rates["up_proj"]),
    )
    wanted = {
        str(choice["gate"]["candidate_id"]),
        str(choice["up"]["candidate_id"]),
    }
    result = []
    found: set[str] = set()
    for batch in reconstruct_coupled_origin_batches(expert_record):
        indices = tuple(
            index for index, member in enumerate(batch) if member.candidate_id in wanted
        )
        if indices:
            found.update(batch[index].candidate_id for index in indices)
            result.append((batch, indices))
    if found != wanted or not 1 <= len(result) <= 2:
        raise ValueError("selected coupled candidates lack unique origin batches")
    return tuple(result)


def load_final_selected_contract(
    *,
    layer: int,
    allocation_path: Path,
    scores_path: Path,
    profile_selection_path: Path,
    allow_historical_pre_coupled_scale: bool = False,
) -> FinalSelectedContract:
    """Fail closed over every allocation, score and profile-selection edge."""

    allocation_path = allocation_path.resolve()
    scores_path = scores_path.resolve()
    profile_selection_path = profile_selection_path.resolve()
    allocation = _read_json(allocation_path)
    scores = _read_json(scores_path)
    selection = _read_json(profile_selection_path)
    if (
        allocation.get("schema") != TRIPLET_ALLOCATION_SCHEMA
        or allocation.get("complete") is not True
        or int(allocation.get("layer", -1)) != layer
        or allocation.get("method")
        != "sqg_native_realized_full_w4a8_triplet_exact_k4_dp"
        or allocation.get("mcg_rate_map_used_for_choice") is not False
        or allocation.get("mcg_payloads_or_transforms_reused") is not False
    ):
        raise ValueError("v3 selected allocation contract differs")
    body = dict(allocation)
    allocation_id = body.pop("allocation_id", None)
    if allocation_id != canonical_sha256(body):
        raise ValueError("v3 allocation ID differs")
    score_binding = allocation.get("score_binding")
    if (
        not isinstance(score_binding, Mapping)
        or score_binding.get("schema") != TRIPLET_SCORE_SCHEMA
        or score_binding.get("loss_definition") != TRIPLET_LOSS_DEFINITION
        or score_binding.get("activation_endpoint") != FULL_W4A8_ENDPOINT
        or score_binding.get("sha256") != sha256_file(scores_path)
    ):
        raise ValueError("v3 allocation score binding differs")
    _validate_triplet_scores(layer, scores)
    score_body = dict(scores)
    score_id = score_body.pop("score_manifest_id", None)
    if score_id != canonical_sha256(score_body):
        raise ValueError("triplet score manifest ID differs")
    if (
        scores.get("profile_selection", {}).get("sha256")
        != sha256_file(profile_selection_path)
        or scores.get("profile_selection", {}).get("selection_id")
        != selection.get("selection_id", selection.get("selected_cell_id"))
    ):
        raise ValueError("triplet scores/profile selection binding differs")
    if (
        selection.get("complete") is not True
        or selection.get("holdout_used_for_choice") is not False
        or int(selection.get("mcg_inputs", -1)) != 0
    ):
        raise ValueError("profile selection is not frozen zero-MCG evidence")
    bit_map = allocation.get("bit_map")
    if not isinstance(bit_map, Mapping):
        raise TypeError("v3 allocation bit map is absent")
    values = tuple(int(value) for value in bit_map.values())
    if (len(values), values.count(3), values.count(4), sum(values)) != (
        NUM_EXPERTS * 3,
        K3_COUNT,
        K4_COUNT,
        BIT_UNITS,
    ):
        raise ValueError("v3 allocation exact layer budget differs")
    for expert in range(NUM_EXPERTS):
        selected_candidate_for_expert(
            allocation,
            scores,
            layer=layer,
            expert=expert,
            allow_historical_pre_coupled_scale=allow_historical_pre_coupled_scale,
        )
    material = {
        "schema": FINAL_CONTRACT_SCHEMA,
        "layer": layer,
        "allocation_sha256": sha256_file(allocation_path),
        "allocation_id": allocation_id,
        "triplet_scores_sha256": sha256_file(scores_path),
        "triplet_score_manifest_id": score_id,
        "profile_selection_sha256": sha256_file(profile_selection_path),
        "profile_selection_id": selection.get(
            "selection_id", selection.get("selected_cell_id")
        ),
        "activation_endpoint": FULL_W4A8_ENDPOINT,
        "selection_rows_used_for_encoding": False,
        "holdout_rows_used": False,
        "mcg_inputs": 0,
        "historical_pre_coupled_scale_compatibility": bool(
            allow_historical_pre_coupled_scale
        ),
    }
    contract_id = canonical_sha256(material)
    return FinalSelectedContract(
        layer=layer,
        allocation=allocation,
        scores=scores,
        profile_selection=selection,
        allocation_path=allocation_path,
        scores_path=scores_path,
        profile_selection_path=profile_selection_path,
        contract_id=contract_id,
        run_id=f"glm52-full-w4a8-{layer:03d}-{contract_id[:16]}",
        historical_pre_coupled_scale=bool(allow_historical_pre_coupled_scale),
    )


def add_derived_parent_compatibility_aliases(encoded: Any) -> None:
    """Expose immutable parent hashes to the ordinary expert-shard validator.

    These are aliases, not a source-kind downgrade: the authoritative source
    remains ``official_bf16_w4a8_derived_fit_target`` and retains the complete
    execution/H,B/target lineage.
    """

    source = encoded.manifest.get("source")
    if not isinstance(source, dict) or source.get("kind") != (
        "official_bf16_w4a8_derived_fit_target"
    ):
        raise ValueError("down tensor does not carry W4A8Derived lineage")
    parent = source.get("official_bf16_parent")
    if not isinstance(parent, Mapping):
        raise ValueError("W4A8Derived source lacks its official parent")
    aliases = {
        "repository_id": parent.get("repository_id"),
        "revision": parent.get("revision"),
        "shard_name": parent.get("shard_name"),
        "shard_sha256": parent.get("shard_sha256"),
        "tensor_payload_sha256": parent.get("tensor_payload_sha256"),
    }
    if any(value is None for value in aliases.values()):
        raise ValueError("W4A8Derived parent identity is incomplete")
    for key, value in aliases.items():
        existing = source.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"W4A8Derived parent alias drift: {key}")
        source[key] = value


def validate_selected_upstream_reencode(
    encoded: Any,
    native: NativeProjection,
    receipt: Mapping[str, Any],
) -> None:
    """Close a selected-only re-encode against frozen scale-search bytes."""

    from src.glm52_fresh_sqg.reference import tensor_sha256

    scale = float(receipt["scale"])
    transform = encoded.manifest.get("transform", {})
    search = transform.get("global_scale_search", {})
    observed = {
        "trellis_sha256": tensor_sha256(encoded.trellis),
        "suh_sha256": tensor_sha256(encoded.suh),
        "svh_sha256": tensor_sha256(encoded.svh),
        "decoded_label_sha256": tensor_sha256(native.weight),
    }
    expected = {key: receipt[key] for key in observed}
    if (
        float(transform.get("global_scale", math.nan)).hex() != scale.hex()
        or search.get("policy") != "activation_parametric_coupled_v1"
        or search.get("evidence_id") != receipt["override_evidence_id"]
        or observed != expected
    ):
        raise RuntimeError("selected coupled upstream re-encode receipt differs")


def replay_selected_upstream_origin_batches(
    *,
    runtime: Any,
    weights: Any,
    permutation: Any,
    gate_profile: Any,
    down_profile: Any,
    h13: Any,
    expert_record: Mapping[str, Any],
    selected: Mapping[str, Any],
    kquant_runtime: Any,
) -> tuple[Any, Any]:
    """Reproduce selected gate/up bytes in their scorer-origin CUDA batches."""

    from src.glm52_fresh_sqg import (
        UniformSQGCandidateRequest,
        encode_uniform_sqg_candidate_batch,
        prepare_dense_h_session,
    )

    rates = {name: int(selected["rates"][name]) for name in TRIPLET_PROJECTIONS}
    choice = validate_coupled_scale_choice(
        selected["coupled_scale_choice"],
        gate_bits=rates["gate_proj"],
        up_bits=rates["up_proj"],
    )
    wanted = {
        str(choice["gate"]["candidate_id"]): "gate_proj",
        str(choice["up"]["candidate_id"]): "up_proj",
    }
    # Bind the reusable Hessian as a final-encode session before replaying
    # scorer-fast requests. This permits selected members to receive complete
    # closure without changing the physical candidate batch sent to KQuant.
    selected_gate_config = _config_at_rate(
        runtime,
        weights,
        "gate_proj",
        permutation,
        gate_profile,
        down_profile,
        bits=rates["gate_proj"],
    )
    session = prepare_dense_h_session(h13, selected_gate_config)
    replayed: dict[str, Any] = {}
    for batch, selected_indices in selected_origin_batches(expert_record, selected):
        requests = []
        for member in batch:
            config = _config_at_rate(
                runtime,
                weights,
                member.projection,
                permutation,
                gate_profile,
                down_profile,
                bits=member.bits,
                candidate_sweep=True,
            )
            config = with_activation_parametric_scale_override(
                config,
                scale=member.scale,
                search_contract_id=str(choice["search_contract_id"]),
            )
            source = (
                weights.gate_hf
                if member.projection == "gate_proj"
                else weights.up_hf
            )
            requests.append(UniformSQGCandidateRequest(source=source, config=config))
        outputs = encode_uniform_sqg_candidate_batch(
            requests,
            h13,
            runtime=kquant_runtime,
            dense_h_session=session,
            materialize_indices=selected_indices,
            full_closure_indices=selected_indices,
        )
        if len(outputs) != len(selected_indices):
            raise RuntimeError("selected origin-batch replay returned partial output")
        for index, encoded in zip(selected_indices, outputs, strict=True):
            member = batch[index]
            projection = wanted.get(member.candidate_id)
            if projection is None or projection in replayed:
                raise RuntimeError("selected origin-batch replay identity differs")
            replayed[projection] = encoded
    if set(replayed) != {"gate_proj", "up_proj"}:
        raise RuntimeError("selected origin-batch replay is incomplete")
    return replayed["gate_proj"], replayed["up_proj"]


def shard_ranges(shard_size: int) -> tuple[tuple[int, int], ...]:
    if shard_size <= 0 or NUM_EXPERTS % shard_size:
        raise ValueError("shard size must be a positive divisor of 256")
    return tuple(
        (start, start + shard_size) for start in range(0, NUM_EXPERTS, shard_size)
    )


def _preliminary_selected_down_anchor(
    runtime: Any,
    weights: Any,
    permutation: Any,
    gate_profile: Any,
    down_profile: Any,
    routed: Any,
    calibration_mask: np.ndarray,
    *,
    bits: int,
    device: torch.device,
    chunk_rows: int,
    kquant_runtime: Any,
) -> tuple[Any, dict[str, Any]]:
    """Encode exactly one selected-rate BF16 anchor to freeze private suh."""

    from src.calibration_hessian import apply_frozen_h2_shrinkage
    from src.fresh_pipeline_common import canonical_sha256 as pipeline_sha256
    from src.glm52_fresh_sqg import DenseHessian, encode_uniform_sqg
    from src.glm52_fresh_sqg.reference import tensor_sha256

    rows = routed.row_indices[calibration_mask]
    gates = routed.applied_gates[torch.from_numpy(calibration_mask)].to(device)
    order = permutation.new_to_old.to(device)
    source_gpu = {
        "gate_proj": weights.gate_hf.T.float().to(device),
        "up_proj": weights.up_hf.T.float().to(device),
    }
    raw = torch.zeros((INTERMEDIATE, INTERMEDIATE), dtype=torch.float32, device=device)
    importance_chunks: list[torch.Tensor] = []
    denominator = torch.zeros((), dtype=torch.float64, device=device)
    for begin, end, hidden in runtime.capture.iter_hidden_prefetch(
        rows, chunk_rows=chunk_rows, device=device, dtype=torch.float32
    ):
        gate = torch.matmul(hidden.float(), source_gpu["gate_proj"]).index_select(1, order)
        up = torch.matmul(hidden.float(), source_gpu["up_proj"]).index_select(1, order)
        activation = (torch.nn.functional.silu(gate) * up).contiguous()
        importance = gates[begin:end].square()
        raw.addmm_(activation.T, activation * importance[:, None])
        importance_chunks.append(importance)
        denominator.add_(importance.double().sum())
        del hidden, gate, up, activation, importance
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
        "matrix_sha256": tensor_sha256(h2),
        "purpose": "derive independent K3/K4 private down suh anchors",
        "shrinkage": {key: float(value) for key, value in shrinkage.items()},
        "selection_rows_used": False,
        "holdout_used": False,
    }
    evidence_id = pipeline_sha256(evidence)
    dense = DenseHessian(
        matrix=h2.contiguous(),
        evidence_id=evidence_id,
        construction=evidence["construction"],
        split_id="fit",
        normalization_count=1,
        routed_sample_count=int(rows.size),
        matrix_sha256=evidence["matrix_sha256"],
    )
    config = _config_at_rate(
        runtime,
        weights,
        "down_proj",
        permutation,
        gate_profile,
        down_profile,
        bits=bits,
    )
    encoded = encode_uniform_sqg(
        weights.down_hf, dense, config, runtime=kquant_runtime
    )
    return encoded, {**evidence, "evidence_id": evidence_id}


def _fit_encode_selected_down(
    runtime: Any,
    weights: Any,
    permutation: Any,
    down_profile: Any,
    routed: Any,
    calibration_mask: np.ndarray,
    gate: NativeProjection,
    up: NativeProjection,
    preliminary_down: Any,
    *,
    coupled_scale_choice_id: str,
    bits: int,
    beta: float,
    device: torch.device,
    hadamard: torch.Tensor,
    chunk_rows: int,
    kquant_runtime: Any,
) -> tuple[Any, dict[str, Any]]:
    """Fit and encode one selected down rate under the exact W4A8 path."""

    from src.calibration_hessian import apply_frozen_h2_shrinkage
    from src.fresh_pipeline_common import canonical_sha256 as pipeline_sha256
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
    source_gpu = {
        "gate_proj": weights.gate_hf.T.float().to(device),
        "up_proj": weights.up_hf.T.float().to(device),
        "down_proj": weights.down_hf.T.float().to(device),
    }
    official_down_exl = (
        weights.down_hf.index_select(
            1, permutation.new_to_old.to(device=weights.down_hf.device)
        ).T.float().to(device)
    ).contiguous()
    anchor = SharedResidualProfile.from_stored_vector(
        preliminary_down.suh,
        side="input",
        profile_id=(
            f"layer-{runtime.layer:03d}/expert-{int(weights.expert):03d}/"
            f"down-k{bits}-w4a8-allocation-anchor-v1"
        ),
        derivation="fit_calibration_official_bf16_preliminary_down_encode",
    )
    anchor = realize_shared_residual_profile(anchor, runtime.device)
    raw_h = torch.zeros((INTERMEDIATE, INTERMEDIATE), dtype=torch.float32, device=device)
    raw_encoder_h = torch.zeros_like(raw_h)
    raw_b = torch.zeros((INTERMEDIATE, HIDDEN), dtype=torch.float32, device=device)
    importance_chunks: list[torch.Tensor] = []
    denominator = torch.zeros((), dtype=torch.float64, device=device)
    teacher_energy = torch.zeros((), dtype=torch.float64, device=device)
    for begin, end, hidden in runtime.capture.iter_hidden_prefetch(
        rows, chunk_rows=chunk_rows, device=device, dtype=torch.float32
    ):
        h_a8, h_observation, _ = prepare_gate_up_operand(
            hidden, gate.suh, hadamard, quantize_a8=True
        )
        if h_observation is None or bool(h_observation.preclamp_overflow.any()):
            raise RuntimeError("selected candidate h-A8 path overflowed")
        _, _, activation = apply_gate_up_output_transform_silu(
            native_label_gemm(h_a8, gate.weight),
            native_label_gemm(h_a8, up.weight),
            gate.svh,
            up.svh,
            hadamard,
        )
        teacher = _teacher_output(hidden, source_gpu)
        importance = gates_gpu[begin:end].square()
        q_label, observation, _ = prepare_down_operand(
            activation,
            anchor.expected_stored_fp16().to(device),
            hadamard,
            quantize_a8=True,
        )
        if observation is None or bool(observation.preclamp_overflow.any()):
            raise RuntimeError("selected candidate act-A8 path overflowed")
        q_eff = effective_canonical_operand_from_quantized_transform(
            q_label, anchor.expected_stored_fp16().to(device), hadamard,
            validate_values=False,
        )
        q_pre = kquant_prefinalize_operand_from_label_operand(
            q_label, anchor.signs, hadamard, validate_values=False
        )
        raw_h.addmm_(q_eff.T, q_eff * importance[:, None])
        raw_b.addmm_(q_eff.T, teacher * importance[:, None])
        raw_encoder_h.addmm_(q_pre.T, q_pre * importance[:, None])
        importance_chunks.append(importance)
        denominator.add_(importance.double().sum())
        teacher_energy.add_(
            torch.sum(teacher.square().sum(dim=1) * importance, dtype=torch.float64)
        )
        del hidden, h_a8, activation, teacher, importance, q_label, q_eff, q_pre
    raw_h.div_(denominator.to(torch.float32))
    raw_b.div_(denominator.to(torch.float32))
    raw_encoder_h.div_(denominator.to(torch.float32))
    raw_h = ((raw_h + raw_h.T) * 0.5).contiguous()
    raw_encoder_h = ((raw_encoder_h + raw_encoder_h.T) * 0.5).contiguous()
    identity_scale = raw_h.diagonal().double().mean()
    fitted = shrink_cross_term_objective(
        CrossTermStatistics(
            hessian=raw_h,
            cross_term=raw_b,
            weight_sum=denominator,
            rows=int(rows.size),
        ),
        official_down_exl,
        local_alpha=beta,
        identity_scale=identity_scale,
    )
    target, solver = solve_with_minimal_official_prior(
        fitted,
        official_down_exl,
        identity_scale=identity_scale,
    )
    encoder_h, encoder_shrinkage = apply_frozen_h2_shrinkage(
        raw_encoder_h, torch.cat(importance_chunks)
    )
    encoder_h = encoder_h.contiguous()
    contract = {
        "schema": "glm52-full-w4a8-triplet-candidate-execution-v1",
        "activation_endpoint": FULL_W4A8_ENDPOINT,
        "h_a8": "mxfp8_e4m3_ue8m0_k32",
        "weight_endpoint": "native_finite_e4m3_sqg_labels",
        "nonlinearity": "torch_silu_gate_times_up_with_fp16_inter_gemm_boundary",
        "act_a8": "mxfp8_e4m3_ue8m0_k32",
        "down_input_anchor_sha256": tensor_sha256(anchor.expected_stored_fp16()),
        "down_output_profile": down_profile.manifest(),
        "h_b_beta": beta,
        "coupled_upstream_scale_choice_id": coupled_scale_choice_id,
        "selection_rows_used": False,
        "holdout_used": False,
    }
    contract_id = pipeline_sha256(contract)
    canonical_h_sha256 = tensor_sha256(raw_h)
    cross_term_sha256 = tensor_sha256(raw_b)
    encoder_h_sha256 = tensor_sha256(encoder_h)
    derived_target_sha256 = tensor_sha256(target)
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
        "canonical_h_sha256": canonical_h_sha256,
        "cross_term_sha256": cross_term_sha256,
        "encoder_h_sha256": encoder_h_sha256,
        "encoder_h_shrinkage": {
            key: float(value) for key, value in encoder_shrinkage.items()
        },
        "derived_target_sha256": derived_target_sha256,
        "candidate_hashes_deferred": False,
        "execution_contract": contract,
        "execution_contract_id": contract_id,
        "coupled_upstream_scale_choice_id": coupled_scale_choice_id,
        "solver": solver,
        "selection_rows_used": False,
        "holdout_used": False,
    }
    from scripts.score_sqg_w4a8_triplet_candidates import (
        deferred_down_objective_evidence,
    )

    candidate_evidence_id = pipeline_sha256(
        deferred_down_objective_evidence(evidence)
    )
    evidence_id = pipeline_sha256(evidence)
    evidence["evidence_id"] = evidence_id
    evidence["candidate_evidence_id"] = candidate_evidence_id
    base = _config_at_rate(
        runtime,
        weights,
        "down_proj",
        permutation,
        None,
        down_profile,
        bits=bits,
    )
    binding = W4A8DerivedTensorBinding(
        official_bf16_parent=base.source_binding,
        execution_contract=contract,
        execution_contract_sha256=contract_id,
        fit_hb_evidence_id=evidence_id,
        fit_hb_evidence_sha256=evidence_id,
        fit_hessian_sha256=encoder_h_sha256,
        fit_cross_term_sha256=cross_term_sha256,
        beta=beta,
        derived_tensor_sha256=derived_target_sha256,
    )
    config = replace(
        base,
        source_binding=binding,
        anchored_input_residual_profile=anchor,
    )
    dense = DenseHessian(
        matrix=encoder_h,
        evidence_id=evidence_id,
        construction=DERIVED_H2_CONSTRUCTION,
        split_id="fit",
        normalization_count=1,
        routed_sample_count=int(rows.size),
        matrix_sha256=encoder_h_sha256,
    )
    encoded = encode_uniform_sqg(
        target.contiguous(), dense, config, runtime=kquant_runtime
    )
    if not torch.equal(encoded.suh, anchor.expected_stored_fp16()):
        raise RuntimeError("selected derived down changed anchored private suh")
    if not torch.equal(encoded.svh, down_profile.expected_stored_fp16()):
        raise RuntimeError("selected derived down changed shared output svh")
    return encoded, evidence


def _encode_one_selected_expert(
    *,
    runtime: Any,
    contract: FinalSelectedContract,
    global_h13: torch.Tensor,
    global_evidence: Mapping[str, Any],
    gate_profile: Any,
    down_profile: Any,
    hadamard: torch.Tensor,
    kquant_runtime: Any,
    lut_by_bits: Mapping[int, torch.Tensor],
    expert: int,
    output_dir: Path,
    chunk_rows: int,
) -> dict[str, Any]:
    from src.fresh_pipeline_artifacts import write_expert_artifact
    from src.fresh_pipeline_common import (
        canonical_sha256 as pipeline_sha256,
        sha256_file as pipeline_file_sha256,
    )
    from src.fresh_pipeline_runner import _load_permutation
    from src.glm52_fresh_sqg.reference import tensor_sha256

    selected = contract.selected(expert)
    rates = {name: int(selected["rates"][name]) for name in TRIPLET_PROJECTIONS}
    scale_choice = validate_coupled_scale_choice(
        selected["coupled_scale_choice"],
        gate_bits=rates["gate_proj"],
        up_bits=rates["up_proj"],
    )
    weights = runtime.source.load_expert_bf16(
        runtime.layer, expert, device=runtime.device
    )
    permutation = _load_permutation(runtime, expert)
    h13, h13_evidence, routed, calibration_mask = _build_expert_h13(
        runtime,
        expert=expert,
        global_h13=global_h13,
        global_evidence=global_evidence,
        gate_profile=gate_profile,
        hadamard=hadamard,
        chunk_rows=chunk_rows,
        device=torch.device(runtime.device),
        defer_hashes=False,
    )
    if h13_evidence["candidate_evidence_id"] != selected["h13_evidence_id"]:
        raise RuntimeError(f"expert {expert}: rebuilt H13 evidence differs")
    expert_record = load_scoring_expert_record(contract, expert)
    gate, up = replay_selected_upstream_origin_batches(
        runtime=runtime,
        weights=weights,
        permutation=permutation,
        gate_profile=gate_profile,
        down_profile=down_profile,
        h13=h13,
        expert_record=expert_record,
        selected=selected,
        kquant_runtime=kquant_runtime,
    )
    expected_payloads = selected["payload_sha256"]
    final_gate_payload = _projection_payload_sha256(gate)
    final_up_payload = _projection_payload_sha256(up)
    native_gate = _native_projection(
        gate,
        bits=rates["gate_proj"],
        lut=lut_by_bits[rates["gate_proj"]],
        device=torch.device(runtime.device),
    )
    native_up = _native_projection(
        up,
        bits=rates["up_proj"],
        lut=lut_by_bits[rates["up_proj"]],
        device=torch.device(runtime.device),
    )
    validate_selected_upstream_reencode(gate, native_gate, scale_choice["gate"])
    validate_selected_upstream_reencode(up, native_up, scale_choice["up"])
    preliminary, preliminary_evidence = _preliminary_selected_down_anchor(
        runtime,
        weights,
        permutation,
        gate_profile,
        down_profile,
        routed,
        calibration_mask,
        bits=rates["down_proj"],
        device=torch.device(runtime.device),
        chunk_rows=chunk_rows,
        kquant_runtime=kquant_runtime,
    )
    down, down_evidence = _fit_encode_selected_down(
        runtime,
        weights,
        permutation,
        down_profile,
        routed,
        calibration_mask,
        native_gate,
        native_up,
        preliminary,
        coupled_scale_choice_id=scale_choice["choice_id"],
        bits=rates["down_proj"],
        beta=contract.beta,
        device=torch.device(runtime.device),
        hadamard=hadamard,
        chunk_rows=chunk_rows,
        kquant_runtime=kquant_runtime,
    )
    if (
        down_evidence["candidate_evidence_id"]
        != selected["down_objective_evidence_id"]
    ):
        raise RuntimeError(f"expert {expert}: selected down objective differs")
    if down_evidence["execution_contract_id"] != selected["execution_contract_id"]:
        raise RuntimeError(f"expert {expert}: selected execution contract differs")
    down_payload = _projection_payload_sha256(down)
    if down_payload != expected_payloads["down_proj"]:
        raise RuntimeError(f"expert {expert}: selected down re-encode is not deterministic")
    # Keep W4A8Derived authoritative while satisfying the existing ordinary
    # expert shard validator's direct immutable-parent lookup.
    add_derived_parent_compatibility_aliases(down)
    encoded = {"gate_proj": gate, "up_proj": up, "down_proj": down}
    upstream = {
        "gate_tensor_manifest_sha256": pipeline_sha256(gate.manifest),
        "up_tensor_manifest_sha256": pipeline_sha256(up.manifest),
        "gate_decoded_exl_sha256": tensor_sha256(gate.reconstructed_exl),
        "up_decoded_exl_sha256": tensor_sha256(up.reconstructed_exl),
        "physical_permutation_sha256": permutation.sha256,
    }
    h2_evidence = {
        **down_evidence,
        "identity_fallback": False,
        "pooled_expert_basis": False,
        "matrix_sha256": down.manifest["dense_h"]["matrix_sha256"],
        "upstream_candidate": upstream,
        "preliminary_selected_anchor": preliminary_evidence,
        "selected_candidate_id": selected["candidate_id"],
        "scoring_upstream_candidate_id": selected["upstream_candidate_id"],
        "coupled_scale_choice": scale_choice,
        "scoring_gate_payload_sha256": expected_payloads["gate_proj"],
        "scoring_up_payload_sha256": expected_payloads["up_proj"],
        "final_gate_payload_sha256": final_gate_payload,
        "final_up_payload_sha256": final_up_payload,
        "gate_up_payload_hash_scope_note": (
            "scoring hashes include four-candidate dense-H session ordinals; "
            "final hashes include selected-only session ordinals"
        ),
        "candidate_payload_sha256_before_parent_alias": down_payload,
    }
    source_evidence = {
        "repository_id": gate.manifest["source"]["repository_id"],
        "revision": gate.manifest["source"]["revision"],
        "tensor_payload_sha256": dict(sorted(weights.tensor_sha256.items())),
        "shard_names": dict(sorted(weights.shard_names.items())),
        "shard_sha256": {
            projection: runtime.source.validation.shards[
                weights.shard_names[projection]
            ].sha256
            for projection in TRIPLET_PROJECTIONS
        },
    }
    calibration_evidence = {
        "capture": runtime.capture.binding(),
        "h13_evidence_id": h13.evidence_id,
        "h13_matrix_sha256": tensor_sha256(h13.matrix),
        "fit_routes": routed.evidence(),
        "permutation_artifact_sha256": pipeline_file_sha256(
            runtime.preparation_dir / "permutations" / f"expert-{expert:03d}.json"
        ),
        "profile_cell": {
            "final_selected_contract_id": contract.contract_id,
            "selection_id": contract.profile_selection.get(
                "selection_id", contract.profile_selection.get("selected_cell_id")
            ),
            "selected_candidate_id": selected["candidate_id"],
            "selected_rates": rates,
            "coupled_scale_choice_id": scale_choice["choice_id"],
            "profile_frozen": True,
            "full_w4a8": True,
        },
        "fit_only": True,
        "selection_used_for_encoding_calibration": False,
        "holdout_used": False,
        "fallback_allowed": False,
    }
    result = write_expert_artifact(
        output_dir,
        run_id=contract.run_id,
        layer=runtime.layer,
        expert=expert,
        encoded=encoded,
        permutation=permutation,
        gate_up_profile=gate_profile,
        down_profile=down_profile,
        bit_map=contract.bit_map,
        h2_evidence=h2_evidence,
        calibration_evidence=calibration_evidence,
        source_evidence=source_evidence,
        selection_evidence_sha256=sha256_file(contract.profile_selection_path),
        purpose=PURPOSE,
    )
    return result


def _worker(args: argparse.Namespace, contract: FinalSelectedContract) -> dict[str, Any]:
    from kquant.sqg_e4m3 import sqg_xor_cheb_t12_bytes
    from scripts.encode_final_shard import _open_fast_sealed_runtime
    from src.fresh_pipeline_artifacts import expert_stem, validate_expert_artifact
    from src.fresh_pipeline_common import atomic_json
    from src.fresh_pipeline_runner import _load_bound_kquant_runtime
    from src.glm52_fresh_sqg.reference import normalized_hadamard
    import src.glm52_fresh_sqg.codec as codec

    if args.start is None or args.end is None or not 0 <= args.start < args.end <= NUM_EXPERTS:
        raise ValueError("worker mode requires 0 <= start < end <= 256")
    runtime = _open_fast_sealed_runtime(args.preflight, layer=args.layer, device=args.device)
    gate_profile, down_profile, selection_evidence = _profile_from_selection(
        runtime, contract.profile_selection_path
    )
    if selection_evidence["sha256"] != sha256_file(contract.profile_selection_path):
        raise ValueError("runtime profile selection reconstruction differs")
    device = torch.device(args.device)
    global_h13, global_evidence = _load_global_h13(
        args.score_root.resolve(), args.layer, device=device
    )
    if global_evidence.get("profile_selection") != selection_evidence:
        raise ValueError("global W4A8 H13/profile selection binding differs")
    if contract.scores.get("profile_selection") != selection_evidence:
        raise ValueError("score/global W4A8 H13 profile binding differs")
    hadamard = normalized_hadamard(device=device, dtype=torch.float32, size=128)
    kquant_runtime = _load_bound_kquant_runtime(runtime)
    lut_by_bits = {
        bits: sqg_xor_cheb_t12_bytes(bits).to(device).contiguous()
        for bits in (3, 4)
    }
    codec.PRODUCTION_H13_CONSTRUCTION = H13_CONSTRUCTION
    codec.PRODUCTION_H2_CONSTRUCTION = PRELIMINARY_H2_CONSTRUCTION
    output_dir = (
        args.output_root.resolve()
        / f"layer_{args.layer:03d}"
        / "expert_shards"
        / f"experts_{args.start:03d}_{args.end:03d}"
    )
    completed = 0
    started = time.monotonic()
    for expert in range(args.start, args.end):
        manifest_path = output_dir / f"{expert_stem(args.layer, expert)}.json"
        if manifest_path.exists():
            existing = validate_expert_artifact(manifest_path)
            if (
                existing.get("run_id") != contract.run_id
                or existing.get("purpose") != PURPOSE
                or existing.get("bits") != contract.selected(expert)["rates"]
            ):
                raise ValueError(f"expert {expert}: existing final artifact differs")
        else:
            _encode_one_selected_expert(
                runtime=runtime,
                contract=contract,
                global_h13=global_h13,
                global_evidence=global_evidence,
                gate_profile=gate_profile,
                down_profile=down_profile,
                hadamard=hadamard,
                kquant_runtime=kquant_runtime,
                lut_by_bits=lut_by_bits,
                expert=expert,
                output_dir=output_dir,
                chunk_rows=args.chunk_rows,
            )
        completed += 1
        atomic_json(
            output_dir / "progress.json",
            {
                "schema": PROGRESS_SCHEMA,
                "complete": completed == args.end - args.start,
                "layer": args.layer,
                "start": args.start,
                "end": args.end,
                "completed_prefix": completed,
                "final_selected_contract_id": contract.contract_id,
                "run_id": contract.run_id,
            },
            overwrite=True,
        )
        print(f"layer {args.layer} full-W4A8 final: expert {expert + 1}/256", flush=True)
    return {
        "complete": True,
        "layer": args.layer,
        "start": args.start,
        "end": args.end,
        "expert_count": completed,
        "output_dir": str(output_dir),
        "contract_id": contract.contract_id,
        "run_id": contract.run_id,
        "elapsed_seconds": time.monotonic() - started,
    }


def _publish(source: Path, destination: Path, expected_sha256: str) -> str:
    if destination.exists():
        if sha256_file(destination) != expected_sha256:
            raise ValueError(f"refusing to overwrite mismatched artifact: {destination}")
        return "existing_equal"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination, follow_symlinks=False)
        return "hardlinked"
    except OSError:
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            with source.open("rb") as reader, temporary.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=64 << 20)
                writer.flush()
                os.fsync(writer.fileno())
            if sha256_file(temporary) != expected_sha256:
                raise ValueError("copied artifact hash differs")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return "copied"


def _finalize(args: argparse.Namespace, contract: FinalSelectedContract) -> dict[str, Any]:
    from scripts.encode_final_shard import _open_fast_sealed_runtime
    from src.fresh_pipeline_artifacts import (
        assemble_layer_artifact,
        expert_stem,
        validate_expert_artifact,
    )

    runtime = _open_fast_sealed_runtime(args.preflight, layer=args.layer, device=args.device)
    gate_profile, down_profile, selection_evidence = _profile_from_selection(
        runtime, contract.profile_selection_path
    )
    layer_root = args.output_root.resolve() / f"layer_{args.layer:03d}"
    shard_root = layer_root / "expert_shards"
    expected_ranges = shard_ranges(args.shard_size)
    expected_names = {f"experts_{start:03d}_{end:03d}" for start, end in expected_ranges}
    observed_names = {
        path.name for path in shard_root.glob("experts_*_*") if path.is_dir()
    }
    if observed_names != expected_names:
        raise ValueError("full-W4A8 final shard directory set differs")
    canonical_experts = layer_root / "experts"
    publication = {"hardlinked": 0, "copied": 0, "existing_equal": 0}
    for start, end in expected_ranges:
        directory = shard_root / f"experts_{start:03d}_{end:03d}"
        for expert in range(start, end):
            stem = expert_stem(args.layer, expert)
            manifest_path = directory / f"{stem}.json"
            artifact = validate_expert_artifact(manifest_path)
            if (
                artifact.get("run_id") != contract.run_id
                or artifact.get("bits") != contract.selected(expert)["rates"]
            ):
                raise ValueError(f"expert {expert}: final shard binding differs")
            for source in (
                manifest_path,
                manifest_path.with_suffix(".json.sha256"),
                manifest_path.with_suffix(".safetensors"),
            ):
                outcome = _publish(
                    source,
                    canonical_experts / source.name,
                    sha256_file(source),
                )
                publication[outcome] += 1
    layer_evidence = {
        "schema": FINALIZE_SCHEMA,
        "complete": True,
        "final_selected_contract_id": contract.contract_id,
        "allocation_id": contract.allocation["allocation_id"],
        "allocation_sha256": sha256_file(contract.allocation_path),
        "triplet_score_manifest_id": contract.scores["score_manifest_id"],
        "triplet_scores_sha256": sha256_file(contract.scores_path),
        "profile_selection": selection_evidence,
        "activation_endpoint": FULL_W4A8_ENDPOINT,
        "selected_only_reencode": True,
        "w4a8_derived_down_lineage": True,
        "selection_rows_used_for_encoding": False,
        "holdout_rows_used": False,
        "mcg_inputs": 0,
        "numerical_device": "cuda:0",
        "cpu_numerical_work": False,
        "fp32_accumulate": True,
        "bf16_source_on_cuda": True,
        "hessian_on_cuda": True,
        "closure_on_cuda": True,
        "capture_storage": "host-mmap-pinned-double-buffered",
        "cpu_roles_after_numerical_closure": ["hashing", "serialization"],
    }
    layer_artifact = assemble_layer_artifact(
        canonical_experts,
        layer_root / "final",
        run_id=contract.run_id,
        layer=args.layer,
        bit_map=contract.bit_map,
        gate_up_profile=gate_profile,
        down_profile=down_profile,
        layer_evidence=layer_evidence,
    )
    return {
        "complete": True,
        "layer": args.layer,
        "contract_id": contract.contract_id,
        "run_id": contract.run_id,
        "expert_count": NUM_EXPERTS,
        "publication": publication,
        "layer_manifest": str(layer_root / "final" / f"fresh-sqg-layer-{args.layer:03d}.json"),
        "layer_shard_sha256": layer_artifact["shard_sha256"],
        "sqg_tensor_count": layer_artifact["lineage"]["sqg_tensor_count"],
        "mcg_tensor_count": layer_artifact["lineage"]["mcg_tensor_count"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--profile-selection", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--triplet-scores", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--finalize", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.threads <= 0 or args.chunk_rows <= 0:
        raise ValueError("threads and chunk rows must be positive")
    contract = load_final_selected_contract(
        layer=args.layer,
        allocation_path=args.allocation,
        scores_path=args.triplet_scores,
        profile_selection_path=args.profile_selection,
    )
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise ValueError("full-W4A8 final encoding/materialization requires CUDA")
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
    result = _finalize(args, contract) if args.finalize else _worker(args, contract)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
