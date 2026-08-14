"""Deterministic SQG-native K3/K4 rate allocation for GLM routed layers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERTS = 256
PROJECTIONS = ("down_proj", "gate_proj", "up_proj")
TENSORS_PER_LAYER = EXPERTS * len(PROJECTIONS)
# 3.0625 bpw: 720 K3 + 48 K4 of the 768 routed tensors per layer.
# The exact multiple-choice DP already accepts any K4 budget; only this
# declared target changes.  The sealed 384/384 tree is left untouched.
K3_COUNT = 720
K4_COUNT = 48
BIT_UNITS = K3_COUNT * 3 + K4_COUNT * 4
LOSS_DEFINITION = (
    "fit_routed_mass_times_candidate_native_hessian_weighted_reconstruction_sse_v1"
)
TRIPLET_SCORE_SCHEMA = "glm52-sqg-w4a8-expert-triplet-scores-v3"
TRIPLET_ALLOCATION_SCHEMA = "glm52-sqg-native-per-tensor-k34-allocation-v3"
TRIPLET_LOSS_DEFINITION = "fit_realized_full_w4a8_complete_expert_gate_square_sse_v1"
TRIPLET_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
FULL_W4A8_ENDPOINT = "full-w4a8"
TENSOR_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>down_proj|gate_proj|up_proj)$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TripletCandidateScore:
    """One realized complete-expert rate candidate used by the v3 allocator."""

    candidate_id: str
    rates: tuple[int, int, int]
    loss: float
    record: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("triplet candidate ID must not be empty")
        if len(self.rates) != len(TRIPLET_PROJECTIONS) or any(
            rate not in (3, 4) for rate in self.rates
        ):
            raise ValueError("triplet candidate rates must be gate/up/down K3 or K4")
        if not math.isfinite(self.loss) or self.loss < 0.0:
            raise ValueError("triplet candidate loss must be finite and nonnegative")

    @property
    def k4_count(self) -> int:
        return sum(rate == 4 for rate in self.rates)

    @property
    def rate_map(self) -> dict[str, int]:
        return dict(zip(TRIPLET_PROJECTIONS, self.rates, strict=True))


@dataclass(frozen=True)
class TripletDPResult:
    """Exact multiple-choice solution under a fixed number of K4 tensors."""

    selected: Mapping[int, TripletCandidateScore]
    objective: float
    target_k4: int


def enumerate_rate_triplets() -> tuple[tuple[int, int, int], ...]:
    """Return the canonical complete gate/up/down K3/K4 candidate grid."""

    return tuple(itertools.product((3, 4), repeat=len(TRIPLET_PROJECTIONS)))


def solve_triplet_dp(
    candidates_by_expert: Mapping[int, Iterable[TripletCandidateScore]],
    *,
    target_k4: int,
    expected_experts: int | None = None,
) -> TripletDPResult:
    """Minimize additive complete-expert loss at one exact K4 budget.

    The dynamic program is exact for the unary realized-W4A8 objective.  Its
    canonical tie break is the lexicographically smallest sequence of rate
    tuples in ascending expert order.  Signed top-k cross-expert terms are a
    separate post-allocation score/refinement and are deliberately not
    mislabeled as part of this additive solve.
    """

    if not isinstance(target_k4, int) or target_k4 < 0:
        raise ValueError("target K4 count must be a nonnegative integer")
    experts = sorted(candidates_by_expert)
    if any(not isinstance(expert, int) or expert < 0 for expert in experts):
        raise ValueError("expert IDs must be nonnegative integers")
    if expected_experts is not None:
        if expected_experts <= 0:
            raise ValueError("expected expert count must be positive")
        if experts != list(range(expected_experts)):
            raise ValueError("triplet candidate expert grid differs")
    if not experts:
        raise ValueError("triplet candidate grid is empty")

    canonical = enumerate_rate_triplets()
    ordinal = {rates: index for index, rates in enumerate(canonical)}
    normalized: dict[int, tuple[TripletCandidateScore, ...]] = {}
    for expert in experts:
        items = tuple(candidates_by_expert[expert])
        if not items:
            raise ValueError(f"expert {expert}: no triplet candidates")
        ids = [item.candidate_id for item in items]
        rates = [item.rates for item in items]
        if len(set(ids)) != len(ids):
            raise ValueError(f"expert {expert}: duplicate candidate ID")
        if len(set(rates)) != len(rates):
            raise ValueError(f"expert {expert}: duplicate rate tuple")
        normalized[expert] = tuple(
            sorted(items, key=lambda item: (ordinal.get(item.rates, 99), item.candidate_id))
        )

    # State: K4 count -> (objective, canonical rate-ordinal path, selections).
    states: dict[
        int, tuple[float, tuple[int, ...], tuple[TripletCandidateScore, ...]]
    ] = {0: (0.0, (), ())}
    for expert in experts:
        next_states: dict[
            int, tuple[float, tuple[int, ...], tuple[TripletCandidateScore, ...]]
        ] = {}
        for used_k4, (objective, tie_path, selections) in states.items():
            for candidate in normalized[expert]:
                new_k4 = used_k4 + candidate.k4_count
                if new_k4 > target_k4:
                    continue
                new_objective = objective + candidate.loss
                if not math.isfinite(new_objective):
                    raise ValueError("triplet DP objective became non-finite")
                new_tie_path = tie_path + (ordinal[candidate.rates],)
                incumbent = next_states.get(new_k4)
                if incumbent is None or (new_objective, new_tie_path) < (
                    incumbent[0],
                    incumbent[1],
                ):
                    next_states[new_k4] = (
                        new_objective,
                        new_tie_path,
                        selections + (candidate,),
                    )
        states = next_states
        if not states:
            raise ValueError(
                f"triplet K4 budget became infeasible after expert {expert}"
            )

    winner = states.get(target_k4)
    if winner is None:
        raise ValueError("triplet candidate grid cannot satisfy the exact K4 budget")
    objective, _, selections = winner
    selected = dict(zip(experts, selections, strict=True))
    if sum(item.k4_count for item in selected.values()) != target_k4:
        raise AssertionError("triplet DP reconstruction changed the K4 budget")
    return TripletDPResult(
        selected=selected,
        objective=objective,
        target_k4=target_k4,
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_grid(layer: int) -> set[str]:
    return {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        for expert in range(EXPERTS)
        for projection in PROJECTIONS
    }


def _validate_source_map(layer: int, value: Any) -> tuple[dict[str, int], dict[str, int]]:
    if not isinstance(value, Mapping):
        raise TypeError("source bit_map must be an object")
    bit_map = {str(name): int(bits) for name, bits in value.items()}
    if set(bit_map) != _tensor_grid(layer):
        raise ValueError("source bit_map tensor grid differs")
    histogram = {"3": 0, "4": 0, "5": 0}
    for name, bits in bit_map.items():
        match = TENSOR_RE.fullmatch(name)
        if match is None or int(match.group("layer")) != layer:
            raise ValueError(f"source tensor identity differs: {name}")
        if bits not in (3, 4, 5):
            raise ValueError(f"unsupported source rate K{bits}: {name}")
        histogram[str(bits)] += 1
    if sum(bits for bits in bit_map.values()) != BIT_UNITS:
        raise ValueError("source allocation does not close at 2,352 bit units")
    return dict(sorted(bit_map.items())), histogram


def _finite_nonnegative(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _validate_triplet_scores(
    layer: int, value: Any
) -> dict[int, tuple[TripletCandidateScore, ...]]:
    """Validate one complete fit-only realized-W4A8 candidate portfolio."""

    if not isinstance(value, Mapping):
        raise TypeError("triplet score manifest must be an object")
    if value.get("schema") != TRIPLET_SCORE_SCHEMA:
        raise ValueError("triplet score schema differs")
    if value.get("complete") is not True or int(value.get("layer", -1)) != layer:
        raise ValueError("triplet score completion/layer differs")
    if (
        value.get("role") != "fit"
        or value.get("selection_used") is not False
        or value.get("holdout_used") is not False
    ):
        raise ValueError("triplet allocation requires fit-only candidate scores")
    if value.get("loss_definition") != TRIPLET_LOSS_DEFINITION:
        raise ValueError("triplet score loss definition differs")
    if value.get("activation_endpoint") != FULL_W4A8_ENDPOINT:
        raise ValueError("triplet scores must execute the full-W4A8 endpoint")
    if value.get("signed_top8_used_for_candidate_choice") is not False:
        raise ValueError("v3 DP input must contain unary complete-expert scores")

    raw_experts = value.get("experts")
    expected_experts = {str(expert) for expert in range(EXPERTS)}
    if not isinstance(raw_experts, Mapping) or set(map(str, raw_experts)) != expected_experts:
        raise ValueError("triplet score expert grid differs")
    expected_rates = set(enumerate_rate_triplets())
    result: dict[int, tuple[TripletCandidateScore, ...]] = {}
    for expert in range(EXPERTS):
        record = raw_experts[str(expert)]
        if not isinstance(record, Mapping) or int(record.get("expert", -1)) != expert:
            raise ValueError(f"expert {expert}: triplet score identity differs")
        raw_candidates = record.get("candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) != len(expected_rates):
            raise ValueError(f"expert {expert}: eight triplet candidates are required")
        candidates: list[TripletCandidateScore] = []
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, Mapping):
                raise TypeError(f"expert {expert}: triplet candidate must be an object")
            raw_rates = raw_candidate.get("rates")
            if not isinstance(raw_rates, Mapping) or set(raw_rates) != set(
                TRIPLET_PROJECTIONS
            ):
                raise ValueError(f"expert {expert}: triplet rate map differs")
            rates = tuple(int(raw_rates[name]) for name in TRIPLET_PROJECTIONS)
            candidate = TripletCandidateScore(
                candidate_id=str(raw_candidate.get("candidate_id", "")),
                rates=rates,
                loss=_finite_nonnegative(
                    raw_candidate.get("realized_w4a8_fit_sse"),
                    label=f"expert {expert} {rates} realized W4A8 fit SSE",
                ),
                record=dict(raw_candidate),
            )
            if int(raw_candidate.get("k4_count", -1)) != candidate.k4_count:
                raise ValueError(f"expert {expert}: triplet K4 count differs")
            if raw_candidate.get("full_w4a8_realized") is not True:
                raise ValueError(f"expert {expert}: candidate was not realized as W4A8")
            payloads = raw_candidate.get("payload_sha256")
            if not isinstance(payloads, Mapping) or set(payloads) != set(
                TRIPLET_PROJECTIONS
            ):
                raise ValueError(f"expert {expert}: payload hash map differs")
            if any(
                not isinstance(payloads[name], str)
                or SHA256_RE.fullmatch(payloads[name]) is None
                for name in TRIPLET_PROJECTIONS
            ):
                raise ValueError(f"expert {expert}: payload hash is malformed")
            for evidence_name in (
                "execution_contract_id",
                "h13_evidence_id",
                "upstream_candidate_id",
                "down_objective_evidence_id",
            ):
                evidence = raw_candidate.get(evidence_name)
                if not isinstance(evidence, str) or not evidence:
                    raise ValueError(
                        f"expert {expert}: {evidence_name} is absent"
                    )
            candidates.append(candidate)
        observed_rates = {candidate.rates for candidate in candidates}
        if observed_rates != expected_rates:
            raise ValueError(f"expert {expert}: triplet rate grid is incomplete")
        result[expert] = tuple(candidates)
    return result


def build_triplet_layer_allocation(
    *,
    layer: int,
    source_sidecar: Path | None,
    score_manifest: Path,
) -> dict[str, Any]:
    """Build a v3 exact-budget allocation from realized full-W4A8 triplets."""

    if source_sidecar is None:
        source_evidence: dict[str, Any] = {
            "kind": "none_native_triplet_allocation",
            "source_checkpoint_input": False,
            "rate_map_used_for_choice": False,
        }
    else:
        source = json.loads(source_sidecar.read_text(encoding="utf-8"))
        if int(source.get("layer", -1)) != layer:
            raise ValueError("source sidecar layer differs")
        source_map, source_histogram = _validate_source_map(
            layer, source.get("bit_map")
        )
        source_evidence = {
            "kind": "historical_sidecar_not_used_for_choice",
            "path": str(source_sidecar.resolve()),
            "sha256": sha256_file(source_sidecar),
            "histogram": source_histogram,
            "bit_units": sum(source_map.values()),
            "source_checkpoint_input": True,
            "rate_map_used_for_choice": False,
        }
    score_payload = json.loads(score_manifest.read_text(encoding="utf-8"))
    candidates = _validate_triplet_scores(layer, score_payload)
    solved = solve_triplet_dp(
        candidates,
        target_k4=K4_COUNT,
        expected_experts=EXPERTS,
    )

    bit_map: dict[str, int] = {}
    assignments: dict[str, Any] = {}
    for expert, candidate in sorted(solved.selected.items()):
        rates = candidate.rate_map
        for projection, bits in rates.items():
            bit_map[
                f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
            ] = bits
        assignments[str(expert)] = {
            "candidate_id": candidate.candidate_id,
            "rates": rates,
            "k4_count": candidate.k4_count,
            "realized_w4a8_fit_sse": candidate.loss,
            "candidate_record": dict(candidate.record or {}),
        }
    bit_map = dict(sorted(bit_map.items()))
    histogram = {
        "3": sum(bits == 3 for bits in bit_map.values()),
        "4": sum(bits == 4 for bits in bit_map.values()),
    }
    if set(bit_map) != _tensor_grid(layer):
        raise AssertionError("v3 allocation tensor grid differs")
    if histogram != {"3": K3_COUNT, "4": K4_COUNT}:
        raise AssertionError(f"v3 allocation histogram differs: {histogram}")
    if sum(bit_map.values()) != BIT_UNITS:
        raise AssertionError("v3 allocation rate map differs from 2,352 bit units")

    body: dict[str, Any] = {
        "schema": TRIPLET_ALLOCATION_SCHEMA,
        "layer": layer,
        "complete": True,
        "method": "sqg_native_realized_full_w4a8_triplet_exact_k4_dp",
        "purpose": "topology-neutral independent per-tensor 3.0625-bpw SQG allocation",
        "source_sidecar": source_evidence,
        "score_binding": {
            "path": str(score_manifest.resolve()),
            "sha256": sha256_file(score_manifest),
            "schema": TRIPLET_SCORE_SCHEMA,
            "loss_definition": TRIPLET_LOSS_DEFINITION,
            "activation_endpoint": FULL_W4A8_ENDPOINT,
        },
        "dp": {
            "experts": EXPERTS,
            "candidates_per_expert": len(enumerate_rate_triplets()),
            "target_k4": K4_COUNT,
            "objective": solved.objective,
            "objective_scope": "additive_complete_expert_unary_sse",
            "tie_break": "lexicographically_smallest_rate_tuple_sequence",
            "signed_top8_cross_terms_optimized": False,
        },
        "expert_assignments": assignments,
        "bit_map": bit_map,
        "histogram": histogram,
        "bit_units": BIT_UNITS,
        "selection_used_for_allocation": False,
        "holdout_used_for_allocation": False,
        "mcg_rate_map_used_for_choice": False,
        "mcg_payloads_or_transforms_reused": False,
    }
    body["allocation_id"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body


def _validate_scores(layer: int, value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError("candidate score manifest must be an object")
    if value.get("schema") != "glm52-sqg-k34-candidate-distortion-v1":
        raise ValueError("candidate score schema differs")
    if int(value.get("layer", -1)) != layer:
        raise ValueError("candidate score layer differs")
    if value.get("role") != "fit" or value.get("holdout_used") is not False:
        raise ValueError("rate allocation must use fit-only candidate scores")
    if value.get("loss_definition") != LOSS_DEFINITION:
        raise ValueError("candidate score loss definition differs")
    raw = value.get("tensors")
    if not isinstance(raw, Mapping) or set(map(str, raw)) != _tensor_grid(layer):
        raise ValueError("candidate score tensor grid differs")

    scores: dict[str, dict[str, Any]] = {}
    for name in sorted(raw):
        record = raw[name]
        if not isinstance(record, Mapping):
            raise TypeError(f"candidate score must be an object: {name}")
        k3 = _finite_nonnegative(record.get("k3_loss"), label=f"{name} K3 loss")
        k4 = _finite_nonnegative(record.get("k4_loss"), label=f"{name} K4 loss")
        routed_mass = _finite_nonnegative(
            record.get("routed_mass"), label=f"{name} routed mass"
        )
        if routed_mass <= 0.0:
            raise ValueError(f"{name} routed mass must be positive")
        hessian_id = record.get("hessian_evidence_id")
        if not isinstance(hessian_id, str) or not hessian_id:
            raise ValueError(f"{name} Hessian evidence ID is absent")
        artifact_hashes: dict[str, str] = {}
        for key in ("k3_artifact_sha256", "k4_artifact_sha256"):
            digest = record.get(key)
            if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{name} {key} is malformed")
            artifact_hashes[key] = digest
        scores[str(name)] = {
            "k3_loss": k3,
            "k4_loss": k4,
            "marginal_benefit": k3 - k4,
            "routed_mass": routed_mass,
            "hessian_evidence_id": hessian_id,
            **artifact_hashes,
        }
    return scores


def build_layer_allocation(
    *,
    layer: int,
    source_sidecar: Path | None,
    score_manifest: Path | None,
) -> dict[str, Any]:
    """Return one exact 720-K3/48-K4 allocation plus complete audit evidence.

    Historical tensor-marginal score manifests retain their v2 behavior.  A
    v3 realized-W4A8 triplet manifest dispatches to the exact multiple-choice
    allocator for every source layer, including layers whose inherited map was
    already K3/K4.
    """

    if score_manifest is not None:
        score_header = json.loads(score_manifest.read_text(encoding="utf-8"))
        if score_header.get("schema") == TRIPLET_SCORE_SCHEMA:
            return build_triplet_layer_allocation(
                layer=layer,
                source_sidecar=source_sidecar,
                score_manifest=score_manifest,
            )

    if source_sidecar is None:
        raise ValueError("historical non-triplet allocation requires --source-sidecar")
    source = json.loads(source_sidecar.read_text(encoding="utf-8"))
    if int(source.get("layer", -1)) != layer:
        raise ValueError("source sidecar layer differs")
    source_map, source_histogram = _validate_source_map(layer, source.get("bit_map"))

    if source_histogram == {"3": K3_COUNT, "4": K4_COUNT, "5": 0}:
        if score_manifest is not None:
            raise ValueError("already mixed K3/K4 layer must not receive reallocation scores")
        bit_map = source_map
        ranking: list[dict[str, Any]] = []
        method = "source_k3_k4_map_preserved"
        score_binding = None
    else:
        if score_manifest is None:
            raise ValueError("source K5 layer requires SQG-native K3/K4 candidate scores")
        score_payload = json.loads(score_manifest.read_text(encoding="utf-8"))
        scores = _validate_scores(layer, score_payload)
        ordered = sorted(scores, key=lambda name: (-scores[name]["marginal_benefit"], name))
        promoted = set(ordered[:K4_COUNT])
        bit_map = {name: (4 if name in promoted else 3) for name in sorted(scores)}
        ranking = [
            {
                "rank": rank,
                "tensor": name,
                "selected_bits": bit_map[name],
                **scores[name],
            }
            for rank, name in enumerate(ordered, start=1)
        ]
        method = "sqg_native_fit_hessian_marginal_top48_k4"
        score_binding = {
            "path": str(score_manifest.resolve()),
            "sha256": sha256_file(score_manifest),
            "schema": score_payload["schema"],
            "loss_definition": LOSS_DEFINITION,
            "tie_break": "descending_marginal_benefit_then_ascending_tensor_name",
        }

    histogram = {
        "3": sum(bits == 3 for bits in bit_map.values()),
        "4": sum(bits == 4 for bits in bit_map.values()),
    }
    if histogram != {"3": K3_COUNT, "4": K4_COUNT}:
        raise AssertionError(f"final K3/K4 histogram differs: {histogram}")
    if sum(bit_map.values()) != BIT_UNITS:
        raise AssertionError("final rate map differs from 2,688 bit units")

    body: dict[str, Any] = {
        "schema": "glm52-sqg-native-per-tensor-k34-allocation-v2",
        "layer": layer,
        "complete": True,
        "method": method,
        "purpose": "topology-neutral independent per-tensor 3.0625-bpw SQG allocation",
        "source_sidecar": {
            "path": str(source_sidecar.resolve()),
            "sha256": sha256_file(source_sidecar),
            "histogram": source_histogram,
            "bit_units": sum(source_map.values()),
        },
        "score_binding": score_binding,
        "bit_map": bit_map,
        "histogram": histogram,
        "bit_units": BIT_UNITS,
        "ranking": ranking,
        "holdout_used_for_allocation": False,
        "mcg_payloads_or_transforms_reused": False,
    }
    body["allocation_id"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body
