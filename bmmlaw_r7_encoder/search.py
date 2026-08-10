"""Resumable proxy-first rotation, block-scale, and permutation search.

Every required draw is proxy-scored. Shared layer-vector decisions promote two
candidates on a mass-stratified expert sample; per-expert decisions select by
proxy, and the same mass-stratified sample receives a full TRELLIS round-trip
validation after the complete permutation/draw/scale bundle is chosen. Every
expert later receives actual 3/4/5 sensitivity probes and a final held-out
encode. Progress is provenance-bound and flushed after each score, so
preemption never restarts a completed full encode.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .backend import LayerCapture, Round7Backend
from .constants import DEFAULT_DRAWS, HAD_K, HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_EXPERTS
from .determinism import (
    atomic_write_json,
    canonical_json_bytes,
    derive_seed,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .layer import LayerProcessor
from .permutation import (
    descending_diag_permutation,
    energy_balanced_permutation,
    identity_permutation,
    permute_expert_hf,
    stored_descending_diag_permutation,
)
from .rotations import rademacher_vector
from .search_artifact import (
    ExpertSearch,
    LayerSearch,
    load_layer_search,
    write_layer_search,
)
from .types import StateShard


def mass_stratified_experts(masses: Sequence[float], count: int) -> tuple[int, ...]:
    if len(masses) != NUM_EXPERTS or not 1 <= count <= NUM_EXPERTS:
        raise ValueError("invalid mass-stratified sample request")
    numeric = tuple(float(value) for value in masses)
    total = sum(numeric)
    if total <= 0:
        raise ValueError("routed mass is empty")
    ordered = sorted(range(NUM_EXPERTS), key=lambda expert: (-numeric[expert], expert))
    cumulative = []
    running = 0.0
    for expert in ordered:
        running += numeric[expert]
        cumulative.append((running / total, expert))
    hot, cold = ordered[0], ordered[-1]
    selected = {hot, cold}
    for index in range(count):
        target = (index + 0.5) / count
        selected.add(
            min(cumulative, key=lambda item: (abs(item[0] - target), item[1]))[1]
        )
    for expert in ordered:
        if len(selected) >= count:
            break
        selected.add(expert)
    middle = sorted(
        selected - {hot, cold}, key=lambda expert: (-numeric[expert], expert)
    )
    result = [hot]
    result.extend(middle[: max(0, count - 2)])
    if count > 1:
        result.append(cold)
    return tuple(result)


def _base_expert(layer: int, expert: int, draw: int = 0) -> ExpertSearch:
    return ExpertSearch(
        permutation=identity_permutation(),
        permutation_policy="identity",
        gate_svh=rademacher_vector(INTERMEDIATE_SIZE, layer, expert, "gate_svh", draw),
        up_svh=rademacher_vector(INTERMEDIATE_SIZE, layer, expert, "up_svh", draw),
        down_suh=rademacher_vector(INTERMEDIATE_SIZE, layer, expert, "down_suh", draw),
        gate_n_g_scale=(1.0,) * (INTERMEDIATE_SIZE // HAD_K),
        up_n_g_scale=(1.0,) * (INTERMEDIATE_SIZE // HAD_K),
        down_k_g_scale=(1.0,) * (INTERMEDIATE_SIZE // HAD_K),
        draw=draw,
        selection_score=float("inf"),
        selection_score_kind="deterministic-sketch",
    )


def _normalized_quarter_scales(values: Sequence[float]) -> tuple[float, ...]:
    cleaned = [max(float(value), 1e-20) for value in values]
    geometric = math.exp(sum(math.log(value) for value in cleaned) / len(cleaned))
    scales = [max(0.5, min(2.0, (geometric / value) ** 0.25)) for value in cleaned]
    normalization = math.exp(sum(math.log(value) for value in scales) / len(scales))
    return tuple(value / normalization for value in scales)


def _inverse(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(1.0 / float(value) for value in values)


class SearchRunner:
    """Proxy-first pilot with immutable per-candidate progress records."""

    def __init__(
        self,
        *,
        processor: LayerProcessor,
        backend: Round7Backend,
        capture: LayerCapture,
        shards: tuple[StateShard, ...],
        output: str | Path,
        draws: int = DEFAULT_DRAWS,
        shared_sample_experts: int = 16,
    ) -> None:
        if not 8 <= draws <= 16:
            raise ValueError("multi-draw count must be in [8,16]")
        self.processor = processor
        self.backend = backend
        self.capture = capture
        self.shards = shards
        self.output = Path(output)
        self.draws = draws
        self.sample = mass_stratified_experts(
            capture.mass_audit.mass_by_expert, shared_sample_experts
        )
        self.bindings = {
            "capture_sha256": capture.digest,
            "state_sha256": capture.state_sha256,
            "source_inventory_sha256": processor.source_inventory_sha256,
            "numeric_environment_sha256": processor.numeric_environment_sha256,
            "runtime_inventory_sha256": processor.runtime_inventory_sha256,
            "backend_fingerprint": backend.fingerprint,
            "draws": str(draws),
            "sample_sha256": sha256_bytes(canonical_json_bytes(list(self.sample))),
        }
        self.progress_path = self.output.with_name(
            f".{self.output.stem}.pilot-progress.json"
        )
        self.progress: dict[str, object] = {
            "schema": "r7-search-progress-v2",
            "layer": capture.layer,
            "bindings": self.bindings,
            "scores": {},
            "expert_results": {},
        }
        if self.progress_path.exists():
            prior = read_json(self.progress_path)
            if (
                prior.get("schema") != "r7-search-progress-v2"
                or int(prior.get("layer", -1)) != capture.layer
                or prior.get("bindings") != self.bindings
            ):
                raise ValueError("search progress provenance drift")
            self.progress = prior
        self._prepared: dict[int, tuple[object, object, object]] = {}

    def _flush(self) -> None:
        atomic_write_json(self.progress_path, self.progress)

    def _record_score(self, key: str, value: float, *, kind: str) -> float:
        if not math.isfinite(value):
            raise ValueError(f"non-finite {kind} search score")
        scores = self.progress["scores"]
        assert isinstance(scores, dict)
        record = {"kind": kind, "value": format(value, ".17g")}
        incumbent = scores.get(key)
        if incumbent is not None and incumbent != record:
            raise ValueError(f"attempt to rewrite search score {key}")
        scores[key] = record
        self._flush()
        return value

    def _cached_score(self, key: str) -> float | None:
        scores = self.progress["scores"]
        assert isinstance(scores, dict)
        record = scores.get(key)
        return None if record is None else float(record["value"])

    def _layer_search(
        self,
        *,
        shared_suh,
        shared_svh,
        shared_k_scales,
        shared_n_scales,
        experts,
        shared_draw: int,
        shared_score: float,
    ) -> LayerSearch:
        return LayerSearch(
            layer=self.capture.layer,
            draws=self.draws,
            gate_up_suh=tuple(shared_suh),
            down_svh=tuple(shared_svh),
            gate_up_k_g_scale=tuple(shared_k_scales),
            down_n_g_scale=tuple(shared_n_scales),
            shared_draw=shared_draw,
            shared_heldout_score=shared_score,
            experts=experts,
            pilot_evidence_sha256="pending",
            unverified=False,
            bindings=self.bindings,
        )

    def _prepare(self, expert: int):
        if expert not in self._prepared:
            weights = self.backend.load_bf16_expert(
                layer=self.capture.layer, expert=expert
            )
            weights.validate_bf16(self.capture.layer, expert)
            covariance = self.processor._covariance(
                self.capture, self.shards, expert, "fit"
            )
            holdout = self.processor._holdout(
                self.capture, self.shards, expert, exclude_ids=covariance.row_ids
            )
            self._prepared[expert] = (weights, covariance, holdout)
        return self._prepared[expert]

    def _score_expert(
        self, search: LayerSearch, expert: int, *, return_diag: bool = False
    ):
        weights, covariance_use, holdout = self._prepare(expert)
        per_expert = search.experts[expert]
        reference = permute_expert_hf(
            weights.gate_hf, weights.up_hf, weights.down_hf, per_expert.permutation
        )
        gate, up, down_weight = self.processor._encode_gate_up(
            layer=self.capture.layer,
            expert=expert,
            weights=weights,
            covariance=covariance_use.matrix,
            search=search,
            gate_bits=4,
            up_bits=4,
        )
        down_covariance, gu_hash = self.processor._down_covariance(
            capture=self.capture,
            shards=self.shards,
            expert=expert,
            gate=gate,
            up=up,
            expected_row_ids=covariance_use.row_ids,
        )
        down = self.processor._encode_down(
            layer=self.capture.layer,
            expert=expert,
            weight_hf=down_weight,
            covariance=down_covariance.matrix,
            bits=4,
            search=search,
            gate_up_provenance=gu_hash,
            bf16_sha256=weights.payload_sha256["down_proj"],
        )
        loss = self.processor._loss(holdout.hidden, reference, gate, up, down)
        diagonal = down_covariance.matrix.diagonal().detach().cpu().tolist()
        return (loss, diagonal) if return_diag else loss

    @staticmethod
    def _sketch(weight, suh: Sequence[float], svh: Sequence[float], seed: int) -> float:
        import torch

        matrix = torch.as_tensor(weight).detach().to("cpu", dtype=torch.float32)
        n, k = matrix.shape
        # Same 512 seeded coordinates and same arithmetic as the original
        # scalar loop, gathered in ONE batched read. The scalar version issued
        # 512 individual element reads per call (a device sync each when the
        # weight lived on CUDA), which dominated the entire search phase.
        rows = []
        cols = []
        for index in range(512):
            coordinate = derive_seed(seed, index, bits=64)
            rows.append(coordinate % n)
            cols.append((coordinate // n) % k)
        row_t = torch.tensor(rows, dtype=torch.long)
        col_t = torch.tensor(cols, dtype=torch.long)
        suh_t = torch.as_tensor(list(suh), dtype=torch.float32)
        svh_t = torch.as_tensor(list(svh), dtype=torch.float32)
        values = (
            matrix[row_t, col_t].double()
            * suh_t[col_t].double()
            * svh_t[row_t].double()
        )
        buckets = torch.zeros(32, dtype=torch.float64)
        buckets.index_add_(0, torch.arange(512, dtype=torch.long) % 32, values)
        absolute = float(values.abs().sum())
        denominator = max(absolute * absolute, 1e-30)
        return float(buckets.square().sum()) / denominator

    def _proxy_expert(
        self, search: LayerSearch, expert: int, diagonal: Sequence[float] | None = None
    ) -> float:
        weights, _, _ = self._prepare(expert)
        per_expert = search.experts[expert]
        gate_w, up_w, down_w = permute_expert_hf(
            weights.gate_hf, weights.up_hf, weights.down_hf, per_expert.permutation
        )
        gate_v, up_v, down_v = self.processor._vectors(search, expert)
        score = self._sketch(gate_w, gate_v.suh, gate_v.svh, derive_seed(expert, "g"))
        score += self._sketch(up_w, up_v.suh, up_v.svh, derive_seed(expert, "u"))
        score += self._sketch(down_w, down_v.suh, down_v.svh, derive_seed(expert, "d"))
        if diagonal is not None:
            ordered = [float(diagonal[index]) for index in per_expert.permutation]
            blocks = [
                sum(ordered[start : start + HAD_K])
                for start in range(0, len(ordered), HAD_K)
            ]
            mean = sum(blocks) / len(blocks)
            score += sum((value - mean) ** 2 for value in blocks) / max(
                mean * mean, 1e-30
            )
        return score

    def _true(self, key: str, search: LayerSearch, expert: int) -> float:
        cached = self._cached_score(key)
        if cached is not None:
            return cached
        return self._record_score(
            key, self._score_expert(search, expert), kind="full-rt"
        )

    def _proxy(
        self, key: str, search: LayerSearch, expert: int, diagonal=None
    ) -> float:
        cached = self._cached_score(key)
        if cached is not None:
            return cached
        return self._record_score(
            key,
            self._proxy_expert(search, expert, diagonal),
            kind="deterministic-sketch",
        )

    def _score_shared(self, search: LayerSearch, key: str) -> float:
        masses = tuple(float(value) for value in self.capture.mass_audit.mass_by_expert)
        denominator = sum(masses[expert] for expert in self.sample)
        scores = self.progress["scores"]
        assert isinstance(scores, dict)
        aggregate_record = scores.get(key)
        if aggregate_record is not None and aggregate_record.get("kind") != (
            "mass-weighted-full-rt"
        ):
            raise ValueError("shared full-RT aggregate kind drift")
        subscores = []
        for expert in self.sample:
            subkey = f"{key}/sample-expert-{expert:03d}"
            subrecord = scores.get(subkey)
            if subrecord is None:
                if aggregate_record is not None:
                    raise ValueError(
                        "shared full-RT aggregate lacks its complete subscore domain"
                    )
                score = self._record_score(
                    subkey,
                    self._score_expert(search, expert),
                    kind="shared-sample-full-rt",
                )
            else:
                if subrecord.get("kind") != "shared-sample-full-rt":
                    raise ValueError("shared full-RT subscore kind drift")
                score = float(subrecord["value"])
            subscores.append((expert, score))
        if len(subscores) != len(self.sample) or {
            expert for expert, _ in subscores
        } != set(self.sample):
            raise AssertionError("shared full-RT subscore domain is incomplete")
        value = sum(masses[expert] * score for expert, score in subscores) / denominator
        if aggregate_record is not None:
            if aggregate_record.get("value") != format(value, ".17g"):
                raise ValueError("shared full-RT aggregate arithmetic drift")
            return value
        return self._record_score(key, value, kind="mass-weighted-full-rt")

    def _proxy_shared(self, search: LayerSearch, key: str) -> float:
        cached = self._cached_score(key)
        if cached is not None:
            return cached
        masses = tuple(float(value) for value in self.capture.mass_audit.mass_by_expert)
        denominator = sum(masses[expert] for expert in self.sample)
        value = (
            sum(
                masses[expert] * self._proxy_expert(search, expert)
                for expert in self.sample
            )
            / denominator
        )
        return self._record_score(key, value, kind="mass-weighted-sketch")

    def _shared_scale_family(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        import torch

        covariance_blocks = torch.zeros(HIDDEN_SIZE // HAD_K, dtype=torch.float64)
        output_blocks = torch.zeros(HIDDEN_SIZE // HAD_K, dtype=torch.float64)
        for expert in self.sample:
            weights, covariance, _ = self._prepare(expert)
            diagonal = (
                covariance.matrix.diagonal().detach().to("cpu", dtype=torch.float64)
            )
            covariance_blocks += diagonal.reshape(-1, HAD_K).mean(dim=1)
            down = (
                torch.as_tensor(weights.down_hf).detach().to("cpu", dtype=torch.float32)
            )
            output_blocks += (
                down.square().mean(dim=1).reshape(-1, HAD_K).mean(dim=1).double()
            )
        return (
            _normalized_quarter_scales(covariance_blocks.tolist()),
            _normalized_quarter_scales(output_blocks.tolist()),
        )

    def _choose_shared(self, experts: dict[int, ExpertSearch]):
        layer = self.capture.layer
        candidates = []
        for draw in range(self.draws):
            suh = rademacher_vector(HIDDEN_SIZE, layer, "shared", "gate_up_suh", draw)
            svh = rademacher_vector(HIDDEN_SIZE, layer, "shared", "down_svh", draw)
            search = self._layer_search(
                shared_suh=suh,
                shared_svh=svh,
                shared_k_scales=(1.0,) * (HIDDEN_SIZE // HAD_K),
                shared_n_scales=(1.0,) * (HIDDEN_SIZE // HAD_K),
                experts=experts,
                shared_draw=draw,
                shared_score=float("inf"),
            )
            proxy = self._proxy_shared(search, f"shared/draw/{draw:02d}/proxy")
            candidates.append((proxy, draw, suh, svh, search))
        promoted = sorted(candidates, key=lambda item: (item[0], item[1]))[:2]
        scored = [
            (
                self._score_shared(item[4], f"shared/draw/{item[1]:02d}/full"),
                *item[1:4],
            )
            for item in promoted
        ]
        score, draw, suh, svh = min(scored, key=lambda item: (item[0], item[1]))

        data_k, data_n = self._shared_scale_family()
        scale_family = (
            ((1.0,) * len(data_k), (1.0,) * len(data_n), "identity"),
            (data_k, data_n, "quarter-rms"),
            (_inverse(data_k), _inverse(data_n), "inverse-quarter-rms"),
        )
        scale_candidates = []
        for k_scales, n_scales, name in scale_family:
            search = self._layer_search(
                shared_suh=suh,
                shared_svh=svh,
                shared_k_scales=k_scales,
                shared_n_scales=n_scales,
                experts=experts,
                shared_draw=draw,
                shared_score=score,
            )
            proxy = self._proxy_shared(search, f"shared/scales/{name}/proxy")
            scale_candidates.append((proxy, name, k_scales, n_scales, search))
        promoted_scales = sorted(scale_candidates, key=lambda item: (item[0], item[1]))[
            :2
        ]
        scored_scales = [
            (
                self._score_shared(item[4], f"shared/scales/{item[1]}/full"),
                *item[1:4],
            )
            for item in promoted_scales
        ]
        final_score, _, k_scales, n_scales = min(
            scored_scales, key=lambda item: (item[0], item[1])
        )
        return suh, svh, k_scales, n_scales, draw, final_score

    @staticmethod
    def _block_energy(weight, *, axis: int) -> tuple[float, ...]:
        import torch

        matrix = torch.as_tensor(weight).detach().to("cpu", dtype=torch.float32)
        values = matrix.square().mean(dim=axis)
        return tuple(values.reshape(-1, HAD_K).mean(dim=1).tolist())

    def _choose_expert(
        self, base: LayerSearch, expert: int, experts: dict[int, ExpertSearch]
    ) -> ExpertSearch:
        incumbent = experts[expert]
        baseline_key = f"expert/{expert:03d}/baseline/full"
        baseline = self._cached_score(baseline_key)
        diagonals = self.progress.setdefault("down_diagonal", {})
        assert isinstance(diagonals, dict)
        if baseline is None or str(expert) not in diagonals:
            observed, diagonal = self._score_expert(base, expert, return_diag=True)
            if baseline is not None and observed != baseline:
                raise ValueError("baseline replay drift while recovering down diagonal")
            scores = self.progress["scores"]
            assert isinstance(scores, dict)
            scores[baseline_key] = {
                "kind": "full-rt",
                "value": format(observed, ".17g"),
            }
            diagonals[str(expert)] = [format(float(value), ".9g") for value in diagonal]
            self._flush()
        else:
            diagonal = [float(value) for value in diagonals[str(expert)]]
        policies = {
            "identity": identity_permutation(),
            "ldlq_visit_descending_diag": descending_diag_permutation(diagonal),
            "stored_descending_diag": stored_descending_diag_permutation(diagonal),
            "energy_balanced": energy_balanced_permutation(diagonal),
            "energy_balanced_contiguous": energy_balanced_permutation(
                diagonal, serpentine=False
            ),
        }
        policy_candidates = []
        for name, permutation in policies.items():
            candidate = replace(
                incumbent, permutation=permutation, permutation_policy=name
            )
            candidate_map = dict(experts)
            candidate_map[expert] = candidate
            search = replace(base, experts=candidate_map)
            proxy = self._proxy(
                f"expert/{expert:03d}/permutation/{name}/proxy",
                search,
                expert,
                diagonal,
            )
            policy_candidates.append((proxy, name, candidate, search))
        _, _, chosen, _ = min(policy_candidates, key=lambda item: (item[0], item[1]))

        draw_candidates = []
        for draw in range(self.draws):
            candidate = replace(
                chosen,
                gate_svh=rademacher_vector(
                    INTERMEDIATE_SIZE, self.capture.layer, expert, "gate_svh", draw
                ),
                up_svh=rademacher_vector(
                    INTERMEDIATE_SIZE, self.capture.layer, expert, "up_svh", draw
                ),
                down_suh=rademacher_vector(
                    INTERMEDIATE_SIZE, self.capture.layer, expert, "down_suh", draw
                ),
                draw=draw,
            )
            candidate_map = dict(experts)
            candidate_map[expert] = candidate
            search = replace(base, experts=candidate_map)
            proxy = self._proxy(
                f"expert/{expert:03d}/draw/{draw:02d}/proxy", search, expert, diagonal
            )
            draw_candidates.append((proxy, draw, candidate, search))
        _, _, chosen, _ = min(draw_candidates, key=lambda item: (item[0], item[1]))

        weights, _, _ = self._prepare(expert)
        gate_energy = self._block_energy(weights.gate_hf, axis=1)
        up_energy = self._block_energy(weights.up_hf, axis=1)
        down_energy = tuple(
            sum(float(value) for value in diagonal[start : start + HAD_K]) / HAD_K
            for start in range(0, len(diagonal), HAD_K)
        )
        data_gate = _normalized_quarter_scales(gate_energy)
        data_up = _normalized_quarter_scales(up_energy)
        data_down = _normalized_quarter_scales(down_energy)
        scale_family = (
            (
                (1.0,) * len(data_gate),
                (1.0,) * len(data_up),
                (1.0,) * len(data_down),
                "identity",
            ),
            (data_gate, data_up, data_down, "quarter-rms"),
            (
                _inverse(data_gate),
                _inverse(data_up),
                _inverse(data_down),
                "inverse-quarter-rms",
            ),
        )
        scale_candidates = []
        for gate_scale, up_scale, down_scale, name in scale_family:
            candidate = replace(
                chosen,
                gate_n_g_scale=gate_scale,
                up_n_g_scale=up_scale,
                down_k_g_scale=down_scale,
            )
            candidate_map = dict(experts)
            candidate_map[expert] = candidate
            search = replace(base, experts=candidate_map)
            proxy = self._proxy(
                f"expert/{expert:03d}/scales/{name}/proxy", search, expert, diagonal
            )
            scale_candidates.append((proxy, name, candidate, search))
        final_proxy, scale_name, chosen, final_search = min(
            scale_candidates, key=lambda item: (item[0], item[1])
        )
        if expert in self.sample:
            final_score = self._true(
                f"expert/{expert:03d}/chosen/{chosen.permutation_policy}/"
                f"draw-{chosen.draw:02d}/scale-{scale_name}/full",
                final_search,
                expert,
            )
            score_kind = "heldout-full-rt"
        else:
            final_score = final_proxy
            score_kind = "deterministic-sketch"
        return replace(
            chosen,
            selection_score=final_score,
            selection_score_kind=score_kind,
        )

    @staticmethod
    def _payload(search: LayerSearch) -> dict[str, object]:
        return {
            "layer": search.layer,
            "draws": search.draws,
            "unverified": search.unverified,
            "pilot_evidence_sha256": search.pilot_evidence_sha256,
            "bindings": dict(search.bindings),
            "shared": {
                "gate_up_suh": list(search.gate_up_suh),
                "down_svh": list(search.down_svh),
                "gate_up_k_g_scale": list(search.gate_up_k_g_scale),
                "down_n_g_scale": list(search.down_n_g_scale),
                "draw": search.shared_draw,
                "heldout_score": format(search.shared_heldout_score, ".17g"),
            },
            "experts": {
                str(expert): {
                    "permutation": list(record.permutation),
                    "permutation_policy": record.permutation_policy,
                    "gate_svh": list(record.gate_svh),
                    "up_svh": list(record.up_svh),
                    "down_suh": list(record.down_suh),
                    "gate_n_g_scale": list(record.gate_n_g_scale),
                    "up_n_g_scale": list(record.up_n_g_scale),
                    "down_k_g_scale": list(record.down_k_g_scale),
                    "draw": record.draw,
                    "selection_score": format(record.selection_score, ".17g"),
                    "selection_score_kind": record.selection_score_kind,
                }
                for expert, record in sorted(search.experts.items())
            },
        }

    def run(self) -> LayerSearch:
        if self.output.exists():
            value = load_layer_search(self.output, require_verified=True)
            if dict(value.bindings) != self.bindings:
                raise ValueError("sealed search artifact provenance drift")
            return value
        experts = {
            expert: _base_expert(self.capture.layer, expert)
            for expert in range(NUM_EXPERTS)
        }
        suh, svh, k_scales, n_scales, draw, score = self._choose_shared(experts)
        base = self._layer_search(
            shared_suh=suh,
            shared_svh=svh,
            shared_k_scales=k_scales,
            shared_n_scales=n_scales,
            experts=experts,
            shared_draw=draw,
            shared_score=score,
        )
        for expert in range(NUM_EXPERTS):
            chosen = self._choose_expert(base, expert, experts)
            experts[expert] = chosen
            base = replace(base, experts=dict(experts))
            results = self.progress["expert_results"]
            assert isinstance(results, dict)
            record = {
                "draw": chosen.draw,
                "permutation_policy": chosen.permutation_policy,
                "selection_score": format(chosen.selection_score, ".17g"),
                "selection_score_kind": chosen.selection_score_kind,
            }
            incumbent = results.get(str(expert))
            if incumbent is not None and incumbent != record:
                raise ValueError("search expert selection drift on resume")
            results[str(expert)] = record
            self._flush()
            # Keep only the shared sample and current expert prepared in RAM.
            if expert not in self.sample:
                self._prepared.pop(expert, None)
        evidence = sha256_file(self.progress_path)
        final = replace(
            base,
            experts=experts,
            pilot_evidence_sha256=evidence,
            unverified=False,
        )
        write_layer_search(self.output, self._payload(final))
        return load_layer_search(self.output, require_verified=True)
