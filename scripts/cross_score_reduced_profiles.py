#!/usr/bin/env python3
"""Cross-score two already encoded profile cells on disjoint full-capture docs.

This script deliberately does not encode anything.  It replays the exact stored
SQG artifacts from two reduced-capture profile cells against BF16 expert outputs
on documents present in the full capture but absent from the reduced plan.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def external_document_epochs(
    full_plan: Mapping[str, object],
    reduced_plan: Mapping[str, object],
    *,
    role: str,
) -> tuple[int, ...]:
    """Return full-plan epochs in ``role`` absent from the entire reduced plan."""

    full_documents = list(full_plan.get("documents", []))
    reduced_documents = list(reduced_plan.get("documents", []))
    if not full_documents or not reduced_documents:
        raise ValueError("both document plans must contain documents")
    reduced_hashes = {str(item["document_sha256"]) for item in reduced_documents}
    if len(reduced_hashes) != len(reduced_documents):
        raise ValueError("reduced document plan contains duplicate document hashes")
    full_hashes = {str(item["document_sha256"]) for item in full_documents}
    if not reduced_hashes <= full_hashes:
        raise ValueError("reduced document plan is not a subset of the full plan")
    selected = tuple(
        int(item["epoch"])
        for item in full_documents
        if str(item["role"]) == role
        and str(item["document_sha256"]) not in reduced_hashes
    )
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("external document epoch selection is empty or duplicated")
    return selected


class FilteredLayerCapture:
    """A role-compatible view restricted to explicit source-capture epochs."""

    def __init__(
        self,
        base: object,
        *,
        source_role: str,
        document_epochs: Sequence[int],
        full_plan_sha256: str,
        reduced_plan_sha256: str,
    ) -> None:
        from src.fresh_pipeline_common import ROLE_TO_ID, canonical_sha256

        if source_role not in ROLE_TO_ID:
            raise ValueError(f"invalid source role {source_role!r}")
        epochs = np.asarray(tuple(int(value) for value in document_epochs), dtype=np.int64)
        if not epochs.size or np.unique(epochs).size != epochs.size:
            raise ValueError("filtered capture requires unique document epochs")
        self.base = base
        self.layer = int(base.layer)
        self.rows = int(base.rows)
        self.doc_epochs = base.doc_epochs
        self.source_role = source_role
        self._source_role_id = int(ROLE_TO_ID[source_role])
        self._epochs = epochs
        self._row_mask = (
            np.asarray(base.role_ids) == self._source_role_id
        ) & np.isin(np.asarray(base.doc_epochs, dtype=np.int64), epochs)
        present = np.unique(np.asarray(base.doc_epochs[self._row_mask], dtype=np.int64))
        if not np.array_equal(np.sort(present), np.sort(epochs)):
            raise ValueError("one or more selected document epochs are absent from capture")
        self._binding: dict[str, object] = {
            "base_capture": base.binding(),
            "filter_schema": "glm52-external-profile-cross-score-filter-v1",
            "source_role": source_role,
            "document_epochs": [int(value) for value in epochs],
            "document_count": int(epochs.size),
            "row_count": int(self._row_mask.sum()),
            "full_plan_sha256": full_plan_sha256,
            "reduced_plan_sha256": reduced_plan_sha256,
        }
        self._binding["filter_id"] = canonical_sha256(self._binding)

    def role_rows(self, role: str) -> np.ndarray:
        if role != self.source_role:
            raise ValueError(
                f"filtered capture exposes only source role {self.source_role!r}"
            )
        return np.flatnonzero(self._row_mask).astype(np.int64, copy=False)

    def load_hidden(self, row_indices: np.ndarray, **kwargs: object) -> torch.Tensor:
        return self.base.load_hidden(row_indices, **kwargs)

    def routed_rows(self, expert: int, role: str):
        from src.fresh_pipeline_calibration import RoutedRows
        from src.fresh_pipeline_common import NUM_EXPERTS, TOPK

        if role != self.source_role:
            raise ValueError("routed role differs from filtered source role")
        if not 0 <= int(expert) < NUM_EXPERTS:
            raise ValueError("expert must lie in [0,255]")
        ids = np.asarray(self.base.topk_ids)
        if ids.shape != (self.rows, TOPK):
            raise ValueError("base top-k routing array has unexpected shape")
        rows, slots = np.nonzero((ids == int(expert)) & self._row_mask[:, None])
        gates = torch.from_numpy(
            np.array(
                self.base.topk_weights[rows, slots], dtype=np.float32, copy=True
            )
        )
        epochs = torch.from_numpy(
            np.array(self.base.doc_epochs[rows], dtype=np.int64, copy=True)
        )
        return RoutedRows(
            role=role,
            expert=int(expert),
            row_indices=rows.astype(np.int64, copy=False),
            route_slots=slots.astype(np.int16, copy=False),
            applied_gates=gates,
            document_epochs=epochs,
        )

    def binding(self) -> dict[str, object]:
        return dict(self._binding)


def _source_from_preflight(preflight_path: Path):
    """Open the already sealed BF16 source without initializing CUDA encoders."""

    from scripts.encode_final_shard import _validated_sealed_shard_identity
    from src.fresh_pipeline_common import load_json_object
    from src.fresh_pipeline_runner import PipelinePaths
    from src.glm52_bf16_source import BF16ExpertSource, SourceValidation, ValidatedShard

    preflight = load_json_object(preflight_path)
    paths = PipelinePaths.resolve(**preflight["paths"])
    source_seal = load_json_object(paths.source_seal)
    index_path = Path(source_seal["index"]["path"]).resolve()
    config_path = Path(source_seal["config"]["path"]).resolve()
    shard_root = Path(source_seal["shard_root"]).resolve()
    with index_path.open("rb") as handle:
        index = json.load(handle)
    weight_map = MappingProxyType(dict(index["weight_map"]))
    expected_identities = preflight.get("source_seal", {}).get(
        "shard_file_identity"
    )
    if not isinstance(expected_identities, dict) or set(expected_identities) != set(
        source_seal["shards"]
    ):
        raise ValueError("sealed BF16 shard identity domain differs")
    shards = {}
    for name, record in source_seal["shards"].items():
        if Path(name).name != name:
            raise ValueError(f"unsafe BF16 shard name: {name!r}")
        shard_path = shard_root / name
        shards[name] = ValidatedShard(
            bytes=int(record["bytes"]),
            sha256=str(record["sha256"]),
            header_sha256=str(record["header_sha256"]),
            selected_tensor_count=int(record["selected_tensor_count"]),
            total_tensor_count=int(record["total_tensor_count"]),
            file_identity=_validated_sealed_shard_identity(
                shard_path, expected_identities[name]
            ),
        )
    validation = SourceValidation(
        index_bytes=int(source_seal["index"]["bytes"]),
        index_sha256=str(source_seal["index"]["sha256"]),
        config_bytes=int(source_seal["config"]["bytes"]),
        config_sha256=str(source_seal["config"]["sha256"]),
        weight_map=weight_map,
        tensor_counts=MappingProxyType(dict(source_seal["tensor_counts"])),
        shards=MappingProxyType(shards),
    )
    source = BF16ExpertSource.__new__(BF16ExpertSource)
    source.index_path = index_path
    source.shard_root = shard_root
    source.config_path = config_path
    source.validation = validation
    source.weight_map = weight_map
    return source, preflight


def _load_or_score(
    *,
    capture: FilteredLayerCapture,
    source: object,
    runtime: object,
    artifact_dir: Path,
    output_dir: Path,
    cell_id: str,
    role: str,
    experts: Sequence[int],
    device: str,
    chunk_rows: int,
) -> dict[str, object]:
    from src.fresh_pipeline_common import atomic_json, canonical_sha256, load_json_object
    from src.fresh_pipeline_evaluation import score_routed_artifacts

    score_path = output_dir / f"{cell_id}.score.json"
    if score_path.is_file():
        score = load_json_object(score_path)
        score_id = score.pop("score_id", None)
        if score_id != canonical_sha256(score):
            raise ValueError(f"existing {cell_id} score ID is invalid")
        score["score_id"] = score_id
        return score
    score = score_routed_artifacts(
        capture,
        source,
        runtime,
        artifact_dir,
        output_dir / f"{cell_id}.scratch",
        role=role,
        experts=experts,
        device=device,
        chunk_rows=chunk_rows,
        cleanup_scratch=True,
    )
    atomic_json(score_path, score)
    return score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--full-capture", required=True)
    parser.add_argument("--full-plan", required=True)
    parser.add_argument("--reduced-plan", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--draw0-artifacts", required=True)
    parser.add_argument("--draw3-artifacts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, default=77)
    parser.add_argument("--role", choices=("fit", "selection", "holdout"), default="selection")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    if args.threads <= 0 or args.chunk_rows <= 0:
        raise ValueError("threads and chunk rows must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    from kquant.sqg_e4m3 import sqg_xor_cheb_t12_bytes
    from src.fresh_pipeline_calibration import LayerCaptureView
    from src.fresh_pipeline_common import (
        atomic_json,
        canonical_sha256,
        load_json_object,
        sha256_file,
    )
    from src.fresh_pipeline_evaluation import paired_document_bootstrap

    full_plan_path = Path(args.full_plan).resolve()
    reduced_plan_path = Path(args.reduced_plan).resolve()
    full_plan = load_json_object(full_plan_path)
    reduced_plan = load_json_object(reduced_plan_path)
    epochs = external_document_epochs(full_plan, reduced_plan, role=args.role)
    base_capture = LayerCaptureView(
        args.full_capture, args.layer, validate=False, verify_hashes=False
    )
    capture = FilteredLayerCapture(
        base_capture,
        source_role=args.role,
        document_epochs=epochs,
        full_plan_sha256=sha256_file(full_plan_path),
        reduced_plan_sha256=sha256_file(reduced_plan_path),
    )
    source, preflight = _source_from_preflight(Path(args.preflight).resolve())
    prereg = load_json_object(args.preregistration)
    experts = tuple(int(value) for value in prereg["selection_panel"])
    if len(experts) != 16 or len(set(experts)) != 16:
        raise ValueError("reduced profile selection panel is not 16 unique experts")
    runtime = SimpleNamespace(
        lut_bytes=lambda bits: sqg_xor_cheb_t12_bytes(bits, device="cpu")
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    draw0 = _load_or_score(
        capture=capture,
        source=source,
        runtime=runtime,
        artifact_dir=Path(args.draw0_artifacts).resolve(),
        output_dir=output_dir,
        cell_id="draw-00__identity",
        role=args.role,
        experts=experts,
        device=args.device,
        chunk_rows=args.chunk_rows,
    )
    draw3 = _load_or_score(
        capture=capture,
        source=source,
        runtime=runtime,
        artifact_dir=Path(args.draw3_artifacts).resolve(),
        output_dir=output_dir,
        cell_id="draw-03__identity",
        role=args.role,
        experts=experts,
        device=args.device,
        chunk_rows=args.chunk_rows,
    )
    bootstrap = paired_document_bootstrap(
        draw0, draw3, seed=770003, iterations=20_000
    )
    aggregate_improvement = (
        float(draw0["aggregate_relative_error"])
        - float(draw3["aggregate_relative_error"])
    )
    aggregate_relative = aggregate_improvement / float(
        draw0["aggregate_relative_error"]
    )
    if not math.isfinite(aggregate_relative):
        raise ValueError("aggregate cross-score comparison is non-finite")
    winner = (
        "draw-03__identity"
        if float(draw3["aggregate_relative_error"])
        < float(draw0["aggregate_relative_error"])
        else "draw-00__identity"
    )
    result: dict[str, Any] = {
        "schema": "glm52-reduced-profile-external-cross-score-v1",
        "complete": True,
        "layer": args.layer,
        "source_role": args.role,
        "construction": "same_reduced_candidate_bytes_scored_on_full_capture_documents_absent_from_reduced_plan",
        "capture_filter": capture.binding(),
        "expert_panel": list(experts),
        "preflight_id": preflight["preflight_id"],
        "cells": {
            "draw-00__identity": {
                "score_id": draw0["score_id"],
                "aggregate_relative_error": draw0["aggregate_relative_error"],
                "mean_document_relative_error": draw0["mean_document_relative_error"],
            },
            "draw-03__identity": {
                "score_id": draw3["score_id"],
                "aggregate_relative_error": draw3["aggregate_relative_error"],
                "mean_document_relative_error": draw3["mean_document_relative_error"],
            },
        },
        "draw3_minus_draw0": {
            "aggregate_relative_error_change": -aggregate_improvement,
            "aggregate_relative_percent_change": -100.0 * aggregate_relative,
        },
        "paired_document_bootstrap_draw0_minus_draw3": bootstrap,
        "winner_by_aggregate_relative_error": winner,
        "causal_scope": "candidate_bytes_held_fixed; evaluates external-document generalization, not corpus-size causality",
    }
    result["result_id"] = canonical_sha256(result)
    result_path = output_dir / "cross_score.json"
    atomic_json(result_path, result, overwrite=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
