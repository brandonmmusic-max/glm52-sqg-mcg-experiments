"""Exact routed functional scoring for fresh SQG candidates.

The evaluator accumulates routed expert contributions before measuring error,
so cross-expert terms are retained.  Selection and holdout are explicit roles.
Temporary memmaps are journaled because an interrupted in-place accumulation
cannot safely be replayed from the middle; an in-flight expert causes only the
evaluator-owned scratch reduction to restart, never an encoded artifact.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .fresh_pipeline_artifacts import (
    expert_stem,
    load_decoded_expert,
    validate_expert_artifact,
)
from .fresh_pipeline_calibration import LayerCaptureView
from .fresh_pipeline_common import (
    HIDDEN,
    NUM_EXPERTS,
    ROLES,
    atomic_json,
    canonical_sha256,
    load_json_object,
    sha256_file,
)
from .glm52_bf16_source import BF16ExpertSource
from .glm52_fresh_sqg import KQuantRuntime


ROUTED_REDUCTION_SCHEMA = "glm52-fresh-sqg-routed-reduction-v1"


def _chunks(total: int, rows: int):
    if rows <= 0:
        raise ValueError("chunk row count must be positive")
    for begin in range(0, total, rows):
        yield begin, min(total, begin + rows)


def _candidate_output(
    hidden: torch.Tensor,
    decoded: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    gate = decoded["gate_proj"]
    up = decoded["up_proj"]
    down = decoded["down_proj"]
    return (F.silu(hidden @ gate) * (hidden @ up)) @ down


def _reference_output(
    hidden: torch.Tensor, weights: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    gate = weights["gate_proj"]
    up = weights["up_proj"]
    down = weights["down_proj"]
    middle = F.silu(F.linear(hidden, gate)) * F.linear(hidden, up)
    return F.linear(middle, down)


def _artifact_binding(
    artifact_dir: Path,
    *,
    layer: int,
    experts: Sequence[int],
) -> tuple[dict[int, dict[str, Any]], dict[str, str]]:
    manifests: dict[int, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for expert in experts:
        path = artifact_dir / f"{expert_stem(layer, expert)}.json"
        item = validate_expert_artifact(path)
        if int(item["layer"]) != layer or int(item["expert"]) != expert:
            raise ValueError("routed-score expert artifact binding differs")
        manifests[expert] = item
        hashes[str(expert)] = sha256_file(path)
    return manifests, hashes


def _new_reduction(
    scratch_dir: Path,
    *,
    rows: int,
    binding: Mapping[str, object],
) -> tuple[np.memmap, np.memmap, dict[str, Any]]:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    error_path = scratch_dir / "routed_error.f32le.bin"
    reference_path = scratch_dir / "routed_reference.f32le.bin"
    for path in (error_path, reference_path):
        path.unlink(missing_ok=True)
    error = np.memmap(error_path, mode="w+", dtype="<f4", shape=(rows, HIDDEN))
    reference = np.memmap(
        reference_path, mode="w+", dtype="<f4", shape=(rows, HIDDEN)
    )
    error[:] = 0
    reference[:] = 0
    error.flush()
    reference.flush()
    journal: dict[str, Any] = {
        "schema": ROUTED_REDUCTION_SCHEMA,
        "binding": dict(binding),
        "rows": rows,
        "hidden": HIDDEN,
        "dtype": "float32_le",
        "completed_experts": [],
        "inflight_expert": None,
        "complete": False,
    }
    atomic_json(scratch_dir / "journal.json", journal, overwrite=True)
    return error, reference, journal


def _open_reduction(
    scratch_dir: Path,
    *,
    rows: int,
    experts: Sequence[int],
    binding: Mapping[str, object],
) -> tuple[np.memmap, np.memmap, dict[str, Any]]:
    journal_path = scratch_dir / "journal.json"
    error_path = scratch_dir / "routed_error.f32le.bin"
    reference_path = scratch_dir / "routed_reference.f32le.bin"
    if not journal_path.exists():
        return _new_reduction(scratch_dir, rows=rows, binding=binding)
    journal = load_json_object(journal_path)
    completed = [int(value) for value in journal.get("completed_experts", [])]
    valid_prefix = list(experts[: len(completed)])
    expected_bytes = rows * HIDDEN * np.dtype("<f4").itemsize
    invalid = (
        journal.get("schema") != ROUTED_REDUCTION_SCHEMA
        or journal.get("binding") != dict(binding)
        or journal.get("rows") != rows
        or journal.get("hidden") != HIDDEN
        or journal.get("dtype") != "float32_le"
        or completed != valid_prefix
        or journal.get("inflight_expert") is not None
        or not error_path.is_file()
        or not reference_path.is_file()
        or error_path.stat().st_size != expected_bytes
        or reference_path.stat().st_size != expected_bytes
    )
    if invalid:
        # These are exact, evaluator-owned scratch paths.  Restarting the
        # reduction is the only correct response to an uncertain partial add.
        return _new_reduction(scratch_dir, rows=rows, binding=binding)
    error = np.memmap(error_path, mode="r+", dtype="<f4", shape=(rows, HIDDEN))
    reference = np.memmap(
        reference_path, mode="r+", dtype="<f4", shape=(rows, HIDDEN)
    )
    return error, reference, journal


def document_scores_from_routed_aggregates(
    error_aggregate: np.ndarray,
    reference_aggregate: np.ndarray,
    document_epochs: np.ndarray,
    *,
    chunk_rows: int = 256,
) -> dict[str, object]:
    """Square already-summed routed outputs and reduce them by document."""

    if (
        error_aggregate.ndim != 2
        or reference_aggregate.shape != error_aggregate.shape
        or document_epochs.shape != (error_aggregate.shape[0],)
        or not error_aggregate.shape[0]
        or not error_aggregate.shape[1]
    ):
        raise ValueError("routed aggregate score shapes differ")
    epochs = np.asarray(document_epochs, dtype=np.int64)
    documents = np.unique(epochs)
    doc_index = np.searchsorted(documents, epochs)
    error_energy = np.zeros(documents.size, dtype=np.float64)
    reference_energy = np.zeros(documents.size, dtype=np.float64)
    rows_per_document = np.zeros(documents.size, dtype=np.int64)
    for begin, end in _chunks(int(epochs.size), chunk_rows):
        error = np.asarray(error_aggregate[begin:end], dtype=np.float64)
        reference = np.asarray(reference_aggregate[begin:end], dtype=np.float64)
        local_error = np.square(error).sum(axis=1)
        local_reference = np.square(reference).sum(axis=1)
        if not np.isfinite(local_error).all() or not np.isfinite(local_reference).all():
            raise ValueError("routed reduction produced non-finite energy")
        local_docs = doc_index[begin:end]
        np.add.at(error_energy, local_docs, local_error)
        np.add.at(reference_energy, local_docs, local_reference)
        np.add.at(rows_per_document, local_docs, 1)
    informative = reference_energy > 1e-30
    if not informative.any():
        raise ValueError("routed reference energy is empty")
    documents = documents[informative]
    error_energy = error_energy[informative]
    reference_energy = reference_energy[informative]
    rows_per_document = rows_per_document[informative]
    relative = error_energy / reference_energy
    records = [
        {
            "document_epoch": int(epoch),
            "rows": int(rows),
            "error_energy": float(err),
            "reference_energy": float(ref),
            "relative_error": float(rel),
        }
        for epoch, rows, err, ref, rel in zip(
            documents,
            rows_per_document,
            error_energy,
            reference_energy,
            relative,
        )
    ]
    return {
        "document_scores": records,
        "aggregate_error_energy": float(error_energy.sum()),
        "aggregate_reference_energy": float(reference_energy.sum()),
        "aggregate_relative_error": float(
            error_energy.sum() / reference_energy.sum()
        ),
        "mean_document_relative_error": float(relative.mean()),
        "median_document_relative_error": float(np.median(relative)),
    }


def score_routed_artifacts(
    capture: LayerCaptureView,
    source: BF16ExpertSource,
    runtime: KQuantRuntime,
    artifact_dir: str | Path,
    scratch_dir: str | Path,
    *,
    role: str,
    experts: Sequence[int],
    device: str | torch.device,
    chunk_rows: int = 256,
    cleanup_scratch: bool = False,
) -> dict[str, object]:
    """Measure exact aggregate routed error for one explicit document role."""

    if role not in ROLES:
        raise ValueError(f"invalid routed-score role {role!r}")
    ordered = tuple(int(value) for value in experts)
    if (
        not ordered
        or len(set(ordered)) != len(ordered)
        or any(value < 0 or value >= NUM_EXPERTS for value in ordered)
    ):
        raise ValueError("routed-score expert panel is invalid")
    target = torch.device(device)
    artifact_root = Path(artifact_dir).resolve()
    scratch_root = Path(scratch_dir).resolve()
    manifests, manifest_hashes = _artifact_binding(
        artifact_root, layer=capture.layer, experts=ordered
    )
    role_rows = capture.role_rows(role)
    if not role_rows.size:
        raise ValueError(f"{role} capture rows are empty")
    compact = np.full(capture.rows, -1, dtype=np.int64)
    compact[role_rows] = np.arange(role_rows.size, dtype=np.int64)
    binding: dict[str, object] = {
        "capture": capture.binding(),
        "role": role,
        "experts": list(ordered),
        "artifact_manifest_sha256": manifest_hashes,
    }
    binding["binding_id"] = canonical_sha256(binding)
    error_map, reference_map, journal = _open_reduction(
        scratch_root,
        rows=int(role_rows.size),
        experts=ordered,
        binding=binding,
    )
    completed = [int(value) for value in journal["completed_experts"]]
    lut_by_bits = {
        bits: runtime.lut_bytes(bits).detach().cpu().contiguous()
        for bits in (3, 4)
    }
    journal_path = scratch_root / "journal.json"
    for expert in ordered[len(completed) :]:
        routed = capture.routed_rows(expert, role)
        if routed.rows == 0:
            raise ValueError(
                f"L{capture.layer} E{expert}: selected expert has no {role} routes"
            )
        journal["inflight_expert"] = expert
        atomic_json(journal_path, journal, overwrite=True)
        decoded_cpu = load_decoded_expert(
            artifact_root / f"{expert_stem(capture.layer, expert)}.json",
            lut_by_bits=lut_by_bits,
        )
        decoded = {
            name: value.to(device=target, dtype=torch.float32)
            for name, value in decoded_cpu.items()
        }
        source_weights = source.load_expert_bf16(
            capture.layer, expert, device="cpu"
        )
        weights = {
            "gate_proj": source_weights.gate_hf.to(
                device=target, dtype=torch.float32
            ),
            "up_proj": source_weights.up_hf.to(
                device=target, dtype=torch.float32
            ),
            "down_proj": source_weights.down_hf.to(
                device=target, dtype=torch.float32
            ),
        }
        for begin, end in _chunks(routed.rows, chunk_rows):
            absolute = routed.row_indices[begin:end]
            positions = compact[absolute]
            if bool((positions < 0).any()):
                raise RuntimeError("routed role position map is inconsistent")
            hidden = capture.load_hidden(
                absolute, device=target, dtype=torch.float32
            )
            with torch.no_grad():
                candidate = _candidate_output(hidden, decoded)
                reference = _reference_output(hidden, weights)
                gates = routed.applied_gates[begin:end].to(
                    device=target, dtype=torch.float32
                )[:, None]
                delta = ((candidate - reference) * gates).cpu().numpy()
                routed_reference = (reference * gates).cpu().numpy()
            error_map[positions] += delta
            reference_map[positions] += routed_reference
        error_map.flush()
        reference_map.flush()
        completed.append(expert)
        journal["completed_experts"] = list(completed)
        journal["inflight_expert"] = None
        atomic_json(journal_path, journal, overwrite=True)
        del decoded, decoded_cpu, weights, source_weights
        if target.type == "cuda":
            torch.cuda.empty_cache()
    if tuple(completed) != ordered:
        raise RuntimeError("routed reduction did not complete its expert panel")

    energies = document_scores_from_routed_aggregates(
        error_map,
        reference_map,
        np.asarray(capture.doc_epochs[role_rows], dtype=np.int64),
        chunk_rows=chunk_rows,
    )
    document_records = energies["document_scores"]
    report: dict[str, object] = {
        "schema": "glm52-fresh-sqg-routed-functional-score-v1",
        "complete": True,
        "binding": binding,
        "layer": capture.layer,
        "role": role,
        "expert_panel": list(ordered),
        "expert_count": len(ordered),
        "role_rows": int(role_rows.size),
        "informative_documents": len(document_records),
        "aggregate_error_energy": energies["aggregate_error_energy"],
        "aggregate_reference_energy": energies["aggregate_reference_energy"],
        "aggregate_relative_error": energies["aggregate_relative_error"],
        "mean_document_relative_error": energies["mean_document_relative_error"],
        "median_document_relative_error": energies[
            "median_document_relative_error"
        ],
        "document_scores": document_records,
        "aggregation": "sum_routed_expert_outputs_then_square_retains_cross_terms",
        "source_reference": "official_bf16_replayed_fp32",
        "candidate_decode": "independent_stored_fp16_sqg",
    }
    report["score_id"] = canonical_sha256(report)
    journal["complete"] = True
    journal["score_id"] = report["score_id"]
    atomic_json(journal_path, journal, overwrite=True)
    if cleanup_scratch:
        # Only exact files owned by this reduction are removed.  The score is
        # fully self-contained and binds every encoded artifact manifest.
        del error_map, reference_map
        for name in (
            "routed_error.f32le.bin",
            "routed_reference.f32le.bin",
            "journal.json",
        ):
            (scratch_root / name).unlink(missing_ok=True)
    return report


def paired_document_bootstrap(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    seed: int,
    iterations: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Bootstrap paired document improvement (baseline minus candidate)."""

    if iterations < 1_000:
        raise ValueError("paired bootstrap requires at least 1000 iterations")
    if not 0.5 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie in (0.5,1)")
    base = {
        int(item["document_epoch"]): float(item["relative_error"])
        for item in baseline["document_scores"]
    }
    trial = {
        int(item["document_epoch"]): float(item["relative_error"])
        for item in candidate["document_scores"]
    }
    if base.keys() != trial.keys() or not base:
        raise ValueError("paired bootstrap document domains differ")
    epochs = np.array(sorted(base), dtype=np.int64)
    differences = np.array(
        [base[int(epoch)] - trial[int(epoch)] for epoch in epochs],
        dtype=np.float64,
    )
    if not np.isfinite(differences).all():
        raise ValueError("paired bootstrap difference is non-finite")
    rng = np.random.default_rng(int(seed))
    means = np.empty(iterations, dtype=np.float64)
    draw_chunk = 256
    for begin, end in _chunks(iterations, draw_chunk):
        indices = rng.integers(
            0, differences.size, size=(end - begin, differences.size)
        )
        means[begin:end] = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    result: dict[str, object] = {
        "schema": "glm52-fresh-sqg-paired-document-bootstrap-v1",
        "iterations": iterations,
        "confidence": confidence,
        "seed": int(seed),
        "documents": int(differences.size),
        "improvement_definition": "baseline_document_relative_error_minus_candidate",
        "observed_mean_improvement": float(differences.mean()),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "lower_bound_gt_zero": bool(lower > 0.0),
        "numpy_version": np.__version__,
    }
    if any(
        not math.isfinite(float(result[key]))
        for key in (
            "observed_mean_improvement",
            "lower_bound",
            "upper_bound",
        )
    ):
        raise ValueError("paired bootstrap result is non-finite")
    result["bootstrap_id"] = canonical_sha256(result)
    return result
