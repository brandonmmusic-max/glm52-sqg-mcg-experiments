"""Dense Hessians from the fresh GLM-5.2 live-router capture.

``H13`` is layer-global and gate-square weighted over fit documents only.  A
down-projection ``H2`` is expert-local and must be rebuilt for each exact
decoded gate/up candidate by replaying GLM's ``SiLU(gate) * up``.  No global
or source-upstream down Hessian is exposed here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from .calibration_capture import HIDDEN, NUM_EXPERTS, SELECTED_LAYERS, TOPK, layer_dir
from .calibration_plan import ROLE_TO_ID


INTERMEDIATE = 2_048
H2_LOCAL_ALPHA_CAP = 0.75
H2_MIN_FIT_DOCUMENTS = 6
H2_MIN_ROUTED_ROWS = 6
FROZEN_KQUANT_CANDIDATE_HESSIAN_SHA256 = (
    "bb4c516b750e72df8d5d1898172f51cd9fa292f83734971bad2fb4ed3c27872c"
)


@lru_cache(maxsize=1)
def _frozen_kquant_adaptive_identity_shrinkage():
    """Load the hash-sealed KQuant implementation, not a local approximation."""

    path = (
        Path(__file__).resolve().parents[1]
        / "kquant"
        / "kquant"
        / "candidate_hessian.py"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FROZEN_KQUANT_CANDIDATE_HESSIAN_SHA256:
        raise RuntimeError(
            "frozen KQuant candidate_hessian.py differs; refusing H2 shrinkage"
        )
    spec = importlib.util.spec_from_file_location(
        "_glm52_frozen_kquant_candidate_hessian", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen KQuant candidate Hessian module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.adaptive_identity_shrinkage


def apply_frozen_h2_shrinkage(
    local_covariance: torch.Tensor,
    route_gate_squares: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the exact hash-sealed KQuant H2 shrinkage policy."""

    shrink = _frozen_kquant_adaptive_identity_shrinkage()
    return shrink(
        local_covariance,
        route_gate_squares,
        max_local_alpha=H2_LOCAL_ALPHA_CAP,
    )


def validate_h2_support(*, routed_rows: int, fit_documents: int) -> None:
    """Fail rather than borrow covariance when one expert lacks fit support."""

    if int(routed_rows) < H2_MIN_ROUTED_ROWS:
        raise ValueError(
            f"{routed_rows} fit routes is below the frozen support floor "
            f"{H2_MIN_ROUTED_ROWS}"
        )
    if int(fit_documents) < H2_MIN_FIT_DOCUMENTS:
        raise ValueError(
            f"{fit_documents} fit documents is below the frozen support floor "
            f"{H2_MIN_FIT_DOCUMENTS}"
        )


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and exact contiguous little-endian payload."""

    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    if value.dtype == torch.bfloat16:
        payload = value.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    else:
        array = value.numpy()
        if array.dtype.itemsize > 1:
            array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        payload = array.tobytes()
    digest.update(payload)
    return digest.hexdigest()


def iter_layer_rows(
    capture_dir: str | Path,
    layer: int,
    *,
    role: str,
    chunk_rows: int = 512,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield copied CPU tensors for one document role from the sealed raw ABI."""

    if layer not in SELECTED_LAYERS:
        raise ValueError(f"layer {layer} is outside the frozen pilot")
    if role not in ROLE_TO_ID:
        raise ValueError(f"invalid calibration role {role!r}")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    directory = layer_dir(capture_dir, layer)
    manifest = json.loads((directory / "layer_manifest.json").read_text(encoding="utf-8"))
    total = int(manifest["tokens"])
    hidden_bits = np.memmap(
        directory / "hidden.bf16.bin",
        mode="r",
        dtype="<u2",
        shape=(total, HIDDEN),
    )
    ids = np.memmap(
        directory / "topk_ids.u8.bin",
        mode="r",
        dtype="u1",
        shape=(total, TOPK),
    )
    weights = np.memmap(
        directory / "topk_weights.f32le.bin",
        mode="r",
        dtype="<f4",
        shape=(total, TOPK),
    )
    epochs = np.memmap(directory / "doc_epochs.u32le.bin", mode="r", dtype="<u4")
    roles = np.memmap(directory / "role_ids.u8.bin", mode="r", dtype="u1")
    role_id = ROLE_TO_ID[role]
    for begin in range(0, total, chunk_rows):
        end = min(total, begin + chunk_rows)
        selected = np.flatnonzero(np.asarray(roles[begin:end]) == role_id)
        if not selected.size:
            continue
        absolute = begin + selected
        # Memmaps are read-only.  Copy before torch views the BF16 bit pattern.
        hidden_u16 = torch.from_numpy(np.asarray(hidden_bits[absolute]).copy())
        yield {
            "hidden": hidden_u16.view(torch.bfloat16),
            "topk_ids": torch.from_numpy(np.asarray(ids[absolute]).copy()),
            "topk_weights": torch.from_numpy(np.asarray(weights[absolute]).copy()),
            "doc_epochs": torch.from_numpy(
                np.asarray(epochs[absolute]).astype(np.int64, copy=True)
            ),
        }


def _new_accumulator(dimension: int, device: torch.device) -> torch.Tensor:
    if dimension <= 0:
        raise ValueError("Hessian dimension must be positive")
    return torch.zeros((dimension, dimension), dtype=torch.float32, device=device)


def weighted_dense_covariance(
    rows: torch.Tensor,
    importance: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Small in-memory reference for the normalized weighted dense covariance."""

    if rows.ndim != 2 or importance.ndim != 1 or rows.shape[0] != importance.shape[0]:
        raise ValueError("weighted covariance row/weight shapes differ")
    rows = rows.float()
    importance = importance.float()
    if (
        not rows.numel()
        or not bool(torch.isfinite(rows).all().item())
        or not bool(torch.isfinite(importance).all().item())
        or bool((importance < 0).any().item())
    ):
        raise ValueError("weighted covariance inputs are empty or invalid")
    denominator = float(importance.double().sum().item())
    accumulator = rows.T @ (rows * importance[:, None])
    return _finish_covariance(accumulator, denominator, return_cpu=False), denominator


def expert_route_selection(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row indices and matching live gate for one nonduplicated expert."""

    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("top-k ID/weight shapes differ")
    matches = topk_ids.to(torch.int64) == int(expert)
    if bool((matches.sum(dim=1) > 1).any().item()):
        raise ValueError("a route row contains the expert more than once")
    selected = torch.nonzero(matches.any(dim=1), as_tuple=False).flatten()
    if not selected.numel():
        return selected, topk_weights.new_empty((0,))
    positions = matches[selected].to(torch.int64).argmax(dim=1)
    return selected, topk_weights[selected, positions]


def glm_candidate_intermediate(
    hidden: torch.Tensor,
    decoded_gate: torch.Tensor,
    decoded_up: torch.Tensor,
) -> torch.Tensor:
    """Reference GLM expert replay used to condition a down candidate."""

    if hidden.ndim != 2 or decoded_gate.ndim != 2 or decoded_up.shape != decoded_gate.shape:
        raise ValueError("invalid GLM candidate replay shapes")
    if hidden.shape[1] != decoded_gate.shape[1]:
        raise ValueError("hidden width and decoded upstream input width differ")
    return F.silu(F.linear(hidden.float(), decoded_gate.float())) * F.linear(
        hidden.float(), decoded_up.float()
    )


def _finish_covariance(
    accumulator: torch.Tensor,
    denominator: float,
    *,
    return_cpu: bool,
) -> torch.Tensor:
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("weighted Hessian has no positive finite support")
    value = accumulator.div_(denominator)
    value = ((value + value.T) * 0.5).contiguous()
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("weighted Hessian is non-finite")
    return value.cpu() if return_cpu else value


def build_h13_dense(
    capture_dir: str | Path,
    layer: int,
    *,
    device: str | torch.device,
    chunk_rows: int = 256,
    return_cpu: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Build fit-only layer-global ``H13`` with ``sum_j gate_j**2`` weights."""

    device = torch.device(device)
    accumulator = _new_accumulator(HIDDEN, device)
    denominator = 0.0
    rows = 0
    documents: set[int] = set()
    for chunk in iter_layer_rows(
        capture_dir, layer, role="fit", chunk_rows=chunk_rows
    ):
        hidden = chunk["hidden"].to(device=device, dtype=torch.float32)
        weights = chunk["topk_weights"].to(device=device, dtype=torch.float32)
        if not bool(torch.isfinite(hidden).all().item()):
            raise ValueError(f"layer {layer}: fit hidden states are non-finite")
        importance = weights.square().sum(dim=1)
        accumulator.addmm_(hidden.T, hidden * importance[:, None])
        denominator += float(importance.double().sum().item())
        rows += int(hidden.shape[0])
        documents.update(int(value) for value in chunk["doc_epochs"].tolist())
    hessian = _finish_covariance(
        accumulator, denominator, return_cpu=return_cpu
    )
    evidence = {
        "kind": "glm52-fresh-sqg-h13",
        "layer": int(layer),
        "role": "fit",
        "scope": "layer-global",
        "dimension": HIDDEN,
        "rows": rows,
        "documents": len(documents),
        "weighting": "per-token sum over routed experts of exact applied_gate**2",
        "gate_sq_denominator": denominator,
        "normalization": "sum(weight * x xT) / sum(weight)",
        "hessian_fp32_sha256": tensor_sha256(hessian),
        "fallback": None,
    }
    return hessian, evidence


def build_candidate_h2_dense(
    capture_dir: str | Path,
    layer: int,
    expert: int,
    decoded_gate: torch.Tensor,
    decoded_up: torch.Tensor,
    *,
    device: str | torch.device,
    chunk_rows: int = 512,
    return_cpu: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Build fit-only expert-local H2 from the exact decoded SQG upstream pair.

    The caller is responsible for passing the independently decoded candidate
    matrices at the stored-FP16 scale boundary.  Their exact tensor hashes are
    bound into the returned evidence.
    """

    layer = int(layer)
    expert = int(expert)
    if layer not in SELECTED_LAYERS or not 0 <= expert < NUM_EXPERTS:
        raise ValueError("layer/expert is outside the frozen GLM pilot")
    expected_shape = (INTERMEDIATE, HIDDEN)
    if tuple(decoded_gate.shape) != expected_shape or tuple(decoded_up.shape) != expected_shape:
        raise ValueError(
            f"decoded gate/up must each have shape {expected_shape}, got "
            f"{tuple(decoded_gate.shape)} / {tuple(decoded_up.shape)}"
        )
    if decoded_gate.device.type != "cpu" or decoded_up.device.type != "cpu":
        raise ValueError("hash-bound decoded gate/up inputs must be CPU tensors")
    if not bool(torch.isfinite(decoded_gate.float()).all().item()) or not bool(
        torch.isfinite(decoded_up.float()).all().item()
    ):
        raise ValueError("decoded gate/up candidate contains non-finite values")
    gate_hash = tensor_sha256(decoded_gate)
    up_hash = tensor_sha256(decoded_up)
    device = torch.device(device)
    gate = decoded_gate.to(device=device, dtype=torch.float32)
    up = decoded_up.to(device=device, dtype=torch.float32)
    accumulator = _new_accumulator(INTERMEDIATE, device)
    denominator = 0.0
    routed_rows = 0
    documents: set[int] = set()
    importance_chunks: list[torch.Tensor] = []
    for chunk in iter_layer_rows(
        capture_dir, layer, role="fit", chunk_rows=chunk_rows
    ):
        ids = chunk["topk_ids"].to(torch.int64)
        selected, selected_gates = expert_route_selection(
            ids, chunk["topk_weights"], expert
        )
        if not selected.numel():
            continue
        route_gates = selected_gates.to(
            device=device, dtype=torch.float32
        )
        hidden = chunk["hidden"][selected].to(device=device, dtype=torch.float32)
        if not bool(torch.isfinite(hidden).all().item()):
            raise ValueError(
                f"layer {layer} expert {expert}: fit hidden states are non-finite"
            )
        # GLM expert intermediate, using the decoded candidate that will feed
        # this exact down projection.  This must be rerun for every candidate.
        intermediate = glm_candidate_intermediate(hidden, gate, up)
        importance = route_gates.square()
        importance_chunks.append(importance.detach().to(device="cpu", dtype=torch.float32))
        accumulator.addmm_(
            intermediate.T, intermediate * importance[:, None]
        )
        denominator += float(importance.double().sum().item())
        routed_rows += int(selected.numel())
        documents.update(int(value) for value in chunk["doc_epochs"][selected].tolist())
    try:
        validate_h2_support(
            routed_rows=routed_rows, fit_documents=len(documents)
        )
    except ValueError as error:
        raise ValueError(f"layer {layer} expert {expert}: {error}") from error
    raw_local = _finish_covariance(
        accumulator, denominator, return_cpu=False
    )
    weights = torch.cat(importance_chunks)
    hessian, shrinkage = apply_frozen_h2_shrinkage(
        raw_local,
        weights,
    )
    if return_cpu:
        hessian = hessian.cpu()
    raw_local_alpha = 1.0 - float(shrinkage["oas_shrinkage"])
    evidence = {
        "kind": "glm52-fresh-sqg-candidate-h2",
        "layer": layer,
        "expert": expert,
        "role": "fit",
        "scope": "expert-local",
        "dimension": INTERMEDIATE,
        "routed_rows": routed_rows,
        "documents": len(documents),
        "weighting": "exact applied route gate**2",
        "gate_sq_denominator": denominator,
        "support_gate": {
            "passed": True,
            "minimum_routed_rows": H2_MIN_ROUTED_ROWS,
            "minimum_fit_documents": H2_MIN_FIT_DOCUMENTS,
        },
        "upstream_replay": "silu(x @ decoded_gate.T) * (x @ decoded_up.T)",
        "decoded_gate": {
            "dtype": str(decoded_gate.dtype),
            "shape": list(decoded_gate.shape),
            "sha256": gate_hash,
        },
        "decoded_up": {
            "dtype": str(decoded_up.dtype),
            "shape": list(decoded_up.shape),
            "sha256": up_hash,
        },
        "normalization": "sum(gate**2 * y yT) / sum(gate**2)",
        "raw_local_hessian_fp32_sha256": tensor_sha256(raw_local),
        "shrinkage_policy": "weighted_oas_scaled_identity",
        "shrinkage_source": {
            "implementation": "kquant.candidate_hessian.adaptive_identity_shrinkage",
            "candidate_hessian_py_sha256": (
                FROZEN_KQUANT_CANDIDATE_HESSIAN_SHA256
            ),
        },
        "effective_sample_size": float(shrinkage["effective_sample_size"]),
        "oas_shrinkage": float(shrinkage["oas_shrinkage"]),
        "raw_local_alpha": raw_local_alpha,
        "local_alpha": float(shrinkage["local_alpha"]),
        "local_alpha_cap": float(shrinkage["max_local_alpha"]),
        "identity_scale": float(shrinkage["identity_scale"]),
        "prior": "candidate/expert-local trace-scaled identity only",
        "hessian_fp32_sha256": tensor_sha256(hessian),
        "borrowing": None,
        "global_fallback": None,
        "fallback": None,
    }
    return hessian, evidence
