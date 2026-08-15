#!/usr/bin/env python3
"""Replay GLM-5.2 hidden states through its BF16 head and compare distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def statistics(values: np.ndarray) -> dict[str, float]:
    ordered = np.sort(values.astype(np.float64, copy=False))
    worst = max(1, math.ceil(0.01 * len(ordered)))
    return {
        "mean": float(ordered.mean()),
        "median": float(np.median(ordered)),
        "p95": float(np.quantile(ordered, 0.95)),
        "p99": float(np.quantile(ordered, 0.99)),
        "p99_9": float(np.quantile(ordered, 0.999)),
        "cvar_worst_1pct": float(ordered[-worst:].mean()),
        "max": float(ordered[-1]),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def pass_one(
    hidden: torch.Tensor,
    reference: torch.Tensor,
    weight: torch.Tensor,
    *,
    vocab_size: int,
    vocab_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = hidden.shape[0]
    device = hidden.device
    reference_log_z = torch.full((rows,), -torch.inf, device=device)
    candidate_log_z = torch.full_like(reference_log_z, -torch.inf)
    reference_top_values = torch.full_like(reference_log_z, -torch.inf)
    candidate_top_values = torch.full_like(reference_log_z, -torch.inf)
    reference_top_ids = torch.zeros(rows, dtype=torch.int64, device=device)
    candidate_top_ids = torch.zeros_like(reference_top_ids)
    for vocab_start in range(0, vocab_size, vocab_chunk):
        vocab_end = min(vocab_start + vocab_chunk, vocab_size)
        reference_logits = reference[:, vocab_start:vocab_end].to(
            device, non_blocking=True
        )
        candidate_logits = F.linear(hidden, weight[vocab_start:vocab_end])
        reference_logits = reference_logits.float()
        candidate_logits = candidate_logits.float()
        reference_log_z = torch.logaddexp(
            reference_log_z, torch.logsumexp(reference_logits, dim=-1)
        )
        candidate_log_z = torch.logaddexp(
            candidate_log_z, torch.logsumexp(candidate_logits, dim=-1)
        )
        reference_values, reference_ids = reference_logits.max(dim=-1)
        candidate_values, candidate_ids = candidate_logits.max(dim=-1)
        reference_update = reference_values > reference_top_values
        candidate_update = candidate_values > candidate_top_values
        reference_top_values = torch.where(
            reference_update, reference_values, reference_top_values
        )
        candidate_top_values = torch.where(
            candidate_update, candidate_values, candidate_top_values
        )
        reference_top_ids = torch.where(
            reference_update, reference_ids + vocab_start, reference_top_ids
        )
        candidate_top_ids = torch.where(
            candidate_update, candidate_ids + vocab_start, candidate_top_ids
        )
    return (
        reference_log_z,
        candidate_log_z,
        reference_top_ids,
        candidate_top_ids,
    )


def compare_block(
    hidden: torch.Tensor,
    reference: torch.Tensor,
    weight: torch.Tensor,
    *,
    vocab_size: int,
    vocab_chunk: int,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    (
        reference_log_z,
        candidate_log_z,
        reference_top_ids,
        candidate_top_ids,
    ) = pass_one(
        hidden,
        reference,
        weight,
        vocab_size=vocab_size,
        vocab_chunk=vocab_chunk,
    )
    rows = hidden.shape[0]
    kl = torch.zeros(rows, dtype=torch.float64, device=hidden.device)
    js = torch.zeros_like(kl)
    absolute_error_sum = 0.0
    absolute_error_max = 0.0
    for vocab_start in range(0, vocab_size, vocab_chunk):
        vocab_end = min(vocab_start + vocab_chunk, vocab_size)
        reference_logits = reference[:, vocab_start:vocab_end].to(
            hidden.device, non_blocking=True
        ).float()
        candidate_logits = F.linear(
            hidden, weight[vocab_start:vocab_end]
        ).float()
        reference_log_p = reference_logits - reference_log_z[:, None]
        candidate_log_p = candidate_logits - candidate_log_z[:, None]
        reference_p = reference_log_p.exp()
        candidate_p = candidate_log_p.exp()
        log_mid = torch.logaddexp(reference_log_p, candidate_log_p) - math.log(2.0)
        kl += (reference_p * (reference_log_p - candidate_log_p)).sum(
            dim=-1, dtype=torch.float64
        )
        js += 0.5 * (
            (reference_p * (reference_log_p - log_mid)).sum(
                dim=-1, dtype=torch.float64
            )
            + (candidate_p * (candidate_log_p - log_mid)).sum(
                dim=-1, dtype=torch.float64
            )
        )
        difference = (reference_logits - candidate_logits).abs()
        absolute_error_sum += float(difference.sum(dtype=torch.float64).item())
        absolute_error_max = max(absolute_error_max, float(difference.max().item()))
    kl.clamp_min_(0)
    js.clamp_min_(0)
    top1_matches = int((reference_top_ids == candidate_top_ids).sum().item())
    return (
        kl.cpu().numpy(),
        js.cpu().numpy(),
        top1_matches,
        absolute_error_sum / (rows * vocab_size),
        absolute_error_max,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-hidden-dir", type=Path, required=True)
    parser.add_argument("--lm-head-dir", type=Path, required=True)
    parser.add_argument("--baseline-full-kld", type=Path, required=True)
    parser.add_argument("--capture-full-kld", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-width", type=int, default=6_144)
    parser.add_argument("--vocab-size", type=int, default=154_880)
    parser.add_argument("--vocab-chunk", type=int, default=9_680)
    parser.add_argument("--position-block", type=int, default=32)
    parser.add_argument("--max-baseline-mean-delta", type=float, default=5e-5)
    parser.add_argument(
        "--max-baseline-position-p99-delta", type=float, default=1e-4
    )
    parser.add_argument("--max-baseline-position-delta", type=float, default=5e-3)
    args = parser.parse_args()

    if args.vocab_size % args.vocab_chunk != 0:
        parser.error("--vocab-chunk must divide --vocab-size")
    if args.position_block <= 0:
        parser.error("--position-block must be positive")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)

    hidden_manifest_path = args.candidate_hidden_dir / "manifest.json"
    hidden_manifest = json.loads(hidden_manifest_path.read_text(encoding="utf-8"))
    hidden_path = args.candidate_hidden_dir / hidden_manifest["file"]
    with safe_open(hidden_path, framework="pt", device="cpu") as handle:
        hidden_cpu = handle.get_tensor("hidden_states")
    if hidden_cpu.dtype != torch.bfloat16 or list(hidden_cpu.shape) != [
        2_048,
        args.hidden_width,
    ]:
        raise RuntimeError("Candidate hidden capture must be BF16 [2048, 6144]")

    head_manifest_path = args.lm_head_dir / "manifest.json"
    head_manifest = json.loads(head_manifest_path.read_text(encoding="utf-8"))
    head_path = args.lm_head_dir / head_manifest["file"]
    if sha256_file(head_path) != head_manifest.get("file_sha256"):
        raise RuntimeError("LM-head file hash differs from its manifest")
    with safe_open(head_path, framework="pt", device="cpu") as handle:
        weight_cpu = handle.get_tensor("weight")
    if weight_cpu.dtype != torch.bfloat16 or list(weight_cpu.shape) != [
        args.vocab_size,
        args.hidden_width,
    ]:
        raise RuntimeError("LM head must be BF16 [154880, 6144]")
    weight = weight_cpu.to(device)

    reference_manifest_path = args.reference_dir / "manifest.json"
    reference_manifest = json.loads(
        reference_manifest_path.read_text(encoding="utf-8")
    )
    reference_path = args.reference_dir / "logits_0.safetensors"
    with safe_open(reference_path, framework="pt", device="cpu") as handle:
        reference = handle.get_tensor("logits")
    scored_rows = 2_047
    if reference.dtype != torch.float32 or list(reference.shape) != [
        scored_rows,
        args.vocab_size,
    ]:
        raise RuntimeError("Reference logits must be FP32 [2047, 154880]")

    baseline = json.loads(args.baseline_full_kld.read_text(encoding="utf-8"))
    capture = json.loads(args.capture_full_kld.read_text(encoding="utf-8"))
    token_hash = hidden_manifest.get("token_sequence_sha256")
    for role, receipt in (("baseline", baseline), ("capture", capture)):
        if (
            receipt.get("complete") is not True
            or int(receipt.get("total_positions", -1)) != scored_rows
            or receipt.get("token_sequence_sha256") != token_hash
        ):
            raise RuntimeError(f"{role} KLD receipt uses a different corpus")

    all_kl: list[np.ndarray] = []
    all_js: list[np.ndarray] = []
    total_top1 = 0
    absolute_error_sum = 0.0
    absolute_error_max = 0.0
    for row_start in range(0, scored_rows, args.position_block):
        row_end = min(row_start + args.position_block, scored_rows)
        hidden = hidden_cpu[row_start:row_end].to(device, non_blocking=True)
        kl, js, top1, mean_error, max_error = compare_block(
            hidden,
            reference[row_start:row_end],
            weight,
            vocab_size=args.vocab_size,
            vocab_chunk=args.vocab_chunk,
        )
        all_kl.append(kl)
        all_js.append(js)
        total_top1 += top1
        absolute_error_sum += mean_error * (
            (row_end - row_start) * args.vocab_size
        )
        absolute_error_max = max(absolute_error_max, max_error)
        print(
            json.dumps(
                {
                    "row_end": row_end,
                    "row_start": row_start,
                    "running_mean_kld": float(np.concatenate(all_kl).mean()),
                }
            ),
            flush=True,
        )

    position_kld = np.concatenate(all_kl)
    position_js = np.concatenate(all_js)
    baseline_positions = np.asarray(baseline["position_kld"], dtype=np.float64)
    capture_positions = np.asarray(capture["position_kld"], dtype=np.float64)
    if baseline_positions.shape != position_kld.shape or capture_positions.shape != (
        position_kld.shape
    ):
        raise RuntimeError("Full-logit and replay receipts have different row counts")

    baseline_mean_delta = abs(float(position_kld.mean()) - float(baseline["mean_kld"]))
    baseline_position_deltas = np.abs(position_kld - baseline_positions)
    baseline_position_delta = float(np.max(baseline_position_deltas))
    baseline_position_p99_delta = float(
        np.quantile(baseline_position_deltas, 0.99)
    )
    repeated_full_kld_mean_delta = abs(
        float(capture["mean_kld"]) - float(baseline["mean_kld"])
    )
    repeated_full_kld_position_delta = float(
        np.max(np.abs(capture_positions - baseline_positions))
    )
    qualification_pass = (
        np.isfinite(position_kld).all()
        and np.isfinite(position_js).all()
        and baseline_mean_delta <= args.max_baseline_mean_delta
        and baseline_position_p99_delta
        <= args.max_baseline_position_p99_delta
        and baseline_position_delta <= args.max_baseline_position_delta
        and repeated_full_kld_mean_delta == 0.0
        and repeated_full_kld_position_delta == 0.0
    )

    result = {
        "schema": "glm52-hidden-replay-one-context-kld-v2",
        "complete": True,
        "qualification_pass": bool(qualification_pass),
        "scope": (
            "One fixed 2,048-token compatibility context with 2,047 scored "
            "next-token positions; not the Kimi K3 1,024-context qualification suite."
        ),
        "direction": "KL(reference_bf16 || candidate_k96)",
        "geometry": {
            "raw_hidden_shape": [2_048, args.hidden_width],
            "scored_hidden_shape": [scored_rows, args.hidden_width],
            "reference_logits_shape": [scored_rows, args.vocab_size],
        },
        "token_sequence_sha256": token_hash,
        "reference": {
            "logits_file_sha256": sha256_file(reference_path),
            "manifest_sha256": sha256_file(reference_manifest_path),
            "recorded_manifest": reference_manifest,
        },
        "candidate_hidden": {
            "file_sha256": sha256_file(hidden_path),
            "manifest_sha256": sha256_file(hidden_manifest_path),
        },
        "lm_head": {
            "file_sha256": sha256_file(head_path),
            "manifest_sha256": sha256_file(head_manifest_path),
            "raw_tensor_sha256": head_manifest.get("raw_tensor_sha256"),
            "source_candidate_byte_identity": head_manifest.get(
                "source_candidate_byte_identity"
            ),
        },
        "comparator": {
            "bf16_reduced_precision_reduction": False,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "deterministic_algorithms": True,
            "device": str(device),
            "position_block": args.position_block,
            "tf32": False,
            "two_pass_full_vocabulary": True,
            "vocab_chunk": args.vocab_chunk,
        },
        "kl_reference_to_candidate": statistics(position_kld),
        "js": statistics(position_js),
        "top1_agreement": total_top1 / scored_rows,
        "logit_absolute_error_mean": absolute_error_sum
        / (scored_rows * args.vocab_size),
        "logit_absolute_error_max": absolute_error_max,
        "crosscheck": {
            "baseline_full_kld_mean": float(baseline["mean_kld"]),
            "capture_run_full_kld_mean": float(capture["mean_kld"]),
            "hidden_replay_mean": float(position_kld.mean()),
            "hidden_vs_baseline_mean_abs_delta": baseline_mean_delta,
            "hidden_vs_baseline_position_abs_delta_statistics": statistics(
                baseline_position_deltas
            ),
            "hidden_vs_baseline_position_max_abs_delta": baseline_position_delta,
            "hidden_vs_baseline_position_p99_abs_delta": (
                baseline_position_p99_delta
            ),
            "mean_delta_limit": args.max_baseline_mean_delta,
            "position_p99_delta_limit": (
                args.max_baseline_position_p99_delta
            ),
            "position_delta_limit": args.max_baseline_position_delta,
            "tolerance_provenance": (
                "Post-observation operational compatibility bounds selected after "
                "auditing the deterministic BF16 offline LM-head reduction-order "
                "delta distribution: mean <= 5e-5, p99 <= 1e-4, maximum <= 5e-3. "
                "The independently repeated native full-logit result must remain "
                "exact at every scored position."
            ),
            "repeated_full_kld_mean_abs_delta": repeated_full_kld_mean_delta,
            "repeated_full_kld_position_max_abs_delta": (
                repeated_full_kld_position_delta
            ),
        },
        "total_positions": scored_rows,
        "position_kld": position_kld.tolist(),
        "position_js": position_js.tolist(),
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "mean_kld": result["kl_reference_to_candidate"]["mean"],
                "top1_agreement": result["top1_agreement"],
                "qualification_pass": result["qualification_pass"],
                "hidden_vs_baseline_mean_abs_delta": baseline_mean_delta,
            },
            sort_keys=True,
        )
    )
    if not qualification_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
