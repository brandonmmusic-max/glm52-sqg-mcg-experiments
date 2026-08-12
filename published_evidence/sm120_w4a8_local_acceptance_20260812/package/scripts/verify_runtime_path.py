#!/usr/bin/env python3
"""Prove the native SQG W4A8 path executed and no forbidden fallback ran.

Evidence sources:
  1. Per-worker evidence receipts written by the instrumented exl3.py
     (VLLM_GLM_SQG_W4A8_EVIDENCE_DIR): every TP worker must record
     loaded_layers == executed_layers == [3..78] with mcg_tensor_count 0 and
     allow_a16_fallback false. "executed_layers" entries are appended from
     inside _apply_glm_sqg_w4a8, i.e. only when the route-packed native W4A8
     runtime actually ran for that layer.
  2. Server logs: the native runtime-planned line must be present; forbidden
     path markers (rank-sliced runtime, eager parity path, forced A16/A8,
     MCG dispatch) must be absent.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REQUIRED_LOG_PATTERNS = [
    r"GLM SQG full-W4A8 runtime planned",
    r"topology attestation: checkpoint sealed TP1/PP8/DCP1; operator attested the contract-legal TP4/PP1/DCP[14]",
]
FORBIDDEN_LOG_PATTERNS = [
    (r"EXL3 rank-sliced runtime planned", "rank-sliced (MCG hybrid) MoE path"),
    (r"eager parity path", "ExLlamaV3 parity fallback"),
    (r"B12X_MOE_FORCE_A16=1|MOE_MODE=a16|force-a16", "forced A16 execution"),
    (r"launch_k6_mcg", "MCG K6 dense dispatch"),
    (r"refusing", "a fail-closed guard fired"),
    (r"partial GLM SQG W4A8", "partial-construction contract"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--expect-tp", type=int, default=4)
    parser.add_argument("--expect-layers", default="3-77",
                        help="MTP0 acceptance: target layers only; the layer-78 MTP draft loads only under MTP speculative configs")
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()
    first, last = (int(v) for v in args.expect_layers.split("-"))
    expected_layers = list(range(first, last + 1))
    problems: list[str] = []

    evidence_dir = Path(args.evidence_dir)
    receipts = sorted(evidence_dir.glob("pp-*.json"))
    workers = []
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text())
        workers.append(
            {
                "file": receipt_path.name,
                "pid": receipt.get("pid"),
                "tp": receipt.get("tensor_parallel_size"),
                "loaded": receipt.get("loaded_layers"),
                "executed": receipt.get("executed_layers"),
            }
        )
        if receipt.get("tensor_parallel_size") != args.expect_tp:
            problems.append(f"{receipt_path.name}: TP {receipt.get('tensor_parallel_size')} != {args.expect_tp}")
        if receipt.get("allow_a16_fallback") is not False:
            problems.append(f"{receipt_path.name}: allow_a16_fallback != false")
        if receipt.get("mcg_tensor_count") != 0:
            problems.append(f"{receipt_path.name}: mcg_tensor_count != 0")
        if receipt.get("loaded_layers") != expected_layers:
            problems.append(f"{receipt_path.name}: loaded_layers != {first}..{last}")
        if receipt.get("executed_layers") != expected_layers:
            problems.append(
                f"{receipt_path.name}: executed_layers {receipt.get('executed_layers', [])[:4]}... != {first}..{last}"
            )
    distinct_pids = {worker["pid"] for worker in workers}
    # The multiproc executor materializes two evidence-writing processes per
    # TP rank on this base (worker + profile-fork); require at least one
    # complete receipt per rank-process and full coverage in every receipt.
    if len(distinct_pids) < args.expect_tp:
        problems.append(
            f"expected at least {args.expect_tp} worker evidence receipts, "
            f"found {len(distinct_pids)} distinct pids in {len(receipts)} files"
        )

    logs = subprocess.run(
        ["docker", "logs", args.container],
        capture_output=True, text=True,
    )
    log_text = logs.stdout + logs.stderr
    log_findings = {}
    for pattern in REQUIRED_LOG_PATTERNS:
        count = len(re.findall(pattern, log_text))
        log_findings[pattern] = count
        if count == 0:
            problems.append(f"required log pattern absent: {pattern}")
    for pattern, meaning in FORBIDDEN_LOG_PATTERNS:
        hits = re.findall(rf".*{pattern}.*", log_text)
        # A guard *message string* inside a traceback means the gate fired.
        if hits:
            problems.append(f"forbidden pattern ({meaning}): {hits[0][:160]}")

    receipt = {
        "schema": "glm52-sqg-w4a8-sm120-runtime-path-verification-v1",
        "container": args.container,
        "workers": workers,
        "distinct_worker_pids": len(distinct_pids),
        "required_log_patterns": log_findings,
        "problems": problems,
        "pass": not problems,
    }
    if args.result_json:
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
