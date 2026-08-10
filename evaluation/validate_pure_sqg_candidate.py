#!/usr/bin/env python3
"""Fail closed unless a candidate matches the sealed four-layer SQG run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.fresh_candidate_materializer import validate_candidate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--teacher-receipt", type=Path, required=True)
    parser.add_argument("--run-seal", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--bit-contract", type=Path, required=True)
    parser.add_argument(
        "--skip-unchanged-hashes",
        action="store_true",
        help="still rehashes every selected SQG tensor payload",
    )
    args = parser.parse_args()
    report = validate_candidate(
        args.candidate,
        source_model=args.source,
        teacher_receipt=args.teacher_receipt,
        run_seal=args.run_seal,
        artifacts_root=args.artifacts_root,
        bit_contract=args.bit_contract,
        verify_all_hashes=not args.skip_unchanged_hashes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
