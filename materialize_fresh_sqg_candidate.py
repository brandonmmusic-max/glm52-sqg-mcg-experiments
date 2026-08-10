#!/usr/bin/env python3
"""Build or validate the sealed four-layer SQG candidate without launching it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.fresh_candidate_materializer import (
    materialize_candidate,
    validate_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--materialize", action="store_true")
    modes.add_argument("--validate", action="store_true")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--teacher-receipt", type=Path, required=True)
    parser.add_argument("--run-seal", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--bit-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-unchanged-hashes",
        action="store_true",
        help="validation only; selected SQG payloads are always rehashed",
    )
    args = parser.parse_args()
    common = {
        "source_model": args.source_model,
        "teacher_receipt": args.teacher_receipt,
        "run_seal": args.run_seal,
        "artifacts_root": args.artifacts_root,
        "bit_contract": args.bit_contract,
    }
    if args.materialize:
        if args.skip_unchanged_hashes:
            parser.error("--skip-unchanged-hashes is validation-only")
        result = materialize_candidate(output=args.output, **common)
    else:
        result = validate_candidate(
            args.output,
            verify_all_hashes=not args.skip_unchanged_hashes,
            **common,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
