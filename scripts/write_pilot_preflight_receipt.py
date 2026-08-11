#!/usr/bin/env python3
"""Write the minimal closed receipt for a freshly built follow-up preflight."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fresh_pipeline_common import atomic_json, canonical_sha256, load_json_object, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    path = args.preflight.resolve()
    preflight = load_json_object(path)
    claimed = preflight.get("preflight_id")
    unhashed = dict(preflight)
    unhashed.pop("preflight_id", None)
    if claimed != canonical_sha256(unhashed):
        raise ValueError("preflight ID does not close")
    receipt = {
        "schema": "glm52-fresh-sqg-contiguous-pilot-preflight-receipt-v1",
        "complete": True,
        "successor_preflight_id": claimed,
        "preflight_sha256": sha256_file(path),
        "reuse": {
            "bf16_or_capture_payload_rehash_on_timed_encode_path": False,
            "fresh_preparation_required": True,
        },
    }
    receipt["receipt_id"] = canonical_sha256(receipt)
    atomic_json(path.parent / "successor_preflight_receipt.json", receipt)


if __name__ == "__main__":
    main()
