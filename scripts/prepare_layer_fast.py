#!/usr/bin/env python3
"""Prepare one layer from a sealed preflight without payload rehash."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from scripts.encode_final_shard import _open_fast_sealed_runtime
    from src.fresh_pipeline_runner import prepare_layer

    runtime = _open_fast_sealed_runtime(
        args.preflight, layer=args.layer, device=args.device
    )
    print(json.dumps(prepare_layer(runtime), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
