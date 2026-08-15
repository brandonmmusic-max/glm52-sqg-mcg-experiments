#!/usr/bin/env python3
from __future__ import annotations

import sys

from huggingface_hub import list_repo_files


REPO = "brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail"
WAVES = ((51, 54), (55, 58), (59, 62), (63, 66), (67, 70), (71, 74), (75, 77))


def main() -> int:
    available = set(list_repo_files(REPO, repo_type="model"))
    required: set[str] = set()
    for layer in range(51, 78):
        padded = f"{layer:03d}"
        required.update(
            {
                f"r7-experts-layer-{padded}.safetensors",
                f"r7-experts-layer-{padded}.json",
                f"r7-experts-layer-{padded}.quality.json",
                f"runtime-oracle-layer-{padded}.json",
            }
        )
    for start, end in WAVES:
        root = f"reproduction/wave-{start:03d}-{end:03d}"
        required.update(
            {
                f"{root}/SHA256SUMS",
                f"{root}/campaign.log",
                f"{root}/recipe-and-profiles.tgz",
                f"{root}/tail-scores.tgz",
                f"{root}/allocations.tgz",
                f"{root}/parity-proofs.tgz",
            }
        )
    missing = sorted(required - available)
    print(f"K96 Hub readiness: present={len(required) - len(missing)}/{len(required)} missing={len(missing)}")
    for path in missing[:12]:
        print(f"missing: {path}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
