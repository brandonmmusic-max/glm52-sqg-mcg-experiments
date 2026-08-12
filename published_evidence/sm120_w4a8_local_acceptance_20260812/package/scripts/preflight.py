#!/usr/bin/env python3
"""Host preflight for the SM120 local serving package (no GPU allocation)."""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--port", type=int, default=9418)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    if len(rows) < 4:
        fail(f"need 4 GPUs, found {len(rows)}")
    for row in rows[:4]:
        index, name, capability, total = [c.strip() for c in row.split(",")]
        if capability != "12.0":
            fail(f"GPU {index} ({name}) compute capability {capability} != 12.0")
    print(f"GPUs OK: {len(rows)} visible, 0-3 are CC 12.0")

    manifest = json.loads((Path(args.reference) / "manifest.json").read_text())
    shape = manifest["windows"][0]["shape"]
    if shape != [2047, 154880]:
        fail(f"sealed reference shape {shape} != [2047, 154880]")
    if not (Path(args.reference) / "logits_0.safetensors").is_file():
        fail("logits_0.safetensors missing from reference dir")
    print(f"Sealed reference OK: {shape}, context {manifest['context_length']}")

    model = Path(args.model)
    for required in ("config.json", "quantization_config.json",
                     "model.safetensors.index.json"):
        if not (model / required).is_file():
            fail(f"model file missing: {required} (download still running?)")
    complete_marker = model / ".download-complete.json"
    print(f"Model dir OK (complete-marker present: {complete_marker.is_file()})")

    free = shutil.disk_usage(model).free / 1e9
    if free < 20:
        fail(f"only {free:.0f} GB free on the model filesystem")
    print(f"Disk OK: {free:.0f} GB free")

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    in_use = probe.connect_ex(("127.0.0.1", args.port)) == 0
    probe.close()
    print(f"Port {args.port}: {'IN USE (must belong to this project)' if in_use else 'free'}")

    have_image = subprocess.run(
        ["docker", "image", "inspect", args.image],
        capture_output=True,
    ).returncode == 0
    print(f"Image {args.image}: {'present' if have_image else 'NOT BUILT YET'}")
    print("preflight: PASS")


if __name__ == "__main__":
    main()
