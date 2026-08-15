#!/usr/bin/env python3
"""Prefetch each sealed remote K96 wave without racing the final merger."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from huggingface_hub import HfApi


REPO = "brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail"
WAVES = ((51, 54), (55, 58), (59, 62), (63, 66), (67, 70), (71, 74), (75, 77))
DEFAULT_PREFETCH_ROOT = Path(
    "/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-hub-prefetch"
)
DEFAULT_MERGE_ROOT = Path(
    "/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-hub-merge"
)
DEFAULT_STATE_ROOT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625/"
    "RESULTS/k96tail-distributed-merge-state"
)
HF = Path("/home/brandonmusic/.local/bin/hf")


def required_files(start: int, end: int) -> set[str]:
    required: set[str] = set()
    for layer in range(start, end + 1):
        padded = f"{layer:03d}"
        required.update(
            {
                f"r7-experts-layer-{padded}.safetensors",
                f"r7-experts-layer-{padded}.json",
                f"r7-experts-layer-{padded}.quality.json",
                f"runtime-oracle-layer-{padded}.json",
            }
        )
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
    return required


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def publish_complete_file(source: Path, destination: Path) -> None:
    if destination.is_file() and not destination.is_symlink():
        return
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"unsafe merge-stage target already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        if not destination.is_file() or destination.is_symlink():
            raise


def fetch_wave(
    api: HfApi,
    available: set[str],
    revision: str,
    start: int,
    end: int,
    prefetch_root: Path,
    merge_root: Path,
    state_root: Path,
) -> bool:
    wave = f"{start:03d}-{end:03d}"
    marker = state_root / f"hub-prefetch-wave-{wave}.complete.json"
    if marker.is_file() and not marker.is_symlink():
        return True
    required = required_files(start, end)
    missing = sorted(required - available)
    if missing:
        print(
            f"K96 Hub prefetch: wave={wave} ready=false missing={len(missing)}",
            flush=True,
        )
        return False

    reproduction_root = f"reproduction/wave-{wave}/"
    files = sorted(required | {name for name in available if name.startswith(reproduction_root)})
    destination = prefetch_root / f"wave-{wave}"
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        str(HF),
        "download",
        REPO,
        "--revision",
        revision,
        "--local-dir",
        str(destination),
        "--max-workers",
        "8",
    ]
    for name in files:
        command.extend(("--include", name))
    print(
        f"K96 Hub prefetch: wave={wave} ready=true files={len(files)} revision={revision}",
        flush=True,
    )
    subprocess.run(command, check=True)
    for name in files:
        source = destination / name
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"prefetched file is absent or unsafe: {source}")

    reproduction = destination / f"reproduction/wave-{wave}"
    subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=reproduction,
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    for name in files:
        publish_complete_file(destination / name, merge_root / name)
    atomic_json(
        marker,
        {
            "schema": "glm52-k96tail-hub-wave-prefetch-v1",
            "complete": True,
            "repo": REPO,
            "revision": revision,
            "wave": [start, end],
            "files": files,
        },
    )
    print(f"K96 Hub prefetch: wave={wave} published=true", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefetch-root", type=Path, default=DEFAULT_PREFETCH_ROOT)
    parser.add_argument("--merge-root", type=Path, default=DEFAULT_MERGE_ROOT)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    args.prefetch_root.mkdir(parents=True, exist_ok=True)
    args.merge_root.mkdir(parents=True, exist_ok=True)
    args.state_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.state_root / "hub-wave-prefetch.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("K96 Hub prefetch: another watcher owns the lock", file=sys.stderr)
            return 2
        api = HfApi()
        while True:
            try:
                info = api.model_info(REPO, files_metadata=False)
                revision = str(info.sha)
                available = {item.rfilename for item in info.siblings}
                complete = [
                    fetch_wave(
                        api,
                        available,
                        revision,
                        start,
                        end,
                        args.prefetch_root,
                        args.merge_root,
                        args.state_root,
                    )
                    for start, end in WAVES
                ]
                if all(complete):
                    print("K96 Hub prefetch: all remote waves published locally", flush=True)
                    return 0
            except Exception as exc:  # keep the persistent watcher fail-soft on Hub errors
                print(f"K96 Hub prefetch: transient error: {exc}", file=sys.stderr, flush=True)
                if args.once:
                    return 1
            if args.once:
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
