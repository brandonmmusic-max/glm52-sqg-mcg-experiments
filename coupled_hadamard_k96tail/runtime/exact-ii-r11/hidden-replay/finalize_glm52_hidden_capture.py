#!/usr/bin/env python3
"""Seal one raw GLM-5.2 prompt capture as a 2,048-row BF16 tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


CHUNK_RE = re.compile(r"hidden\.rows-(\d+)-(\d+)\.safetensors$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def token_sequence_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids).encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--full-kld-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=2_048)
    parser.add_argument("--expected-width", type=int, default=6_144)
    args = parser.parse_args()

    token_ids = json.loads(args.token_file.read_text(encoding="utf-8"))
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise TypeError("Token file must be a JSON integer array")
    if len(token_ids) != args.expected_rows:
        raise RuntimeError(
            f"Expected {args.expected_rows} tokens; got {len(token_ids)}"
        )
    token_sha256 = token_sequence_sha256(token_ids)

    full_kld = json.loads(args.full_kld_receipt.read_text(encoding="utf-8"))
    if (
        full_kld.get("complete") is not True
        or int(full_kld.get("total_positions", -1)) != args.expected_rows - 1
        or full_kld.get("token_sequence_sha256") != token_sha256
    ):
        raise RuntimeError("Full-logit KLD receipt does not match the token capture")

    chunks: list[tuple[int, int, Path]] = []
    for path in args.raw_dir.rglob("hidden.rows-*.safetensors"):
        match = CHUNK_RE.fullmatch(path.name)
        if match is not None:
            chunks.append((int(match.group(1)), int(match.group(2)), path))
    chunks.sort()
    if not chunks:
        raise RuntimeError(f"No hidden-state chunks found in {args.raw_dir}")

    next_row = 0
    request_ids: set[str] = set()
    tensors: list[torch.Tensor] = []
    chunk_records: list[dict[str, Any]] = []
    for start, end, path in chunks:
        if start != next_row or end <= start:
            raise RuntimeError(
                f"Expected row {next_row}; found [{start}, {end}) in {path}"
            )
        with safe_open(path, framework="pt", device="cpu") as handle:
            if list(handle.keys()) != ["hidden_states"]:
                raise RuntimeError(f"Unexpected keys in {path}")
            tensor_slice = handle.get_slice("hidden_states")
            if tensor_slice.get_dtype() != "BF16":
                raise RuntimeError(f"Expected BF16 hidden states in {path}")
            if tensor_slice.get_shape() != [end - start, args.expected_width]:
                raise RuntimeError(f"Unexpected shape in {path}")
            metadata = handle.metadata() or {}
        if metadata.get("semantic_point") != (
            "after_final_rmsnorm_before_lm_head"
        ):
            raise RuntimeError(f"Semantic-point identity is absent from {path}")
        request_ids.add(metadata.get("request_id", ""))
        tensors.append(load_file(str(path), device="cpu")["hidden_states"])
        chunk_records.append(
            {
                "file": str(path.relative_to(args.raw_dir)),
                "row_end": end,
                "row_start": start,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
        next_row = end

    if next_row != args.expected_rows or len(request_ids) != 1:
        raise RuntimeError(
            f"Capture must contain one {args.expected_rows}-row request; "
            f"got rows={next_row}, request_ids={sorted(request_ids)}"
        )
    hidden_states = torch.cat(tensors, dim=0).contiguous()
    if hidden_states.dtype != torch.bfloat16 or list(hidden_states.shape) != [
        args.expected_rows,
        args.expected_width,
    ]:
        raise RuntimeError("Final hidden-state tensor has the wrong contract")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "hidden_states.safetensors"
    if not output_path.exists():
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        save_file(
            {"hidden_states": hidden_states},
            str(temporary),
            metadata={
                "raw_rows": str(args.expected_rows),
                "scored_rows": str(args.expected_rows - 1),
                "semantic_point": "after_final_rmsnorm_before_lm_head",
                "token_sequence_sha256": token_sha256,
            },
        )
        temporary.replace(output_path)

    with safe_open(output_path, framework="pt", device="cpu") as handle:
        stored = handle.get_slice("hidden_states")
        if stored.get_dtype() != "BF16" or stored.get_shape() != [
            args.expected_rows,
            args.expected_width,
        ]:
            raise RuntimeError("Existing finalized hidden tensor is incompatible")
        stored_metadata = handle.metadata() or {}
    if stored_metadata.get("token_sequence_sha256") != token_sha256:
        raise RuntimeError("Existing finalized hidden tensor uses other tokens")

    manifest = {
        "schema": "glm52-pre-lm-head-hidden-one-context-v1",
        "complete": True,
        "created_utc": datetime.now(UTC).isoformat(),
        "dtype": "BF16",
        "file": output_path.name,
        "file_sha256": sha256_file(output_path),
        "full_kld_receipt": str(args.full_kld_receipt.resolve()),
        "full_kld_receipt_sha256": sha256_file(args.full_kld_receipt),
        "hidden_width": args.expected_width,
        "interpretation": (
            "Raw prompt capture retains all 2,048 rows; rows 0..2,046 are "
            "scored against next-token reference logits and row 2,047 is retained "
            "but not scored because its continuation target is absent."
        ),
        "key": "hidden_states",
        "raw_chunks": chunk_records,
        "raw_rows": args.expected_rows,
        "request_id": next(iter(request_ids)),
        "scored_rows": args.expected_rows - 1,
        "semantic_point": "after_final_rmsnorm_before_lm_head",
        "shape": [args.expected_rows, args.expected_width],
        "size_bytes": output_path.stat().st_size,
        "token_file": str(args.token_file.resolve()),
        "token_file_sha256": sha256_file(args.token_file),
        "token_sequence_sha256": token_sha256,
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
