#!/usr/bin/env python3
"""Export and attest the unchanged BF16 GLM-5.2 LM-head tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    contiguous = tensor.contiguous()
    byte_view = contiguous.view(torch.uint8).numpy()
    data = memoryview(byte_view).cast("B")
    digest = hashlib.sha256()
    block_size = 16 * 1024 * 1024
    for offset in range(0, data.nbytes, block_size):
        digest.update(data[offset : offset + block_size])
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_checkpoint_tensor(
    model_dir: Path,
    tensor_key: str,
) -> tuple[torch.Tensor, Path, Path]:
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    try:
        shard_name = index["weight_map"][tensor_key]
    except KeyError as error:
        raise RuntimeError(f"{tensor_key} is absent from {index_path}") from error
    shard_path = model_dir / shard_name
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(tensor_key)
    return tensor, index_path, shard_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-key", default="lm_head.weight")
    parser.add_argument("--expected-vocab", type=int, default=154_880)
    parser.add_argument("--expected-hidden-width", type=int, default=6_144)
    args = parser.parse_args()

    expected_shape = [args.expected_vocab, args.expected_hidden_width]
    source, source_index, source_shard = load_checkpoint_tensor(
        args.source_model, args.tensor_key
    )
    candidate, candidate_index, candidate_shard = load_checkpoint_tensor(
        args.candidate_model, args.tensor_key
    )
    for role, tensor in (("source", source), ("candidate", candidate)):
        if tensor.dtype != torch.bfloat16 or list(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"{role} LM head must be BF16 {expected_shape}; "
                f"got {tensor.dtype} {list(tensor.shape)}"
            )

    source_raw_sha256 = sha256_tensor(source)
    candidate_raw_sha256 = sha256_tensor(candidate)
    if source_raw_sha256 != candidate_raw_sha256:
        raise RuntimeError(
            "Candidate checkpoint changed the BF16 LM head: "
            f"{source_raw_sha256} != {candidate_raw_sha256}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "weight.safetensors"
    manifest_path = args.output_dir / "manifest.json"
    if not output_path.exists():
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        save_file(
            {"weight": source.contiguous()},
            str(temporary),
            metadata={
                "checkpoint_tensor_key": args.tensor_key,
                "raw_tensor_sha256": source_raw_sha256,
            },
        )
        temporary.replace(output_path)

    with safe_open(output_path, framework="pt", device="cpu") as handle:
        exported = handle.get_tensor("weight")
    if sha256_tensor(exported) != source_raw_sha256:
        raise RuntimeError("Exported LM-head payload differs from the checkpoint")

    manifest = {
        "schema": "glm52-canonical-bf16-lm-head-v1",
        "complete": True,
        "candidate_checkpoint": str(args.candidate_model.resolve()),
        "candidate_index_sha256": sha256_file(candidate_index),
        "candidate_shard": candidate_shard.name,
        "candidate_shard_sha256": sha256_file(candidate_shard),
        "checkpoint_tensor_key": args.tensor_key,
        "dtype": "BF16",
        "file": output_path.name,
        "file_sha256": sha256_file(output_path),
        "key": "weight",
        "raw_tensor_sha256": source_raw_sha256,
        "shape": expected_shape,
        "size_bytes": output_path.stat().st_size,
        "source_checkpoint": str(args.source_model.resolve()),
        "source_index_sha256": sha256_file(source_index),
        "source_shard": source_shard.name,
        "source_shard_sha256": sha256_file(source_shard),
        "source_candidate_byte_identity": True,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
