from __future__ import annotations

from pathlib import Path

import torch

from bmmlaw_r7_encoder.safetensors_io import (
    SafeTensorReader,
    read_torch_tensor,
    torch_tensor_entry,
    write_safetensors_atomic,
)
from glm52_fresh_sqg.manifest import SQG_MARKER


def test_scalar_sqg_marker_round_trips_without_changing_shape(tmp_path: Path) -> None:
    marker = torch.tensor(SQG_MARKER, dtype=torch.int32)
    entry = torch_tensor_entry("model.test.sqg", marker)

    assert entry.shape == ()
    assert entry.nbytes == 4
    assert bytes(entry.payload) == SQG_MARKER.to_bytes(4, "little", signed=True)

    shard = tmp_path / "scalar.safetensors"
    write_safetensors_atomic(shard, [entry], metadata={"format": "pt"})
    reader = SafeTensorReader(shard)
    restored = read_torch_tensor(reader, "model.test.sqg")

    assert restored.shape == torch.Size([])
    assert restored.dtype == torch.int32
    assert restored.item() == SQG_MARKER


def test_scalar_byte_extraction_supports_every_writer_dtype() -> None:
    values = {
        torch.bool: True,
        torch.uint8: 7,
        torch.int8: -7,
        torch.int16: -700,
        torch.float16: 1.5,
        torch.bfloat16: 1.5,
        torch.int32: SQG_MARKER,
        torch.float32: 1.5,
        torch.int64: -70000,
        torch.float64: 1.5,
    }
    for dtype, value in values.items():
        tensor = torch.tensor(value, dtype=dtype)
        entry = torch_tensor_entry(str(dtype), tensor)
        assert entry.shape == ()
        assert entry.nbytes == tensor.element_size()
