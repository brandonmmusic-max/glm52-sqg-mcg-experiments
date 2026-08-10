"""Independent CPU/PyTorch closure for uniform-rate SQG EXL tensors.

This module deliberately does not import the KQuant encoder backend.  It
unpacks the native EXL word order, reconstructs cyclic L16 trellis states,
looks up the frozen E4M3 labels, and applies the stored FP16 scale vectors.
That separation makes the post-encode closure meaningful: the encoder and
the verifier do not share packing or transform code.
"""

from __future__ import annotations

import hashlib
import math

import torch


TILE_CHANNELS = 16
TILE_VALUES = TILE_CHANNELS * TILE_CHANNELS
HADAMARD_BLOCK = 128
SUPPORTED_BITS = (3, 4)
PACK_CHUNK_TILES = 4096


def _validate_bits(bits: int) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in SUPPORTED_BITS:
        raise ValueError("fresh GLM SQG supports only uniform integer K3 or K4")


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a tensor's dtype, shape, and exact contiguous payload bytes."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor_sha256 expects a torch.Tensor")
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in value.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(value.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def payload_sha256(tensor: torch.Tensor) -> str:
    """Hash only a tensor's exact contiguous payload bytes."""

    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(memoryview(value.view(torch.uint8).numpy()).cast("B")).hexdigest()


def tensor_core_permutation(device: torch.device | str = "cpu") -> torch.Tensor:
    """Return EXL's row-major 16x16 to tensor-core element permutation."""

    permutation = [0] * TILE_VALUES
    for thread in range(32):
        row0 = (thread % 4) * 2
        row1 = row0 + 1
        row2 = row0 + 8
        row3 = row0 + 9
        column0 = thread // 4
        column1 = column0 + 8
        permutation[thread * 8 + 0] = row0 * 16 + column0
        permutation[thread * 8 + 1] = row1 * 16 + column0
        permutation[thread * 8 + 2] = row2 * 16 + column0
        permutation[thread * 8 + 3] = row3 * 16 + column0
        permutation[thread * 8 + 4] = row0 * 16 + column1
        permutation[thread * 8 + 5] = row1 * 16 + column1
        permutation[thread * 8 + 6] = row2 * 16 + column1
        permutation[thread * 8 + 7] = row3 * 16 + column1
    return torch.tensor(permutation, dtype=torch.long, device=device)


def pack_trellis_states(states: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack the low K edge bits of legal cyclic states in native EXL order."""

    _validate_bits(bits)
    if states.ndim != 3 or states.shape[-1] != TILE_VALUES:
        raise ValueError("states must have shape [K/16, N/16, 256]")
    if states.dtype != torch.int16:
        raise TypeError("states must use torch.int16")
    tiles = states.reshape(-1, TILE_VALUES)
    packed = torch.empty(
        (tiles.shape[0], TILE_CHANNELS * bits),
        dtype=torch.int16,
        device=states.device,
    )
    symbol_shifts = torch.arange(bits - 1, -1, -1, device=states.device)
    word_shifts = torch.arange(15, -1, -1, device=states.device)
    for begin in range(0, tiles.shape[0], PACK_CHUNK_TILES):
        values = tiles[begin : begin + PACK_CHUNK_TILES].to(torch.int64)
        values &= (1 << bits) - 1
        spans = values.reshape(-1, TILE_CHANNELS, TILE_CHANNELS)
        bitstream = ((spans[..., None] >> symbol_shifts) & 1).reshape(
            -1, TILE_CHANNELS, bits * TILE_CHANNELS
        )
        words = (
            bitstream.reshape(-1, TILE_CHANNELS, bits, 16) << word_shifts
        ).sum(dim=-1)
        flat = words.reshape(-1, TILE_CHANNELS * bits)
        # Native pack.cu applies SWAP16: adjacent uint16 words exchange places.
        swapped = flat.reshape(flat.shape[0], -1, 2).flip(-1).reshape(flat.shape)
        packed[begin : begin + values.shape[0]].copy_(swapped.to(torch.int16))
    return packed.reshape(*states.shape[:-1], TILE_CHANNELS * bits).contiguous()


def unpack_trellis_edges(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Invert native EXL word packing to the stored K-bit edge symbols."""

    _validate_bits(bits)
    if packed.ndim != 3 or packed.shape[-1] != TILE_CHANNELS * bits:
        raise ValueError(
            f"packed trellis must have shape [K/16, N/16, {TILE_CHANNELS * bits}]"
        )
    if packed.dtype != torch.int16:
        raise TypeError("packed trellis must use torch.int16")
    packed_tiles = packed.reshape(-1, TILE_CHANNELS * bits)
    edges = torch.empty(
        (packed_tiles.shape[0], TILE_VALUES),
        dtype=torch.int16,
        device=packed.device,
    )
    word_shifts = torch.arange(15, -1, -1, device=packed.device)
    symbol_shifts = torch.arange(bits - 1, -1, -1, device=packed.device)
    for begin in range(0, packed_tiles.shape[0], PACK_CHUNK_TILES):
        words = packed_tiles[begin : begin + PACK_CHUNK_TILES].to(torch.int64)
        words &= 0xFFFF
        words = words.reshape(words.shape[0], -1, 2).flip(-1).reshape(words.shape)
        words = words.reshape(-1, TILE_CHANNELS, bits)
        bitstream = ((words[..., None] >> word_shifts) & 1).reshape(
            -1, TILE_CHANNELS, bits * TILE_CHANNELS
        )
        symbol_bits = bitstream.reshape(
            -1, TILE_CHANNELS, TILE_CHANNELS, bits
        )
        values = (symbol_bits << symbol_shifts).sum(dim=-1)
        edges[begin : begin + words.shape[0]].copy_(
            values.reshape(-1, TILE_VALUES).to(torch.int16)
        )
    return edges.reshape(*packed.shape[:-1], TILE_VALUES).contiguous()


def reconstruct_trellis_states(edges: torch.Tensor, bits: int) -> torch.Tensor:
    """Reconstruct every cyclic 16-bit L16 state from its stored edges."""

    _validate_bits(bits)
    if edges.ndim != 3 or edges.shape[-1] != TILE_VALUES:
        raise ValueError("edges must have shape [K/16, N/16, 256]")
    if edges.dtype == torch.bool or edges.is_floating_point():
        raise TypeError("edges must use an integer dtype")
    values = edges.to(dtype=torch.int64) & ((1 << bits) - 1)
    states = torch.zeros_like(values)
    for lag in range(math.ceil(16 / bits)):
        states |= torch.roll(values, shifts=lag, dims=-1) << (lag * bits)
    return (states & 0xFFFF).to(dtype=torch.int16).contiguous()


def unpack_trellis_states(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack a uniform-rate EXL payload to full cyclic L16 states."""

    return reconstruct_trellis_states(unpack_trellis_edges(packed, bits), bits)


def normalized_hadamard(
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    size: int = HADAMARD_BLOCK,
) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError("Hadamard size must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix * (1.0 / math.sqrt(size))


def decode_regularized_states(
    states: torch.Tensor,
    codebook_e4m3: torch.Tensor,
) -> torch.Tensor:
    """Decode state-labelled tiles to the regularized EXL [K, N] matrix."""

    if states.ndim != 3 or states.shape[-1] != TILE_VALUES:
        raise ValueError("states must have shape [K/16, N/16, 256]")
    if states.dtype != torch.int16:
        raise TypeError("states must use torch.int16")
    if codebook_e4m3.dtype != torch.uint8 or tuple(codebook_e4m3.shape) != (1 << 16,):
        raise ValueError("SQG codebook must be exactly 65,536 raw uint8 E4M3 labels")
    lut = codebook_e4m3.to(device=states.device).view(torch.float8_e4m3fn).float()
    if not bool(torch.isfinite(lut).all()):
        raise ValueError("SQG codebook contains non-finite E4M3 labels")
    indices = (states.to(torch.int64) & 0xFFFF).long()
    decoded = lut.index_select(0, indices.flatten()).reshape_as(states)
    inverse = torch.argsort(tensor_core_permutation(states.device))
    decoded = decoded.index_select(-1, inverse)
    k_tiles, n_tiles, _ = decoded.shape
    return (
        decoded.reshape(k_tiles, n_tiles, TILE_CHANNELS, TILE_CHANNELS)
        .permute(0, 2, 1, 3)
        .reshape(k_tiles * TILE_CHANNELS, n_tiles * TILE_CHANNELS)
        .contiguous()
    )


def decode_stored_fp16(
    packed: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    bits: int,
    codebook_e4m3: torch.Tensor,
) -> torch.Tensor:
    """Decode only from packed words and the persisted FP16 vectors."""

    _validate_bits(bits)
    if suh.dtype != torch.float16 or svh.dtype != torch.float16:
        raise TypeError("stored SQG scale vectors must use torch.float16")
    states = unpack_trellis_states(packed, bits)
    weight = decode_regularized_states(states, codebook_e4m3)
    k, n = weight.shape
    if k % HADAMARD_BLOCK or n % HADAMARD_BLOCK:
        raise ValueError("EXL K and N dimensions must be divisible by 128")
    if suh.ndim != 1 or suh.numel() != k:
        raise ValueError("suh length does not match packed EXL K")
    if svh.ndim != 1 or svh.numel() != n:
        raise ValueError("svh length does not match packed EXL N")
    hadamard = normalized_hadamard(device=weight.device, dtype=weight.dtype)
    left_blocks = weight.reshape(-1, HADAMARD_BLOCK, n)
    weight = torch.matmul(hadamard, left_blocks).reshape(k, n)
    weight *= suh.to(device=weight.device, dtype=weight.dtype).unsqueeze(1)
    right_blocks = weight.reshape(k, -1, HADAMARD_BLOCK)
    weight = torch.matmul(right_blocks, hadamard).reshape(k, n)
    weight *= svh.to(device=weight.device, dtype=weight.dtype).unsqueeze(0)
    return weight.contiguous()


def relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return RMSE(actual-expected) divided by RMS(expected)."""

    if actual.shape != expected.shape:
        raise ValueError("closure tensors must have the same shape")
    difference = actual.float() - expected.float()
    denominator = expected.float().square().mean().sqrt().clamp_min(1e-20)
    return float(difference.square().mean().sqrt() / denominator)
