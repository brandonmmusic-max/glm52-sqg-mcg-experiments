"""Opt-in configuration for exact four-layer follow-up pilots.

The original experiment remains the default. A follow-up run may select a
different exact four-layer set by exporting ``FRESH_SQG_SELECTED_LAYERS``.
Keeping the switch here prevents capture, source, encoder, and materializer
from silently disagreeing about the treatment set.
"""

from __future__ import annotations

import os


DEFAULT_SELECTED_LAYERS = (6, 28, 52, 77)


def selected_layers() -> tuple[int, int, int, int]:
    raw = os.environ.get("FRESH_SQG_SELECTED_LAYERS")
    if raw is None:
        return DEFAULT_SELECTED_LAYERS
    try:
        values = tuple(int(token.strip()) for token in raw.split(","))
    except ValueError as error:
        raise ValueError(
            "FRESH_SQG_SELECTED_LAYERS must be comma-separated integers"
        ) from error
    if (
        len(values) != 4
        or len(set(values)) != 4
        or values != tuple(sorted(values))
        or any(layer < 3 or layer > 77 for layer in values)
    ):
        raise ValueError(
            "FRESH_SQG_SELECTED_LAYERS must contain four unique ascending routed "
            "layers in [3,77]"
        )
    return values  # type: ignore[return-value]


def validated_sha256_env(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value
