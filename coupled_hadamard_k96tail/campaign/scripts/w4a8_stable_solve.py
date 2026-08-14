#!/usr/bin/env python3
"""Deterministically stabilize a W4A8 cross-term solve toward official BF16.

The realized activation covariance can be rank deficient for low-support
experts, especially at beta 1 where the ordinary joint-shrinkage objective
contains no official-weight prior.  Adding jitter to H alone would select zero
in unconstrained directions.  This helper instead adds the smallest working
joint `(scale I, scale W_official)` prior, preserving the official BF16 weight
in directions unsupported by routed fit data.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch

from scripts.w4a8_cross_term import (
    CrossTermStatistics,
    shrink_cross_term_objective,
    solve_cross_term_target,
)


DEFAULT_PRIOR_FRACTIONS = tuple(2.0**exponent for exponent in range(-24, -7, 2))


def solve_with_minimal_official_prior(
    statistics: CrossTermStatistics,
    official_weight_exl: torch.Tensor,
    *,
    identity_scale: float | torch.Tensor,
    prior_fractions: Iterable[float] = DEFAULT_PRIOR_FRACTIONS,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve directly, or use the first deterministic joint prior that is PD."""

    if isinstance(identity_scale, torch.Tensor):
        if identity_scale.numel() != 1:
            raise ValueError("identity_scale must be scalar")
        if statistics.validate_values and (
            not bool(torch.isfinite(identity_scale))
            or not bool(identity_scale > 0)
        ):
            raise ValueError("identity_scale must be positive and finite")
    elif not math.isfinite(identity_scale) or identity_scale <= 0.0:
        raise ValueError("identity_scale must be positive and finite")
    fractions = tuple(float(value) for value in prior_fractions)
    if any(
        not math.isfinite(value) or not 0.0 < value < 1.0
        for value in fractions
    ):
        raise ValueError("official-prior fractions must lie strictly in (0,1)")
    if tuple(sorted(set(fractions))) != fractions:
        raise ValueError("official-prior fractions must be unique and increasing")

    try:
        target, solver = solve_cross_term_target(statistics)
    except ValueError as error:
        if str(error) != "cross-term H is not positive definite":
            raise
    else:
        # Preserve the pre-stabilization evidence bytes for ordinary PD
        # experts. Only experts that actually need the fallback gain new
        # stabilization fields, so a resumed run remains canonical.
        return target, solver

    for fraction in fractions:
        stabilized = shrink_cross_term_objective(
            statistics,
            official_weight_exl,
            local_alpha=1.0 - fraction,
            identity_scale=identity_scale,
        )
        try:
            target, solver = solve_cross_term_target(stabilized)
        except ValueError as error:
            if str(error) == "cross-term H is not positive definite":
                continue
            raise
        return target, {
            **solver,
            "stabilization": "minimal_joint_official_bf16_prior",
            "official_prior_fraction": fraction,
            "effective_candidate_fraction": 1.0 - fraction,
            "identity_scale": float(identity_scale),
        }
    raise ValueError("cross-term H remained non-PD after official-prior grid")


__all__ = ["DEFAULT_PRIOR_FRACTIONS", "solve_with_minimal_official_prior"]
