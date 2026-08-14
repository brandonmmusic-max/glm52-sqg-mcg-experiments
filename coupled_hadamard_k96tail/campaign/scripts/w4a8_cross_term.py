"""Cross-term-aware down-projection objectives for GLM W4A8/h-A8.

For an exact upstream candidate operand ``Q`` and BF16 teacher output ``Y``,
the fitted down target minimizes ``||Q W - Y||^2`` through the normal
equations ``H W = B`` where ``H = Q.T Q`` and ``B = Q.T Y``.  Reliability
shrinkage is applied jointly to both sufficient statistics so the prior
optimum remains the official BF16 down matrix instead of being biased toward
zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class CrossTermStatistics:
    hessian: torch.Tensor
    cross_term: torch.Tensor
    weight_sum: float | torch.Tensor
    rows: int
    validate_values: bool = True

    def __post_init__(self) -> None:
        hessian = self.hessian
        cross_term = self.cross_term
        if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
            raise ValueError("H must be a square matrix")
        if cross_term.ndim != 2 or cross_term.shape[0] != hessian.shape[0]:
            raise ValueError("B input dimension must match H")
        if hessian.dtype != torch.float32 or cross_term.dtype != torch.float32:
            raise TypeError("cross-term sufficient statistics must use float32")
        if hessian.device != cross_term.device:
            raise ValueError("H and B must share one device")
        if not isinstance(self.validate_values, bool):
            raise TypeError("validate_values must be boolean")
        if self.validate_values:
            if not bool(torch.isfinite(hessian).all()) or not bool(
                torch.isfinite(cross_term).all()
            ):
                raise ValueError("cross-term sufficient statistics must be finite")
            if not torch.allclose(hessian, hessian.T, rtol=1e-5, atol=1e-6):
                raise ValueError("H must be symmetric")
        if isinstance(self.weight_sum, torch.Tensor):
            if self.weight_sum.numel() != 1:
                raise ValueError("weight_sum must be scalar")
            if self.validate_values and (
                not bool(torch.isfinite(self.weight_sum))
                or not bool(self.weight_sum > 0)
            ):
                raise ValueError("weight_sum must be positive and finite")
        elif not math.isfinite(self.weight_sum) or self.weight_sum <= 0:
            raise ValueError("weight_sum must be positive and finite")
        if self.rows <= 0:
            raise ValueError("rows must be positive")


def kquant_prefinalize_operand_from_label_operand(
    label_operand: torch.Tensor,
    finalize_input_signs: torch.Tensor,
    hadamard: torch.Tensor,
    *,
    validate_values: bool = True,
) -> torch.Tensor:
    """Map the realized label operand into KQuant's caller-H coordinates.

    KQuant's ``finalize_capture_H`` maps a caller-owned Hessian ``A`` to
    ``R D A D R``, where ``R`` is blockwise normalized H128 and ``D`` is the
    exact input-sign diagonal used by finalize.  For a realized A8/native-label
    operand ``Q``, returning ``Q_pre = Q R D`` makes that congruence recover
    the required label-space Hessian exactly::

        R D (Q_pre.T @ Q_pre) D R = Q.T @ Q

    This operand is only for the encoder DenseH.  It must not replace the
    separately inverse-scaled canonical operand used by an ``(H, B)`` target
    solve.
    """

    if label_operand.ndim != 2:
        raise ValueError("label operand must be rank two")
    if not label_operand.dtype.is_floating_point:
        raise TypeError("label operand must use a floating dtype")
    rows, columns = label_operand.shape
    if rows <= 0 or columns <= 0:
        raise ValueError("label operand must be nonempty")
    if validate_values and not bool(torch.isfinite(label_operand).all()):
        raise ValueError("label operand must be finite")

    if finalize_input_signs.ndim != 1 or finalize_input_signs.numel() != columns:
        raise ValueError("finalize input signs do not match the operand width")
    signs = finalize_input_signs.to(
        device=label_operand.device, dtype=torch.float32
    )
    if validate_values and (not bool(torch.isfinite(signs).all()) or not bool(
        ((signs == -1) | (signs == 1)).all()
    )):
        raise ValueError("finalize input signs must contain finite exact +/-1 values")

    if hadamard.ndim != 2 or hadamard.shape[0] != hadamard.shape[1]:
        raise ValueError("Hadamard must be square")
    if not hadamard.dtype.is_floating_point:
        raise TypeError("Hadamard must use a floating dtype")
    block = int(hadamard.shape[0])
    if block <= 0 or columns % block:
        raise ValueError("operand width must be divisible by the Hadamard block")
    rotation = hadamard.to(device=label_operand.device, dtype=torch.float32)
    if validate_values and not bool(torch.isfinite(rotation).all()):
        raise ValueError("Hadamard must be finite")
    if validate_values and not torch.allclose(
        rotation, rotation.T, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("Hadamard must be symmetric")
    identity = torch.eye(block, device=rotation.device, dtype=rotation.dtype)
    if validate_values and not torch.allclose(
        rotation @ rotation,
        identity,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError("Hadamard must be normalized and involutory")

    blocks = label_operand.float().reshape(rows, -1, block)
    prefinalize = torch.matmul(blocks, rotation).reshape(rows, columns)
    prefinalize = (prefinalize * signs).contiguous()
    if validate_values and not bool(torch.isfinite(prefinalize).all()):
        raise RuntimeError("KQuant pre-finalize operand is non-finite")
    return prefinalize


def effective_canonical_operand_from_quantized_transform(
    quantized_operand: torch.Tensor,
    stored_suh: torch.Tensor,
    hadamard: torch.Tensor,
    *,
    validate_values: bool = True,
) -> torch.Tensor:
    """Map an actually quantized post-H operand back to canonical coordinates.

    The down runtime forms ``Q = QDQ((act * suh) @ H)``.  For fitting a
    canonical down target while preserving that exact nonlinear QDQ result,
    use ``Q_eff = (Q @ H) / suh``.  If the encoder reproduces the anchored
    ``suh`` bytes, ``Q_eff @ W_canonical`` is algebraically identical to the
    native-coordinate W4A8 path before the output transform.  A later scorer
    remains authoritative when the re-encode changes the private input scale.
    """

    if quantized_operand.ndim != 2:
        raise ValueError("quantized operand must be rank two")
    if stored_suh.ndim != 1 or stored_suh.numel() != quantized_operand.shape[1]:
        raise ValueError("stored suh does not match the operand width")
    if hadamard.ndim != 2 or hadamard.shape[0] != hadamard.shape[1]:
        raise ValueError("Hadamard must be square")
    block = int(hadamard.shape[0])
    rows, columns = quantized_operand.shape
    if columns % block:
        raise ValueError("operand width must be divisible by the Hadamard block")
    suh = stored_suh.to(
        device=quantized_operand.device, dtype=torch.float32
    )
    if validate_values and (
        not bool(torch.isfinite(suh).all()) or bool((suh == 0).any())
    ):
        raise ValueError("stored suh must be finite and nonzero")
    blocks = quantized_operand.float().reshape(rows, -1, block)
    canonical = torch.matmul(blocks, hadamard.float()).reshape(rows, columns)
    canonical = (canonical / suh).contiguous()
    if validate_values and not bool(torch.isfinite(canonical).all()):
        raise ValueError("effective canonical operand is non-finite")
    return canonical


def cross_term_statistics(
    operand: torch.Tensor,
    teacher_output: torch.Tensor,
    route_gates: torch.Tensor,
) -> CrossTermStatistics:
    """Return normalized gate-square-weighted ``(H,B)`` statistics."""

    if operand.ndim != 2 or teacher_output.ndim != 2:
        raise ValueError("operand and teacher output must be rank two")
    if operand.shape[0] != teacher_output.shape[0]:
        raise ValueError("operand and teacher output row counts differ")
    if route_gates.ndim != 1 or route_gates.shape[0] != operand.shape[0]:
        raise ValueError("route gates must provide one scalar per row")
    if operand.shape[0] == 0:
        raise ValueError("cross-term capture is empty")
    q = operand.float()
    y = teacher_output.float()
    weights = route_gates.float().square()
    if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("operand and teacher output must be finite")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("route gates must be finite")
    denominator = weights.double().sum()
    if not bool(torch.isfinite(denominator)) or not bool(denominator > 0):
        raise ValueError("route gates have no positive squared mass")
    hessian = q.T @ (q * weights[:, None])
    cross_term = q.T @ (y * weights[:, None])
    hessian = (hessian / denominator).float()
    hessian = ((hessian + hessian.T) * 0.5).contiguous()
    cross_term = (cross_term / denominator).float().contiguous()
    return CrossTermStatistics(
        hessian=hessian,
        cross_term=cross_term,
        weight_sum=denominator,
        rows=int(q.shape[0]),
    )


def shrink_cross_term_objective(
    statistics: CrossTermStatistics,
    official_weight_exl: torch.Tensor,
    *,
    local_alpha: float,
    identity_scale: float | torch.Tensor | None = None,
) -> CrossTermStatistics:
    """Shrink ``H`` and ``B`` jointly toward the official BF16 optimum.

    ``official_weight_exl`` uses ``[input, output]`` orientation.  At alpha
    zero the returned system has the exact solution ``official_weight_exl``.
    """

    if not math.isfinite(local_alpha) or not 0.0 <= local_alpha <= 1.0:
        raise ValueError("local_alpha must lie in [0,1]")
    hessian = statistics.hessian
    source = official_weight_exl.to(
        device=hessian.device, dtype=torch.float32
    )
    if source.shape != statistics.cross_term.shape:
        raise ValueError("official down weight shape must match B")
    if statistics.validate_values and not bool(torch.isfinite(source).all()):
        raise ValueError("official down weight must be finite")
    scale = (
        hessian.diagonal().double().mean()
        if identity_scale is None
        else torch.as_tensor(identity_scale, device=hessian.device, dtype=torch.float64)
    )
    if scale.numel() != 1:
        raise ValueError("identity scale must be scalar")
    if statistics.validate_values and (
        not bool(torch.isfinite(scale)) or not bool(scale > 0)
    ):
        raise ValueError("identity scale must be positive and finite")
    scale32 = scale.to(dtype=torch.float32)
    prior = torch.eye(
        hessian.shape[0], dtype=torch.float32, device=hessian.device
    ).mul_(scale32)
    shrunk_h = torch.lerp(prior, hessian, local_alpha)
    shrunk_h = ((shrunk_h + shrunk_h.T) * 0.5).contiguous()
    prior_b = source * scale32
    shrunk_b = torch.lerp(prior_b, statistics.cross_term, local_alpha).contiguous()
    return CrossTermStatistics(
        hessian=shrunk_h,
        cross_term=shrunk_b,
        weight_sum=statistics.weight_sum,
        rows=statistics.rows,
        validate_values=statistics.validate_values,
    )


def solve_cross_term_target(
    statistics: CrossTermStatistics,
    *,
    diagonal_jitter: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    """Solve the down target without explicitly forming an inverse."""

    if not math.isfinite(diagonal_jitter) or diagonal_jitter < 0:
        raise ValueError("diagonal_jitter must be finite and nonnegative")
    hessian = statistics.hessian
    if diagonal_jitter:
        hessian = hessian + torch.eye(
            hessian.shape[0], dtype=hessian.dtype, device=hessian.device
        ) * diagonal_jitter
    factor, info = torch.linalg.cholesky_ex(hessian)
    if int(info.max()) != 0:
        raise ValueError("cross-term H is not positive definite")
    target = torch.cholesky_solve(statistics.cross_term, factor).contiguous()
    if not bool(torch.isfinite(target).all()):
        raise ValueError("cross-term solve produced non-finite weights")
    residual = hessian @ target - statistics.cross_term
    relative_residual = float(
        residual.double().norm()
        / statistics.cross_term.double().norm().clamp_min(1e-30)
    )
    return target, {
        "solver": "torch_cholesky_solve",
        "diagonal_jitter": diagonal_jitter,
        "relative_normal_equation_residual": relative_residual,
    }


def weighted_output_sse(
    operand: torch.Tensor,
    weight_exl: torch.Tensor,
    teacher_output: torch.Tensor,
    route_gates: torch.Tensor,
) -> float:
    """Evaluate the exact gate-square-weighted fitted-output SSE."""

    prediction = operand.float() @ weight_exl.float()
    difference = prediction - teacher_output.float()
    weights = route_gates.float().square()
    return float(
        torch.sum(
            difference.square().sum(dim=1) * weights,
            dtype=torch.float64,
        )
    )


__all__ = [
    "CrossTermStatistics",
    "cross_term_statistics",
    "effective_canonical_operand_from_quantized_transform",
    "kquant_prefinalize_operand_from_label_operand",
    "shrink_cross_term_objective",
    "solve_cross_term_target",
    "weighted_output_sse",
]
