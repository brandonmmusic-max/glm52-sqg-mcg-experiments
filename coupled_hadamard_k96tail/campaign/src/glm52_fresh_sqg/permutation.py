"""Fresh topology-neutral GLM expert-coordinate permutations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
import torch.nn.functional as F

from .reference import tensor_sha256


TEST_PERMUTATION_POLICY = "fresh_random_new_to_old_test_v1"
PRODUCTION_PERMUTATION_POLICY = "fresh_glm_h2_reverse_16x128_v1"
GLM_INTERMEDIATE_FEATURES = 2048
CONTEXT_GROUP_CHANNELS = 4
RECORD_CHANNELS = 128
RECORDS_PER_EXPERT = GLM_INTERMEDIATE_FEATURES // RECORD_CHANNELS


@dataclass(frozen=True)
class FreshExpertPermutation:
    """A newly drawn physical intermediate-coordinate permutation.

    ``new_to_old[new] == old``.  No constructor or loader for an MCG
    permutation is provided by this package.
    """

    seed: int
    scope: str
    new_to_old: torch.Tensor
    policy: str
    evidence_sha256: str | None = None
    block_contexts_sha256: str | None = None
    block_scores_sha256: str | None = None
    calibration_split_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("permutation seed must be a non-negative integer")
        if not self.scope:
            raise ValueError("permutation scope must not be empty")
        if self.policy not in (TEST_PERMUTATION_POLICY, PRODUCTION_PERMUTATION_POLICY):
            raise ValueError("unsupported fresh permutation policy")
        value = self.new_to_old
        if value.ndim != 1 or value.dtype != torch.int64:
            raise TypeError("new_to_old must be a one-dimensional torch.int64 tensor")
        expected = torch.arange(value.numel(), dtype=torch.int64, device=value.device)
        if not torch.equal(torch.sort(value).values, expected):
            raise ValueError("new_to_old must be a complete permutation")
        if self.policy == PRODUCTION_PERMUTATION_POLICY:
            if value.numel() != GLM_INTERMEDIATE_FEATURES:
                raise ValueError("production GLM permutation must be exactly 2048-wide")
            for name in (
                "evidence_sha256",
                "block_contexts_sha256",
                "block_scores_sha256",
            ):
                digest = getattr(self, name)
                if digest is None or len(digest) != 64 or any(
                    char not in "0123456789abcdef" for char in digest
                ):
                    raise ValueError(f"production permutation requires {name}")
            if self.calibration_split_id != "fit":
                raise ValueError(
                    "production permutation requires the exact fit calibration split"
                )
        elif any(
            value is not None
            for value in (
                self.evidence_sha256,
                self.block_contexts_sha256,
                self.block_scores_sha256,
                self.calibration_split_id,
            )
        ):
            raise ValueError("random test permutations cannot claim calibration evidence")

    @property
    def intermediate_features(self) -> int:
        return self.new_to_old.numel()

    @property
    def sha256(self) -> str:
        return tensor_sha256(self.new_to_old)

    @property
    def production_qualified(self) -> bool:
        return self.policy == PRODUCTION_PERMUTATION_POLICY

    def manifest(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "scope": self.scope,
            "new_to_old_sha256": self.sha256,
            "intermediate_features": self.intermediate_features,
            "legacy_permutation_input": False,
            "calibration_evidence_sha256": self.evidence_sha256,
            "block_contexts_sha256": self.block_contexts_sha256,
            "block_scores_sha256": self.block_scores_sha256,
            "calibration_split_id": self.calibration_split_id,
            "production_qualified": self.production_qualified,
        }


def draw_fresh_expert_permutation(
    intermediate_features: int,
    *,
    seed: int,
    scope: str,
) -> FreshExpertPermutation:
    """Draw a deterministic fresh map from only a seed and scope string."""

    if (
        isinstance(intermediate_features, bool)
        or not isinstance(intermediate_features, int)
        or intermediate_features <= 0
    ):
        raise ValueError("intermediate_features must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("permutation seed must be a non-negative integer")
    if not scope:
        raise ValueError("permutation scope must not be empty")
    domain = hashlib.sha256(
        f"{TEST_PERMUTATION_POLICY}\0{seed}\0{scope}".encode()
    ).digest()
    derived_seed = int.from_bytes(domain[:8], "little") & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu").manual_seed(derived_seed)
    new_to_old = torch.randperm(
        intermediate_features,
        generator=generator,
        dtype=torch.int64,
    )
    return FreshExpertPermutation(
        seed=seed,
        scope=scope,
        new_to_old=new_to_old,
        policy=TEST_PERMUTATION_POLICY,
    )


def derive_glm_h2_reverse_permutation(
    middle: torch.Tensor,
    route_gates: torch.Tensor,
    *,
    scope: str,
    evidence_id: str,
    split_id: str,
) -> FreshExpertPermutation:
    """Derive KQuant's fresh GLM h2-reverse order from calibration rows.

    Four-neuron groups are ranked by gate-square-weighted post-SiLU energy,
    divided into sixteen equal 128-channel contexts, and stored low-to-high.
    Dense-H LDLQ traverses the encoded K axis backwards, so the highest-energy
    context is visited first.  Only fresh middle rows and applied route gates
    enter the construction.
    """

    if not scope or not evidence_id or not split_id:
        raise ValueError("calibration-derived permutation identity must not be empty")
    if split_id != "fit":
        raise ValueError("production h2_reverse may consume only the fit split")
    if (
        middle.ndim != 2
        or middle.shape[1] != GLM_INTERMEDIATE_FEATURES
        or route_gates.ndim != 1
        or middle.shape[0] != route_gates.shape[0]
        or middle.shape[0] == 0
    ):
        raise ValueError("invalid GLM post-SiLU calibration evidence shapes")
    middle_f = middle.detach().float()
    gates_f = route_gates.detach().to(device=middle_f.device, dtype=torch.float32)
    if not bool(torch.isfinite(middle_f).all()) or not bool(torch.isfinite(gates_f).all()):
        raise ValueError("permutation calibration evidence must be finite")
    weights = gates_f.square()
    denominator = weights.double().sum()
    if not bool(denominator > 0):
        raise ValueError("permutation route gates must have positive squared mass")
    energy = (
        middle_f.square().mul(weights[:, None]).sum(dim=0).double() / denominator
    )
    scores = energy.reshape(-1, CONTEXT_GROUP_CHANNELS).mean(dim=1).float()
    group_order = torch.argsort(scores, stable=True)
    groups_per_context = RECORD_CHANNELS // CONTEXT_GROUP_CHANNELS
    contexts = torch.empty_like(group_order)
    contexts[group_order] = torch.div(
        torch.arange(group_order.numel(), device=group_order.device),
        groups_per_context,
        rounding_mode="floor",
    )
    if torch.bincount(contexts, minlength=RECORDS_PER_EXPERT).tolist() != [
        groups_per_context
    ] * RECORDS_PER_EXPERT:
        raise RuntimeError("h2_reverse did not form sixteen equal 128-channel contexts")
    offsets = torch.arange(CONTEXT_GROUP_CHANNELS, device=group_order.device)
    new_to_old = (
        group_order[:, None] * CONTEXT_GROUP_CHANNELS + offsets[None]
    ).flatten().to(device="cpu", dtype=torch.int64).contiguous()
    evidence_record = {
        "policy": PRODUCTION_PERMUTATION_POLICY,
        "scope": scope,
        "evidence_id": evidence_id,
        "split_id": split_id,
        "middle_sha256": tensor_sha256(middle_f),
        "route_gates_sha256": tensor_sha256(gates_f),
        "rows": int(middle.shape[0]),
    }
    evidence_sha256 = hashlib.sha256(
        repr(sorted(evidence_record.items())).encode()
    ).hexdigest()
    return FreshExpertPermutation(
        seed=0,
        scope=scope,
        new_to_old=new_to_old,
        policy=PRODUCTION_PERMUTATION_POLICY,
        evidence_sha256=evidence_sha256,
        block_contexts_sha256=tensor_sha256(contexts.cpu().to(torch.int16)),
        block_scores_sha256=tensor_sha256(scores.cpu()),
        calibration_split_id=split_id,
    )


def apply_glm_expert_permutation(
    gate_hf: torch.Tensor,
    up_hf: torch.Tensor,
    down_hf: torch.Tensor,
    permutation: FreshExpertPermutation,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply ``P Wgate``, ``P Wup``, and ``Wdown P^T`` on cloned tensors.

    Inputs use Hugging Face orientation: gate/up are ``[I, H]`` and down is
    ``[H, I]``.  The returned tensors never alias the BF16 source tensors.
    """

    if gate_hf.ndim != 2 or up_hf.ndim != 2 or down_hf.ndim != 2:
        raise ValueError("gate, up, and down must be rank-two matrices")
    if gate_hf.shape != up_hf.shape:
        raise ValueError("gate and up shapes must match")
    intermediate, hidden = gate_hf.shape
    if down_hf.shape != (hidden, intermediate):
        raise ValueError("down must have shape [hidden, intermediate]")
    if permutation.intermediate_features != intermediate:
        raise ValueError("permutation width does not match the expert intermediate axis")
    if gate_hf.device != up_hf.device or gate_hf.device != down_hf.device:
        raise ValueError("expert matrices must be on the same device")
    order = permutation.new_to_old.to(device=gate_hf.device)
    gate = gate_hf.detach().index_select(0, order).clone().contiguous()
    up = up_hf.detach().index_select(0, order).clone().contiguous()
    down = down_hf.detach().index_select(1, order).clone().contiguous()
    return gate, up, down


def glm_expert_forward(
    inputs: torch.Tensor,
    gate_hf: torch.Tensor,
    up_hf: torch.Tensor,
    down_hf: torch.Tensor,
) -> torch.Tensor:
    """Reference GLM routed-expert equation for permutation closure tests."""

    gate = F.linear(inputs, gate_hf)
    up = F.linear(inputs, up_hf)
    return F.linear(F.silu(gate) * up, down_hf)
