"""Fit/selection/holdout-safe calibration reductions for fresh GLM SQG.

All large arrays are memory mapped.  The API exposes roles explicitly and
never returns an unlabelled row set, which prevents holdout data from leaking
into Hessians, permutations, transforms, or shared-profile choices.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .calibration_capture import (
    HIDDEN,
    NUM_EXPERTS,
    TOPK,
    validate_capture,
)
from .calibration_hessian import (
    FROZEN_KQUANT_CANDIDATE_HESSIAN_SHA256,
    H2_LOCAL_ALPHA_CAP,
    H2_MIN_FIT_DOCUMENTS,
    H2_MIN_ROUTED_ROWS,
    build_candidate_h2_dense,
    validate_h2_support,
)
from .fresh_pipeline_common import (
    HADAMARD_BLOCK,
    INTERMEDIATE,
    ROLE_TO_ID,
    ROLES,
    canonical_sha256,
    expand_blocks,
    normalized_quarter_scales,
    permutation_scope,
    sha256_file,
    validate_layer,
)
from .glm52_fresh_sqg import (
    DenseHessian,
    FreshExpertPermutation,
    SharedResidualProfile,
    derive_glm_h2_reverse_permutation,
)
from .glm52_fresh_sqg.reference import tensor_sha256


H13_CONSTRUCTION = "fit_gate_square_layer_global_dense_covariance_v1"
H2_CONSTRUCTION = (
    "fit_applied_gate_square_decoded_gate_up_candidate_conditional_"
    "weighted_oas_scaled_identity_cap_0p75_v1"
)
PROFILE_SCALE_CONSTRUCTION = (
    "official_bf16_fit_mass_aggregate_rms_plus_bmmlaw_quarter_family_v1"
)


def _torch_from_bf16_words(words: np.ndarray) -> torch.Tensor:
    if words.dtype != np.dtype("<u2") and words.dtype != np.dtype("uint16"):
        raise TypeError("BF16 word array must use little-endian uint16")
    # Advanced indexing is expected to copy.  Make that ownership explicit so
    # PyTorch never aliases a read-only memmap.
    return torch.from_numpy(np.array(words, dtype=np.uint16, copy=True)).view(
        torch.bfloat16
    )


def _chunks(total: int, rows: int) -> Iterator[tuple[int, int]]:
    if rows <= 0:
        raise ValueError("chunk row count must be positive")
    for begin in range(0, total, rows):
        yield begin, min(total, begin + rows)


@dataclass(frozen=True)
class RoutedRows:
    role: str
    expert: int
    row_indices: np.ndarray
    route_slots: np.ndarray
    applied_gates: torch.Tensor
    document_epochs: torch.Tensor

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError("routed rows have an invalid role")
        rows = int(self.row_indices.size)
        if (
            self.row_indices.ndim != 1
            or self.route_slots.shape != self.row_indices.shape
            or self.applied_gates.shape != (rows,)
            or self.document_epochs.shape != (rows,)
        ):
            raise ValueError("routed-row fields have inconsistent shapes")
        if rows and np.any(self.row_indices[1:] <= self.row_indices[:-1]):
            raise ValueError("expert routed rows must be unique and increasing")
        if not bool(torch.isfinite(self.applied_gates).all()) or bool(
            (self.applied_gates < 0).any()
        ):
            raise ValueError("routed applied gates are invalid")

    @property
    def rows(self) -> int:
        return int(self.row_indices.size)

    @property
    def distinct_documents(self) -> int:
        return int(torch.unique(self.document_epochs).numel())

    @property
    def gate_square_weights(self) -> torch.Tensor:
        return self.applied_gates.float().square()

    def evidence(self) -> dict[str, object]:
        row_tensor = torch.from_numpy(self.row_indices.astype(np.int64, copy=True))
        slot_tensor = torch.from_numpy(self.route_slots.astype(np.int16, copy=True))
        return {
            "role": self.role,
            "expert": self.expert,
            "rows": self.rows,
            "distinct_documents": self.distinct_documents,
            "row_indices_sha256": tensor_sha256(row_tensor),
            "route_slots_sha256": tensor_sha256(slot_tensor),
            "applied_gates_sha256": tensor_sha256(self.applied_gates.float()),
            "gate_square_sum": float(self.gate_square_weights.double().sum()),
        }


class LayerCaptureView:
    """Memory-mapped, role-aware view of one sealed layer capture."""

    def __init__(
        self,
        capture_dir: str | Path,
        layer: int,
        *,
        validate: bool = True,
        verify_hashes: bool = True,
    ) -> None:
        self.root = Path(capture_dir).resolve()
        self.layer = validate_layer(layer)
        if validate:
            self.capture_manifest = validate_capture(
                self.root, verify_hashes=verify_hashes
            )
        else:
            import json

            self.capture_manifest = json.loads(
                (self.root / "capture_manifest.json").read_text()
            )
        self.directory = self.root / f"layer_{self.layer:03d}"
        import json

        self.layer_manifest = json.loads(
            (self.directory / "layer_manifest.json").read_text()
        )
        self.rows = int(self.layer_manifest["tokens"])
        self.hidden_words = np.memmap(
            self.directory / "hidden.bf16.bin",
            mode="r",
            dtype="<u2",
            shape=(self.rows, HIDDEN),
        )
        self.topk_ids = np.memmap(
            self.directory / "topk_ids.u8.bin",
            mode="r",
            dtype="u1",
            shape=(self.rows, TOPK),
        )
        self.topk_weights = np.memmap(
            self.directory / "topk_weights.f32le.bin",
            mode="r",
            dtype="<f4",
            shape=(self.rows, TOPK),
        )
        self.doc_epochs = np.memmap(
            self.directory / "doc_epochs.u32le.bin",
            mode="r",
            dtype="<u4",
            shape=(self.rows,),
        )
        self.role_ids = np.memmap(
            self.directory / "role_ids.u8.bin",
            mode="r",
            dtype="u1",
            shape=(self.rows,),
        )

    def role_mask(self, role: str) -> np.ndarray:
        if role not in ROLE_TO_ID:
            raise ValueError(f"invalid capture role {role!r}")
        return np.asarray(self.role_ids == ROLE_TO_ID[role])

    def role_rows(self, role: str) -> np.ndarray:
        return np.flatnonzero(self.role_mask(role)).astype(np.int64, copy=False)

    def load_hidden(
        self,
        row_indices: np.ndarray,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if row_indices.ndim != 1:
            raise ValueError("hidden row indices must be one-dimensional")
        if row_indices.size and (
            int(row_indices[0]) < 0 or int(row_indices[-1]) >= self.rows
        ):
            raise ValueError("hidden row index lies outside capture")
        bf16 = _torch_from_bf16_words(self.hidden_words[row_indices])
        return bf16.to(device=device, dtype=dtype).contiguous()

    def routed_rows(self, expert: int, role: str) -> RoutedRows:
        if not 0 <= int(expert) < NUM_EXPERTS:
            raise ValueError("expert must lie in [0,255]")
        if role not in ROLE_TO_ID:
            raise ValueError(f"invalid capture role {role!r}")
        match = (np.asarray(self.topk_ids) == int(expert)) & (
            np.asarray(self.role_ids)[:, None] == ROLE_TO_ID[role]
        )
        rows, slots = np.nonzero(match)
        # Capture validation already prohibits duplicate experts per token.
        if rows.size and np.any(rows[1:] == rows[:-1]):
            raise ValueError("capture routes one token to the same expert twice")
        gates = torch.from_numpy(
            np.array(self.topk_weights[rows, slots], dtype=np.float32, copy=True)
        )
        epochs = torch.from_numpy(
            np.array(self.doc_epochs[rows], dtype=np.int64, copy=True)
        )
        return RoutedRows(
            role=role,
            expert=int(expert),
            row_indices=rows.astype(np.int64, copy=False),
            route_slots=slots.astype(np.int16, copy=False),
            applied_gates=gates,
            document_epochs=epochs,
        )

    def role_gate_square_mass_by_expert(self, role: str) -> torch.Tensor:
        if role not in ROLE_TO_ID:
            raise ValueError(f"invalid capture role {role!r}")
        result = torch.zeros(NUM_EXPERTS, dtype=torch.float64)
        mask = self.role_mask(role)
        ids = torch.from_numpy(np.array(self.topk_ids[mask], copy=True)).long()
        gates = torch.from_numpy(
            np.array(self.topk_weights[mask], dtype=np.float64, copy=True)
        )
        result.scatter_add_(0, ids.flatten(), gates.square().flatten())
        if not bool(torch.isfinite(result).all()) or float(result.sum()) <= 0:
            raise ValueError(f"{role} routed gate-square mass is invalid")
        return result

    def binding(self) -> dict[str, object]:
        manifest_path = self.directory / "layer_manifest.json"
        return {
            "capture_manifest_sha256": sha256_file(
                self.root / "capture_manifest.json"
            ),
            "layer_manifest_sha256": sha256_file(manifest_path),
            "capture_run_uuid": self.layer_manifest["capture_run_uuid"],
            "document_plan_fingerprint": self.layer_manifest[
                "document_plan_fingerprint"
            ],
            "layer": self.layer,
        }


def build_layer_h13(
    capture: LayerCaptureView,
    *,
    device: str | torch.device,
    chunk_rows: int = 512,
    accumulator_dtype: torch.dtype = torch.float32,
) -> tuple[DenseHessian, dict[str, object]]:
    """Build fit-only layer-global H13 weighted by total applied-gate square."""

    device = torch.device(device)
    if accumulator_dtype not in (torch.float32, torch.float64):
        raise ValueError("H13 accumulator must use float32 or float64")
    accumulator = torch.zeros(
        (HIDDEN, HIDDEN), dtype=accumulator_dtype, device=device
    )
    denominator = 0.0
    selected_rows = 0
    role_id = ROLE_TO_ID["fit"]
    for begin, end in _chunks(capture.rows, chunk_rows):
        local_roles = np.asarray(capture.role_ids[begin:end])
        local_mask = local_roles == role_id
        if not local_mask.any():
            continue
        words = capture.hidden_words[begin:end][local_mask]
        x = _torch_from_bf16_words(words).to(
            device=device, dtype=accumulator_dtype
        )
        gates = torch.from_numpy(
            np.array(
                capture.topk_weights[begin:end][local_mask],
                dtype=np.float64 if accumulator_dtype == torch.float64 else np.float32,
                copy=True,
            )
        ).to(device=device, dtype=accumulator_dtype)
        row_weights = gates.square().sum(dim=1)
        if not bool(torch.isfinite(x).all()) or not bool(
            torch.isfinite(row_weights).all()
        ):
            raise ValueError("fit H13 input is non-finite")
        accumulator.addmm_(x.T, x * row_weights[:, None])
        denominator += float(row_weights.double().sum())
        selected_rows += int(x.shape[0])
    if selected_rows <= 0 or not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("fit capture cannot construct a nonempty H13")
    covariance = accumulator.div_(denominator)
    covariance = ((covariance + covariance.T) * 0.5).float().cpu().contiguous()
    diag_mean = float(covariance.diagonal().double().mean())
    if (
        not bool(torch.isfinite(covariance).all())
        or not math.isfinite(diag_mean)
        or diag_mean <= 1e-20
    ):
        raise ValueError("fit H13 is degenerate; fallback is forbidden")
    evidence = {
        "construction": H13_CONSTRUCTION,
        "role": "fit",
        "layer": capture.layer,
        "fit_rows": selected_rows,
        "gate_square_sum": denominator,
        "matrix_sha256": tensor_sha256(covariance),
        "diagonal_mean": diag_mean,
        "accumulator_dtype": str(accumulator_dtype).removeprefix("torch."),
        "capture": capture.binding(),
    }
    evidence_id = canonical_sha256(evidence)
    dense = DenseHessian(
        matrix=covariance,
        evidence_id=evidence_id,
        construction=H13_CONSTRUCTION,
        split_id="fit",
        normalization_count=1,
        routed_sample_count=selected_rows,
    )
    return dense, {**evidence, "evidence_id": evidence_id}


def replay_official_middle(
    capture: LayerCaptureView,
    routed: RoutedRows,
    gate_hf: torch.Tensor,
    up_hf: torch.Tensor,
    *,
    device: str | torch.device,
    chunk_rows: int = 512,
) -> torch.Tensor:
    """Replay exact official-BF16 source gate/up on one explicit row role."""

    if gate_hf.dtype != torch.bfloat16 or up_hf.dtype != torch.bfloat16:
        raise TypeError("official middle replay requires preserved BF16 source")
    if tuple(gate_hf.shape) != (INTERMEDIATE, HIDDEN) or gate_hf.shape != up_hf.shape:
        raise ValueError("official gate/up geometry differs")
    if routed.rows == 0:
        raise ValueError("expert has no routed fit rows; fallback is forbidden")
    target = torch.device(device)
    gate = gate_hf.to(device=target, dtype=torch.float32)
    up = up_hf.to(device=target, dtype=torch.float32)
    result: list[torch.Tensor] = []
    for begin, end in _chunks(routed.rows, chunk_rows):
        indices = routed.row_indices[begin:end]
        x = capture.load_hidden(indices, device=target, dtype=torch.float32)
        middle = F.silu(F.linear(x, gate)) * F.linear(x, up)
        if not bool(torch.isfinite(middle).all()):
            raise ValueError("official BF16 post-SiLU replay is non-finite")
        result.append(middle.cpu())
    return torch.cat(result, dim=0).contiguous()


def derive_expert_permutation(
    capture: LayerCaptureView,
    expert: int,
    gate_hf: torch.Tensor,
    up_hf: torch.Tensor,
    *,
    device: str | torch.device,
    chunk_rows: int = 512,
) -> tuple[FreshExpertPermutation, dict[str, object]]:
    routed = capture.routed_rows(expert, "fit")
    try:
        validate_h2_support(
            routed_rows=routed.rows,
            fit_documents=routed.distinct_documents,
        )
    except ValueError as error:
        raise ValueError(
            f"L{capture.layer} E{expert}: insufficient fit support for fresh "
            "h2_reverse; no permutation fallback is permitted: {error}"
        ) from error
    middle = replay_official_middle(
        capture,
        routed,
        gate_hf,
        up_hf,
        device=device,
        chunk_rows=chunk_rows,
    )
    evidence_seed = {
        "construction": "kquant_h2_reverse_4_neuron_16x128_v1",
        "layer": capture.layer,
        "expert": expert,
        "routed": routed.evidence(),
        "middle_sha256": tensor_sha256(middle),
        "source_gate_payload_sha256": hashlib.sha256(
            gate_hf.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "source_up_payload_sha256": hashlib.sha256(
            up_hf.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }
    permutation = derive_glm_h2_reverse_permutation(
        middle,
        routed.applied_gates,
        scope=permutation_scope(capture.layer, expert),
        evidence_id=canonical_sha256(evidence_seed),
        split_id="fit",
    )
    evidence = {
        **evidence_seed,
        "permutation": permutation.manifest(),
        "fit_only": True,
        "legacy_permutation_input": False,
    }
    del middle
    return permutation, evidence


def build_candidate_h2(
    capture: LayerCaptureView,
    routed: RoutedRows,
    decoded_gate_exl: torch.Tensor,
    decoded_up_exl: torch.Tensor,
    *,
    device: str | torch.device,
    upstream_evidence: Mapping[str, object],
    chunk_rows: int = 512,
) -> tuple[DenseHessian, dict[str, object]]:
    """Construct one decoded-upstream, expert-local, fit-only dense H2."""

    if routed.role != "fit":
        raise ValueError("candidate H2 may consume only fit rows")
    try:
        validate_h2_support(
            routed_rows=routed.rows,
            fit_documents=routed.distinct_documents,
        )
    except ValueError as error:
        raise ValueError(
            "candidate H2 lacks frozen expert-local fit support; no fallback is "
            f"permitted: {error}"
        ) from error
    if tuple(decoded_gate_exl.shape) != (HIDDEN, INTERMEDIATE) or (
        decoded_up_exl.shape != decoded_gate_exl.shape
    ):
        raise ValueError("decoded gate/up EXL geometry differs")
    # The hash-sealed capture implementation takes HF [out,in] orientation;
    # the bridge's independent decode is EXL [in,out].  This transpose is the
    # sole orientation conversion and is itself bound in the evidence below.
    decoded_gate_hf = decoded_gate_exl.detach().T.cpu().contiguous()
    decoded_up_hf = decoded_up_exl.detach().T.cpu().contiguous()
    shrunk, frozen_evidence = build_candidate_h2_dense(
        capture.root,
        capture.layer,
        routed.expert,
        decoded_gate_hf,
        decoded_up_hf,
        device=device,
        chunk_rows=chunk_rows,
        return_cpu=True,
    )
    shrunk = shrunk.float().cpu().contiguous()
    diag_mean = float(shrunk.diagonal().double().mean())
    if not math.isfinite(diag_mean) or diag_mean <= 1e-20:
        raise ValueError("shrunk candidate H2 remains degenerate")
    evidence = {
        "construction": H2_CONSTRUCTION,
        "role": "fit",
        "layer": capture.layer,
        "expert": routed.expert,
        "routed": routed.evidence(),
        "matrix_sha256": tensor_sha256(shrunk),
        "gate_square_sum": float(routed.gate_square_weights.double().sum()),
        "shrinkage": {
            "effective_sample_size": frozen_evidence["effective_sample_size"],
            "oas_shrinkage": frozen_evidence["oas_shrinkage"],
            "local_alpha": frozen_evidence["local_alpha"],
            "max_local_alpha": frozen_evidence["local_alpha_cap"],
            "identity_scale": frozen_evidence["identity_scale"],
        },
        "support_floor": {
            "minimum_routed_rows": H2_MIN_ROUTED_ROWS,
            "minimum_fit_documents": H2_MIN_FIT_DOCUMENTS,
        },
        "frozen_kquant_candidate_hessian_sha256": (
            FROZEN_KQUANT_CANDIDATE_HESSIAN_SHA256
        ),
        "local_alpha_cap": H2_LOCAL_ALPHA_CAP,
        "hash_sealed_capture_h2_evidence": frozen_evidence,
        "diagonal_mean": diag_mean,
        "upstream_candidate": dict(upstream_evidence),
        "identity_fallback": False,
        "pooled_expert_basis": False,
    }
    evidence_id = canonical_sha256(evidence)
    dense = DenseHessian(
        matrix=shrunk,
        evidence_id=evidence_id,
        construction=H2_CONSTRUCTION,
        split_id="fit",
        normalization_count=1,
        routed_sample_count=routed.rows,
    )
    return dense, {**evidence, "evidence_id": evidence_id}


def mass_stratified_experts(masses: torch.Tensor, count: int = 16) -> tuple[int, ...]:
    """Select hot/cold and cumulative-mass quantiles deterministically."""

    if masses.shape != (NUM_EXPERTS,) or not 2 <= count <= NUM_EXPERTS:
        raise ValueError("invalid mass-stratified panel request")
    numeric = masses.double().tolist()
    total = sum(numeric)
    if total <= 0:
        raise ValueError("routed mass is empty")
    ordered = sorted(range(NUM_EXPERTS), key=lambda expert: (-numeric[expert], expert))
    cumulative: list[tuple[float, int]] = []
    running = 0.0
    for expert in ordered:
        running += numeric[expert]
        cumulative.append((running / total, expert))
    selected = {ordered[0], ordered[-1]}
    for index in range(count):
        target = (index + 0.5) / count
        selected.add(
            min(cumulative, key=lambda item: (abs(item[0] - target), item[1]))[1]
        )
    for expert in ordered:
        if len(selected) >= count:
            break
        selected.add(expert)
    return tuple(
        sorted(selected, key=lambda expert: (-numeric[expert], expert))[:count]
    )


@dataclass(frozen=True)
class ProfileScaleEvidence:
    gate_input_base: torch.Tensor
    down_output_base: torch.Tensor
    gate_block_quarter: torch.Tensor
    down_block_quarter: torch.Tensor
    evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in ("gate_input_base", "down_output_base"):
            value = getattr(self, name)
            if value.shape != (HIDDEN,) or not bool(torch.isfinite(value).all()) or bool(
                (value <= 0).any()
            ):
                raise ValueError(f"invalid fresh shared-profile scale evidence: {name}")
        for name in ("gate_block_quarter", "down_block_quarter"):
            value = getattr(self, name)
            if value.shape != (HIDDEN // HADAMARD_BLOCK,):
                raise ValueError(f"invalid block scale evidence: {name}")

    def families(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        gate_adjust = expand_blocks(self.gate_block_quarter, HIDDEN)
        down_adjust = expand_blocks(self.down_block_quarter, HIDDEN)

        def relative(value: torch.Tensor) -> torch.Tensor:
            result = value.float().clamp_min(1e-20)
            return (result / result.mean()).contiguous()

        # KQuant consumes an explicit input-channel profile as the absolute
        # row-RMS scale; unlike its implicit output profile, it does not divide
        # this vector by its mean.  Keep every gate/up family at the absolute
        # aggregate RMS amplitude while varying only its relative modulation.
        gate_reference = self.gate_input_base.float().mean()

        def absolute_gate(value: torch.Tensor) -> torch.Tensor:
            return (relative(value) * gate_reference).contiguous()

        return {
            # A genuine fresh-codec control: uniform absolute input RMS with no
            # channel modulation, but still freshly drawn signs and full
            # KQuant calibration/encoding. This is not an inherited scale.
            "identity": (
                torch.full_like(
                    self.gate_input_base,
                    float(gate_reference),
                    dtype=torch.float32,
                ),
                torch.ones_like(self.down_output_base, dtype=torch.float32),
            ),
            "aggregate_rms": (
                absolute_gate(self.gate_input_base),
                relative(self.down_output_base),
            ),
            "quarter_rms": (
                absolute_gate(self.gate_input_base * gate_adjust),
                relative(self.down_output_base * down_adjust),
            ),
            "inverse_quarter_rms": (
                absolute_gate(self.gate_input_base / gate_adjust),
                relative(self.down_output_base / down_adjust),
            ),
        }


def build_profile_scale_evidence(
    capture: LayerCaptureView,
    source: object,
    h13: DenseHessian,
    *,
    device: str | torch.device,
) -> ProfileScaleEvidence:
    """Build fresh aggregate channel scales from BF16 weights and fit mass.

    ``source`` is duck-typed as :class:`BF16ExpertSource` to keep this module's
    CPU-only tests independent from the 90-GB source provider.
    """

    if h13.split_id != "fit" or tuple(h13.matrix.shape) != (HIDDEN, HIDDEN):
        raise ValueError("shared profile scale construction requires fit H13")
    masses = capture.role_gate_square_mass_by_expert("fit")
    total_mass = float(masses.sum())
    gate_energy = torch.zeros(HIDDEN, dtype=torch.float64)
    down_energy = torch.zeros(HIDDEN, dtype=torch.float64)
    for expert in range(NUM_EXPERTS):
        weights = source.load_expert_bf16(capture.layer, expert, device="cpu")
        mass = float(masses[expert])
        if mass <= 0:
            raise ValueError(
                f"L{capture.layer} E{expert}: no fit mass for shared-profile "
                "construction; fallback is forbidden"
            )
        gate = weights.gate_hf.float()
        up = weights.up_hf.float()
        down = weights.down_hf.float()
        # Shared gate/up input side: aggregate official source energy for each
        # hidden input coordinate across both matrices.  Shared down output
        # side: KQuant's output-channel RMS statistic in HF orientation.
        gate_energy += mass * (
            gate.square().mean(dim=0).double()
            + up.square().mean(dim=0).double()
        ) * 0.5
        down_energy += mass * down.square().mean(dim=1).double()
    gate_energy /= total_mass
    down_energy /= total_mass
    gate_base = gate_energy.clamp_min(1e-30).sqrt().float()
    down_base = down_energy.clamp_min(1e-30).sqrt().float()
    h13_blocks = (
        h13.matrix.diagonal().double().reshape(-1, HADAMARD_BLOCK).mean(dim=1)
    )
    down_blocks = down_energy.reshape(-1, HADAMARD_BLOCK).mean(dim=1)
    gate_quarter = normalized_quarter_scales(h13_blocks.tolist())
    down_quarter = normalized_quarter_scales(down_blocks.tolist())
    evidence = {
        "construction": PROFILE_SCALE_CONSTRUCTION,
        "role": "fit",
        "layer": capture.layer,
        "fit_gate_square_mass_sha256": tensor_sha256(masses),
        "fit_gate_square_mass_sum": total_mass,
        "gate_input_base_sha256": tensor_sha256(gate_base),
        "down_output_base_sha256": tensor_sha256(down_base),
        "gate_block_quarter_sha256": tensor_sha256(gate_quarter),
        "down_block_quarter_sha256": tensor_sha256(down_quarter),
        "h13_evidence_id": h13.evidence_id,
        "official_bf16_only": True,
        "legacy_scale_input": False,
    }
    return ProfileScaleEvidence(
        gate_input_base=gate_base,
        down_output_base=down_base,
        gate_block_quarter=gate_quarter,
        down_block_quarter=down_quarter,
        evidence={**evidence, "evidence_id": canonical_sha256(evidence)},
    )


def make_shared_profiles(
    *,
    layer: int,
    draw: int,
    family: str,
    scales: ProfileScaleEvidence,
    sign_factory,
) -> tuple[SharedResidualProfile, SharedResidualProfile]:
    """Create one paired fresh sign/scale candidate for selection scoring."""

    families = scales.families()
    if family not in families:
        raise ValueError(f"unknown scale family {family!r}")
    gate_scales, down_scales = families[family]
    gate_signs = sign_factory(HIDDEN, layer, "shared_gate_up_input", draw)
    down_signs = sign_factory(HIDDEN, layer, "shared_down_output", draw)
    evidence_id = str(scales.evidence["evidence_id"])
    gate = SharedResidualProfile(
        side="input",
        signs=gate_signs,
        channel_scales=gate_scales,
        profile_id=f"L{layer:03d}/draw-{draw:02d}/{family}/gate-up-input",
        derivation=(
            f"{PROFILE_SCALE_CONSTRUCTION};fit_evidence={evidence_id};"
            "choice_role=selection;legacy_input=false"
        ),
    )
    down = SharedResidualProfile(
        side="output",
        signs=down_signs,
        channel_scales=down_scales,
        profile_id=f"L{layer:03d}/draw-{draw:02d}/{family}/down-output",
        derivation=(
            f"{PROFILE_SCALE_CONSTRUCTION};fit_evidence={evidence_id};"
            "choice_role=selection;legacy_input=false"
        ),
    )
    return gate, down
