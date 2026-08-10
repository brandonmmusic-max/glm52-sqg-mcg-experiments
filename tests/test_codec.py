from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace

import pytest
import torch

from kquant.sqg_e4m3 import sqg_xor_cheb_t12_bytes

from glm52_fresh_sqg.codec import (
    APPROVED_KQUANT_BACKEND_SHA256,
    APPROVED_KQUANT_STATUS_PORCELAIN,
    APPROVED_KQUANT_STATUS_SHA256,
    APPROVED_KQUANT_TRACKED_DIFF_SHA256,
    CODEBOOK_SCALE,
    FROZEN_KQUANT_REVISION,
    PRODUCTION_H13_CONSTRUCTION,
    LUT_SHA256,
    DenseHessian,
    KQuantRuntime,
    SharedResidualProfile,
    UniformSQGConfig,
    encode_uniform_sqg,
    prepare_dense_h_session,
    realize_shared_residual_profile,
    resume_dense_h_session,
    validate_gate_up_encode_smoke,
)
from glm52_fresh_sqg.manifest import (
    BF16TensorBinding,
    FORBIDDEN_MCG_READS,
    FrozenBitBudget,
    SQG_MARKER,
    SyntheticTensorBinding,
    build_run_manifest,
)
from glm52_fresh_sqg.permutation import (
    derive_glm_h2_reverse_permutation,
    draw_fresh_expert_permutation,
)
from glm52_fresh_sqg.reference import (
    decode_stored_fp16,
    pack_trellis_states,
    payload_sha256,
    reconstruct_trellis_states,
    tensor_sha256,
)


@dataclass
class FakeBackend:
    fallback: bool = False
    finalize_scale: float = 1.0
    calls: list[dict] = field(default_factory=list)

    def finalize_capture_H(self, H_data, quant_args, verbose=False):
        del verbose
        H_data["q_fallback"] = self.fallback
        H_data["finalized"] = True
        H_data["L"] = torch.eye(
            H_data["H"].shape[0], dtype=torch.float32, device=H_data["H"].device
        ) * self.finalize_scale
        signs = quant_args["input_signs"].to(
            device=H_data["H"].device, dtype=torch.float32
        ).reshape(-1, 1)
        return self.fallback, H_data["H"], H_data["L"], signs, None

    def quantize_qsrt(
        self,
        weight,
        H_data,
        quant_args,
        return_weight_q,
        progress_str=None,
        verbose=False,
        swap_to_device=None,
        save_reg=None,
    ):
        del progress_str, verbose, swap_to_device, save_reg
        assert return_weight_q is True
        bits = quant_args["K"]
        k, n = weight.shape
        edges = (
            torch.arange(256, device=weight.device, dtype=torch.int16)
            .remainder(1 << bits)
            .reshape(1, 1, 256)
            .expand(k // 16, n // 16, 256)
            .clone()
        )
        states = reconstruct_trellis_states(edges, bits)
        packed = quant_args["pack_trellis_fn"](states, quant_args)
        if "input_signs" in quant_args:
            suh = (
                quant_args["input_signs"].float()
                * quant_args["input_channel_scale_profile"].float()
                / (-CODEBOOK_SCALE)
                + 1e-10
            ).half()
        else:
            suh = torch.ones(k, dtype=torch.float16, device=weight.device)
        if "output_signs" in quant_args:
            svh = (
                quant_args["output_signs"].float()
                * quant_args["output_channel_scale_profile"].float()
                + 1e-10
            ).half()
        else:
            svh = torch.ones(n, dtype=torch.float16, device=weight.device)
        reconstructed = decode_stored_fp16(
            packed,
            suh,
            svh,
            bits=bits,
            codebook_e4m3=quant_args["sqg_e4m3_lut"],
        )
        if not isinstance(H_data.get("L"), torch.Tensor):
            H_data["L"] = torch.eye(
                H_data["H"].shape[0],
                dtype=torch.float32,
                device=H_data["H"].device,
            )
        H_data["finalized"] = True
        H_data["q_fallback"] = self.fallback
        self.calls.append(
            {
                "quant_args": dict(quant_args),
                "weight_data_ptr": weight.data_ptr(),
                "hessian_data_ptr": H_data["H"].data_ptr(),
            }
        )
        # Prove that the bridge gave the backend private mutable clones.
        weight.zero_()
        H_data["H"].zero_()
        quant_args.update(
            {
                "q_fallback": self.fallback,
                "g_scale": 1.125,
                "apply_out_scales": bool(quant_args.get("apply_out_scales")),
            }
        )
        return reconstructed, 0.125, {"trellis": packed, "suh": suh, "svh": svh}


def fake_runtime(backend: FakeBackend) -> KQuantRuntime:
    return KQuantRuntime(
        backend=backend,
        lut_bytes=sqg_xor_cheb_t12_bytes,
        pack_states=lambda states, args: pack_trellis_states(states, args["K"]),
        revision=FROZEN_KQUANT_REVISION,
        backend_sha256=APPROVED_KQUANT_BACKEND_SHA256,
        working_tree_dirty=True,
        tracked_diff_sha256=APPROVED_KQUANT_TRACKED_DIFF_SHA256,
        status_sha256=APPROVED_KQUANT_STATUS_SHA256,
        status_porcelain=APPROVED_KQUANT_STATUS_PORCELAIN,
        requires_cuda=False,
    )


def synthetic_fixture(seed: int = 31):
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn((128, 128), generator=generator)
    hessian = torch.eye(128, dtype=torch.float32)
    binding = SyntheticTensorBinding(
        tensor_name="synthetic.weight",
        tensor_sha256=tensor_sha256(source),
        fixture_id=f"fixture-{seed}",
    )
    dense_h = DenseHessian(
        hessian,
        evidence_id=f"dense-h-{seed}",
        construction="synthetic_spd_fixture",
        split_id="fit",
        routed_sample_count=512,
    )
    return source, hessian, binding, dense_h


def test_gate_up_encode_smoke_rejects_scale_ceiling_and_catastrophic_rmse() -> None:
    validate_gate_up_encode_smoke(global_scale=1.25, source_relative_rmse=0.14)
    with pytest.raises(RuntimeError, match="upper boundary"):
        validate_gate_up_encode_smoke(global_scale=2.05, source_relative_rmse=0.14)
    with pytest.raises(RuntimeError, match="catastrophically inaccurate"):
        validate_gate_up_encode_smoke(global_scale=1.25, source_relative_rmse=1.34)


@pytest.mark.parametrize("bits", (3, 4))
@pytest.mark.parametrize("side", ("input", "output"))
def test_fresh_uniform_encode_is_sqg_only_dense_h_and_profile_explicit(
    bits: int,
    side: str,
) -> None:
    source, hessian, binding, dense_h = synthetic_fixture(bits * 10 + len(side))
    source_before = source.clone()
    hessian_before = hessian.clone()
    stored = torch.linspace(0.5, 1.5, 128, dtype=torch.float16)
    stored[1::2] *= -1
    profile = SharedResidualProfile.from_stored_vector(
        stored,
        side=side,
        profile_id=f"fresh-{side}-profile",
        derivation="synthetic caller-owned layer profile",
    )
    backend = FakeBackend()
    result = encode_uniform_sqg(
        source,
        dense_h,
        UniformSQGConfig(
            tensor_id=f"tensor-k{bits}-{side}",
            bits=bits,
            matrix_role="synthetic_exl",
            transform_seed=1000 + bits,
            output_sign_seed=2000 + bits,
            source_binding=binding,
            shared_residual_profile=profile,
            device="cpu",
            production=False,
        ),
        runtime=fake_runtime(backend),
    )

    assert torch.equal(source, source_before)
    assert torch.equal(hessian, hessian_before)
    assert backend.calls[0]["weight_data_ptr"] != source.data_ptr()
    shared = result.suh if side == "input" else result.svh
    assert torch.equal(shared, stored)
    assert result.sqg.dtype == torch.int32
    assert result.sqg.item() == SQG_MARKER
    assert set(result.named_tensors("model.expert.weight")) == {
        "model.expert.weight.trellis",
        "model.expert.weight.suh",
        "model.expert.weight.svh",
        "model.expert.weight.sqg",
    }
    manifest = result.manifest
    assert manifest["bits"] == bits
    assert manifest["codebook_lut_sha256"] == LUT_SHA256[bits]
    assert manifest["dense_h"]["block_ldlq"] is True
    assert manifest["dense_h"]["fallback"] is False
    assert manifest["decoded_closure"]["passed"] is True
    assert manifest["forbidden_input_reads"] == list(FORBIDDEN_MCG_READS)
    args = backend.calls[0]["quant_args"]
    assert args["tailbite_context"] == 128
    assert payload_sha256(args["sqg_e4m3_lut"]) == LUT_SHA256[bits]
    assert not {"mcg", "mul1", "shared_input_scales_key"}.intersection(args)
    if side == "input":
        assert args["g_scale_into_sv"] is True
    else:
        assert args["g_scale_into_sv"] is False


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability(0) != (12, 0),
    reason="production CUDA FP16 boundary regression requires SM120",
)
def test_shared_profile_uses_exact_assigned_cuda_fp16_realization() -> None:
    profile = SharedResidualProfile(
        side="input",
        signs=torch.tensor([1.0]),
        channel_scales=torch.tensor([0.8262054324150085]),
        profile_id="sm120-boundary-vector",
        derivation="exact production rounding boundary regression",
    )
    assert profile.expected_stored_fp16().item() == -0.66455078125

    realized = realize_shared_residual_profile(profile, "cuda:0")

    assert realized is not profile
    assert realized.expected_stored_fp16().item() == -0.6640625
    assert torch.equal(realized.signs, profile.signs)
    assert torch.equal(realized.channel_scales, profile.channel_scales)
    evidence = realized.manifest()["stored_fp16_realization"]
    assert evidence["mismatch_count"] == 1
    assert evidence["backend_expression_unchanged"] is True
    assert evidence["backend_output_overwritten"] is False
    assert realize_shared_residual_profile(realized, "cuda:0") is realized


def test_dense_h_fallback_is_a_hard_failure() -> None:
    source, _, binding, dense_h = synthetic_fixture(88)
    with pytest.raises(RuntimeError, match="forbidden non-dense-H fallback"):
        encode_uniform_sqg(
            source,
            dense_h,
            UniformSQGConfig(
                tensor_id="fallback-forbidden",
                bits=3,
                matrix_role="synthetic_exl",
                transform_seed=1,
                output_sign_seed=2,
                source_binding=binding,
                device="cpu",
                production=False,
            ),
            runtime=fake_runtime(FakeBackend(fallback=True)),
        )


def test_hessian_that_would_fallback_is_rejected_before_backend() -> None:
    source, _, binding, _ = synthetic_fixture(89)
    dense_h = DenseHessian(
        torch.zeros((128, 128)),
        evidence_id="zero-h",
        construction="invalid",
        split_id="fit",
    )
    backend = FakeBackend()
    with pytest.raises(ValueError, match="fallback path"):
        encode_uniform_sqg(
            source,
            dense_h,
            UniformSQGConfig(
                tensor_id="zero-h",
                bits=4,
                matrix_role="synthetic_exl",
                transform_seed=3,
                output_sign_seed=4,
                source_binding=binding,
                device="cpu",
                production=False,
            ),
            runtime=fake_runtime(backend),
        )
    assert not backend.calls


def test_production_binding_applies_only_fresh_gate_row_permutation() -> None:
    generator = torch.Generator().manual_seed(191)
    source = torch.randn((2048, 128), generator=generator).to(torch.bfloat16)
    source_before = source.clone()
    binding = BF16TensorBinding(
        repository_id="zai-org/GLM-5.2",
        revision="1" * 40,
        shard_name="model-00001-of-00096.safetensors",
        shard_sha256="2" * 64,
        tensor_name="model.layers.6.mlp.experts.0.gate_proj.weight",
        tensor_sha256=payload_sha256(source),
    )
    permutation = derive_glm_h2_reverse_permutation(
        torch.randn((7, 2048), generator=generator),
        torch.rand((7,), generator=generator),
        scope="layer-006/expert-000",
        evidence_id="fresh-post-silu-l006-e000",
        split_id="fit",
    )
    dense_h = DenseHessian(
        torch.eye(128),
        evidence_id="h13-l006-e000",
        construction=PRODUCTION_H13_CONSTRUCTION,
        split_id="fit",
        routed_sample_count=2048,
    )
    config = UniformSQGConfig(
        tensor_id=binding.tensor_name.removesuffix(".weight"),
        bits=3,
        matrix_role="gate",
        transform_seed=7001,
        output_sign_seed=7002,
        source_binding=binding,
        physical_permutation=permutation,
        device="cpu",
        production=True,
    )
    session = prepare_dense_h_session(dense_h, config)
    result = encode_uniform_sqg(
        source,
        dense_h,
        config,
        runtime=fake_runtime(FakeBackend()),
        dense_h_session=session,
    )

    assert torch.equal(source, source_before)
    transform = result.manifest["transform"]
    assert transform["operation"] == "fresh_P_rows_then_hf_to_exl_transpose"
    assert transform["physical_permutation"]["new_to_old_sha256"] == permutation.sha256
    assert transform["physical_permutation"]["legacy_permutation_input"] is False
    assert result.manifest["source"]["kind"] == "official_bf16"
    assert result.manifest["source"]["shard_sha256"] == "2" * 64
    assert result.manifest["dense_h"]["session"]["use_ordinal"] == 1

    with pytest.raises(ValueError, match="only the fit split"):
        encode_uniform_sqg(
            source,
            replace(dense_h, split_id="selection"),
            config,
            runtime=fake_runtime(FakeBackend()),
        )
    with pytest.raises(ValueError, match="dense-H construction differs"):
        encode_uniform_sqg(
            source,
            replace(dense_h, construction="fit_identity_or_global_fallback"),
            config,
            runtime=fake_runtime(FakeBackend()),
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("revision", "0" * 40),
        ("backend_sha256", "0" * 64),
        ("tracked_diff_sha256", "0" * 64),
        ("status_sha256", "0" * 64),
        ("status_porcelain", ""),
        ("working_tree_dirty", False),
    ),
)
def test_production_encode_rejects_injected_runtime_provenance_drift(
    field_name: str,
    bad_value: object,
) -> None:
    generator = torch.Generator().manual_seed(197)
    source = torch.randn((2048, 128), generator=generator).to(torch.bfloat16)
    binding = BF16TensorBinding(
        repository_id="zai-org/GLM-5.2",
        revision="1" * 40,
        shard_name="model-00001-of-00096.safetensors",
        shard_sha256="2" * 64,
        tensor_name="model.layers.6.mlp.experts.0.gate_proj.weight",
        tensor_sha256=payload_sha256(source),
    )
    permutation = derive_glm_h2_reverse_permutation(
        torch.randn((7, 2048), generator=generator),
        torch.rand((7,), generator=generator),
        scope="layer-006/expert-000",
        evidence_id="fresh-post-silu-l006-e000",
        split_id="fit",
    )
    dense_h = DenseHessian(
        torch.eye(128),
        evidence_id="h13-l006-e000",
        construction=PRODUCTION_H13_CONSTRUCTION,
        split_id="fit",
        routed_sample_count=2048,
    )
    config = UniformSQGConfig(
        tensor_id=binding.tensor_name.removesuffix(".weight"),
        bits=3,
        matrix_role="gate",
        transform_seed=7001,
        output_sign_seed=7002,
        source_binding=binding,
        physical_permutation=permutation,
        device="cpu",
        production=True,
    )
    backend = FakeBackend()
    runtime = replace(fake_runtime(backend), **{field_name: bad_value})

    with pytest.raises(RuntimeError, match="runtime provenance"):
        encode_uniform_sqg(source, dense_h, config, runtime=runtime)
    assert not backend.calls


def test_nonproduction_encode_allows_unsealed_synthetic_runtime() -> None:
    source, _, binding, dense_h = synthetic_fixture(198)
    backend = FakeBackend()
    runtime = replace(
        fake_runtime(backend),
        revision="synthetic-test-runtime",
        backend_sha256="0" * 64,
        working_tree_dirty=False,
        tracked_diff_sha256="0" * 64,
        status_sha256="0" * 64,
        status_porcelain="",
    )
    result = encode_uniform_sqg(
        source,
        dense_h,
        UniformSQGConfig(
            tensor_id="synthetic-unsealed-runtime",
            bits=3,
            matrix_role="synthetic_exl",
            transform_seed=1,
            output_sign_seed=2,
            source_binding=binding,
            device="cpu",
            production=False,
        ),
        runtime=runtime,
    )
    assert result.manifest["encoder"]["kquant_revision"] == "synthetic-test-runtime"


def test_caller_owned_dense_h_session_reuses_one_state_and_binds_input_profile() -> None:
    source, _, binding, dense_h = synthetic_fixture(211)
    stored = torch.ones(128, dtype=torch.float16)
    profile = SharedResidualProfile.from_stored_vector(
        stored,
        side="input",
        profile_id="shared-h13-profile",
        derivation="fresh layer profile",
    )
    first_config = UniformSQGConfig(
        tensor_id="gate-e0",
        bits=3,
        matrix_role="synthetic_exl",
        transform_seed=90,
        output_sign_seed=91,
        source_binding=binding,
        shared_residual_profile=profile,
        device="cpu",
        production=False,
    )
    second_config = UniformSQGConfig(
        tensor_id="up-e0",
        bits=4,
        matrix_role="synthetic_exl",
        transform_seed=90,
        output_sign_seed=92,
        source_binding=binding,
        shared_residual_profile=profile,
        device="cpu",
        production=False,
    )
    session = prepare_dense_h_session(dense_h, first_config)
    backend = FakeBackend()
    runtime = fake_runtime(backend)
    first = encode_uniform_sqg(
        source,
        dense_h,
        first_config,
        runtime=runtime,
        dense_h_session=session,
    )
    session.h_data["finalized"] = True  # what the real KQuant first call sets
    second = encode_uniform_sqg(
        source,
        dense_h,
        second_config,
        runtime=runtime,
        dense_h_session=session,
    )

    assert backend.calls[0]["hessian_data_ptr"] == backend.calls[1]["hessian_data_ptr"]
    assert first.manifest["dense_h"]["session"]["use_ordinal"] == 1
    assert second.manifest["dense_h"]["session"]["use_ordinal"] == 2
    assert second.manifest["dense_h"]["session"]["reused_finalized_block_ldl"] is True
    assert session.use_count == 2


def test_dense_h_session_resume_restores_exact_completed_prefix_ordinal() -> None:
    source, _, binding, dense_h = synthetic_fixture(212)
    profile = SharedResidualProfile.from_stored_vector(
        torch.ones(128, dtype=torch.float16),
        side="input",
        profile_id="resume-shared-h13-profile",
        derivation="fresh layer profile",
    )

    def config(tensor_id: str, bits: int, seed: int) -> UniformSQGConfig:
        return UniformSQGConfig(
            tensor_id=tensor_id,
            bits=bits,
            matrix_role="synthetic_exl",
            transform_seed=90,
            output_sign_seed=seed,
            source_binding=binding,
            shared_residual_profile=profile,
            device="cpu",
            production=False,
        )

    runtime = fake_runtime(FakeBackend())
    first_config = config("gate-e0", 3, 91)
    original = prepare_dense_h_session(dense_h, first_config)
    first = encode_uniform_sqg(
        source,
        dense_h,
        first_config,
        runtime=runtime,
        dense_h_session=original,
    )
    # The synthetic quantizer does not factor H; reproduce the state the real
    # backend leaves after its first encode so the second record is a reuse.
    original.h_data["finalized"] = True
    original.h_data["q_fallback"] = False
    second = encode_uniform_sqg(
        source,
        dense_h,
        config("up-e0", 4, 92),
        runtime=runtime,
        dense_h_session=original,
    )
    first_manifest = copy.deepcopy(first.manifest)
    first_manifest["matrix_role"] = "gate"
    second_manifest = copy.deepcopy(second.manifest)
    second_manifest["matrix_role"] = "up"

    resumed = prepare_dense_h_session(dense_h, first_config)
    evidence = resume_dense_h_session(
        resumed,
        dense_h,
        first_config,
        runtime,
        prior_tensor_manifests=[first_manifest, second_manifest],
        expected_tensor_ids=["gate-e0", "up-e0"],
    )
    third = encode_uniform_sqg(
        source,
        dense_h,
        config("gate-e1", 3, 93),
        runtime=runtime,
        dense_h_session=resumed,
    )

    assert evidence["restored_use_count"] == 2
    assert evidence["weight_encode_used_for_restore"] is False
    assert third.manifest["dense_h"]["session"]["use_ordinal"] == 3
    assert third.manifest["dense_h"]["session"]["reused_finalized_block_ldl"] is True

    with pytest.raises(ValueError, match="tensor IDs differ"):
        resume_dense_h_session(
            prepare_dense_h_session(dense_h, first_config),
            dense_h,
            first_config,
            runtime,
            prior_tensor_manifests=[first_manifest, second_manifest],
            expected_tensor_ids=["up-e0", "gate-e0"],
        )

    with pytest.raises(RuntimeError, match="fingerprint differs"):
        resume_dense_h_session(
            prepare_dense_h_session(dense_h, first_config),
            dense_h,
            first_config,
            fake_runtime(FakeBackend(finalize_scale=2.0)),
            prior_tensor_manifests=[first_manifest, second_manifest],
            expected_tensor_ids=["gate-e0", "up-e0"],
        )


def test_dense_h_session_resume_rejects_noncontiguous_prefix() -> None:
    source, _, binding, dense_h = synthetic_fixture(213)
    profile = SharedResidualProfile.from_stored_vector(
        torch.ones(128, dtype=torch.float16),
        side="input",
        profile_id="resume-prefix-profile",
        derivation="fresh layer profile",
    )
    config = UniformSQGConfig(
        tensor_id="gate-e0",
        bits=3,
        matrix_role="synthetic_exl",
        transform_seed=90,
        output_sign_seed=91,
        source_binding=binding,
        shared_residual_profile=profile,
        device="cpu",
        production=False,
    )
    runtime = fake_runtime(FakeBackend())
    original = prepare_dense_h_session(dense_h, config)
    item = encode_uniform_sqg(
        source,
        dense_h,
        config,
        runtime=runtime,
        dense_h_session=original,
    )
    broken = copy.deepcopy(item.manifest)
    broken["matrix_role"] = "gate"
    broken["dense_h"]["session"]["use_ordinal"] = 2
    resumed = prepare_dense_h_session(dense_h, config)
    with pytest.raises(ValueError, match="ordinals are not contiguous"):
        resume_dense_h_session(
            resumed,
            dense_h,
            config,
            runtime,
            prior_tensor_manifests=[broken],
        )


def test_run_manifest_seals_forbidden_reads_and_frozen_bit_budget() -> None:
    manifests = []
    for bits in (3, 4):
        source, _, binding, dense_h = synthetic_fixture(300 + bits)
        result = encode_uniform_sqg(
            source,
            dense_h,
            UniformSQGConfig(
                tensor_id=f"t{bits}",
                bits=bits,
                matrix_role="synthetic_exl",
                transform_seed=bits,
                output_sign_seed=bits + 10,
                source_binding=binding,
                device="cpu",
                production=False,
            ),
            runtime=fake_runtime(FakeBackend()),
        )
        manifests.append(result.manifest)
    generator = torch.Generator().manual_seed(303)
    permutation = derive_glm_h2_reverse_permutation(
        torch.randn((5, 2048), generator=generator),
        torch.rand((5,), generator=generator),
        scope="test-run",
        evidence_id="synthetic-post-silu",
        split_id="fit",
    )
    run = build_run_manifest(
        run_id="synthetic-k3-k4",
        bit_budget=FrozenBitBudget(
            bit_map={"t3": 3, "t4": 4},
            expected_k3=1,
            expected_k4=1,
        ),
        tensor_manifests=manifests,
        fresh_permutations=[permutation.manifest()],
        production=False,
    )

    assert run["forbidden_reads"]["items"] == list(FORBIDDEN_MCG_READS)
    assert run["forbidden_reads"]["observed"] == []
    assert run["codec"]["fallback_allowed"] is False
    assert run["frozen_bit_budget"]["controlled_input"] == "bit_map_only"
    assert run["frozen_bit_budget"]["other_rate_count"] == 0


def test_production_mode_rejects_synthetic_source_binding() -> None:
    source, _, binding, _ = synthetic_fixture(401)
    with pytest.raises(TypeError, match="official BF16"):
        UniformSQGConfig(
            tensor_id="not-production",
            bits=3,
            matrix_role="gate",
            transform_seed=1,
            output_sign_seed=2,
            source_binding=binding,
            physical_permutation=draw_fresh_expert_permutation(
                128, seed=1, scope="reject"
            ),
            production=True,
        )


def test_production_run_requires_one_permutation_across_coupled_projection_axes() -> None:
    source, _, binding, dense_h = synthetic_fixture(501)
    base = encode_uniform_sqg(
        source,
        dense_h,
        UniformSQGConfig(
            tensor_id="base",
            bits=3,
            matrix_role="synthetic_exl",
            transform_seed=1,
            output_sign_seed=2,
            source_binding=binding,
            device="cpu",
            production=False,
        ),
        runtime=fake_runtime(FakeBackend()),
    ).manifest
    generator = torch.Generator().manual_seed(502)
    permutation = derive_glm_h2_reverse_permutation(
        torch.randn((5, 2048), generator=generator),
        torch.rand((5,), generator=generator),
        scope="layer-006/expert-000",
        evidence_id="fresh-fit-middle",
        split_id="fit",
    )
    manifests = []
    bit_map = {}
    for projection, role, bits in (
        ("gate_proj", "gate", 3),
        ("up_proj", "up", 4),
        ("down_proj", "down", 3),
    ):
        item = copy.deepcopy(base)
        tensor_id = f"model.layers.6.mlp.experts.0.{projection}"
        item["tensor_id"] = tensor_id
        item["matrix_role"] = role
        item["bits"] = bits
        item["codebook_lut_sha256"] = LUT_SHA256[bits]
        item["source"] = {
            "kind": "official_bf16",
            "repository_id": "zai-org/GLM-5.2",
            "revision": "1" * 40,
            "shard_name": "model-00001-of-00282.safetensors",
            "shard_sha256": "2" * 64,
            "tensor_name": f"{tensor_id}.weight",
            "tensor_payload_sha256": "3" * 64,
            "dtype": "bfloat16",
            "mcg_source": False,
        }
        item["transform"]["physical_permutation"] = permutation.manifest()
        manifests.append(item)
        bit_map[tensor_id] = bits
    run = build_run_manifest(
        run_id="coupled-permutation",
        bit_budget=FrozenBitBudget(bit_map, expected_k3=2, expected_k4=1),
        tensor_manifests=manifests,
        fresh_permutations=[permutation.manifest()],
    )
    assert len(run["fresh_physical_permutations"]) == 1

    broken = copy.deepcopy(manifests)
    broken[-1]["transform"]["physical_permutation"]["new_to_old_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not share one"):
        build_run_manifest(
            run_id="broken-coupled-permutation",
            bit_budget=FrozenBitBudget(bit_map, expected_k3=2, expected_k4=1),
            tensor_manifests=broken,
            fresh_permutations=[permutation.manifest()],
        )
