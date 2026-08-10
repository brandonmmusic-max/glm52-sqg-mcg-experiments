from __future__ import annotations

import json
import struct
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from r7_encoder.assemble import _bind_assembly_transaction
from r7_encoder.constants import TensorId
from r7_encoder.convert_v2_to_v1 import (
    _assert_scalar_bit_compat,
    _bind_conversion_transaction,
)
from r7_encoder.determinism import (
    atomic_write_bytes,
    atomic_write_json,
    configure_deterministic_environment,
    derive_seed,
    read_json,
    sha256_file,
)
from r7_encoder import expert_cache, row_cache
from r7_encoder.expert_cache import load_cached_expert, write_cached_expert
from r7_encoder.inventory import (
    build_checkpoint_inventory,
    load_checkpoint_inventory,
    verify_checkpoint_inventory,
)
from r7_encoder.glm52_backend import GLM52Backend, GLM52Runtime
from r7_encoder.permutation import (
    descending_diag_permutation,
    ldlq_visit_old_indices,
)
from r7_encoder.routing import RoutedMassAccumulator
from r7_encoder.safetensors_io import (
    SafeTensorReader,
    TensorEntry,
    torch_tensor_entry,
    write_safetensors_atomic,
)
from r7_encoder.search import mass_stratified_experts
from r7_encoder.search import SearchRunner
from r7_encoder.sensitivity import FixedPointController
from r7_encoder.state import Journal, StageSeal, StateStore
from r7_encoder.transformers_runtime import TransformersSequentialRuntime
from r7_encoder.trellis import (
    CodecConfig,
    Exl3TrellisCodec,
    cuda_only_block_ldl,
    load_sealed_extension,
)
from r7_encoder.types import EncodedTensor, LayerAllocation, RoutedBatch, StateShard
from r7_encoder.walk import SequentialWalk


class _SyntheticRuntime(GLM52Runtime):
    def __init__(self):
        self.metadata_seen = None

    @property
    def fingerprint(self) -> str:
        return "synthetic-runtime-sha256"

    def prepare_corpus_plan(self, *, corpus):
        return {
            "corpus_plan_sha256": "b" * 64,
            "corpus_plan_artifact_sha256": "c" * 64,
            "expected_shards": {"0000": 1},
        }

    def initialize_carried_state(self, **kwargs):
        return ()

    def prepare_moe_input(self, *, layer, hidden, attention_metadata):
        return hidden.to(torch.bfloat16)

    def route_exact(self, *, layer, moe_hidden, attention_metadata):
        assert layer != 78
        self.metadata_seen = attention_metadata
        ids, weights = _routes(moe_hidden.shape[0])
        return RoutedBatch(ids, weights, 2.5)

    def capture_arithmetic_audit(self, *, layer):
        return {
            "schema": "r7-carried-layer-arithmetic-v1",
            "layer": layer,
            "runtime_fingerprint": self.fingerprint,
            "attention_implementation": "eager",
            "dispatch_audit_sha256": "d" * 64,
            "tensor_records_sha256": "a" * 64,
            "tensor_count": 1,
            "passed": True,
        }

    def install_encoded_expert(self, *, layer, expert, encoded):
        packed = {
            key: str(value["packed_sha256"]) for key, value in sorted(encoded.items())
        }
        reconstructed = {
            key: str(value["reconstruction_sha256"])
            for key, value in sorted(encoded.items())
        }
        return {
            "schema": "r7-packed-install-record-v2",
            "layer": layer,
            "expert": expert,
            "runtime_fingerprint": self.fingerprint,
            "activation_dtype": "BF16",
            "packed_decoded": True,
            "packed_sha256": packed,
            "reconstruction_sha256": reconstructed,
            "installed_shape_kn": {
                key: ([6144, 2048] if not key.endswith("down_proj") else [2048, 6144])
                for key in sorted(encoded)
            },
            "passed": True,
        }

    def audit_installed_layer(self, *, layer):
        return {
            "schema": "r7-official-installed-layer-audit-v1",
            "layer": layer,
            "runtime_fingerprint": self.fingerprint,
            "activation_dtype": "BF16",
            "sample_rows": 32,
            "sample_seed": derive_seed(layer, "official-installed-layer-audit-v1"),
            "sample_input_sha256": "1" * 64,
            "top_k": 8,
            "topk_indices_sha256": "2" * 64,
            "topk_weights_sha256": "3" * 64,
            "routed_scaling_factor": "2.5",
            "unique_experts_per_token": True,
            "shared_expert_hits": True,
            "official_module": (
                "transformers.models.glm_moe_dsa.modeling_glm_moe_dsa.GlmMoeDsaNaiveMoe"
            ),
            "experts_implementation": "eager",
            "dispatch_audit_sha256": "4" * 64,
            "official_first_output_sha256": "5" * 64,
            "official_second_output_sha256": "5" * 64,
            "reference_output_sha256": "6" * 64,
            "official_repeat_exact": True,
            "max_abs_error": "0",
            "relative_l2_error": "0",
            "max_abs_tolerance": "0.03125",
            "relative_l2_tolerance": "0.001953125",
            "passed": True,
        }

    def restore_encoded_layer(self, **kwargs):
        return None

    def forward_installed_layer(self, *, layer, **kwargs):
        assert layer != 78
        return ()


def _routes(rows: int):
    ids = torch.stack(
        [torch.arange(offset, offset + 8) % 256 for offset in range(rows)]
    ).to(torch.int64)
    weights = torch.full((rows, 8), 2.5 / 8, dtype=torch.float32)
    return ids, weights


def test_routed_mass_is_batch_invariant_and_rejects_duplicates():
    ids, weights = _routes(17)
    whole = RoutedMassAccumulator(3)
    whole.add(ids, weights, 2.5)
    split = RoutedMassAccumulator(3)
    split.add(ids[:5], weights[:5], 2.5)
    split.add(ids[5:11], weights[5:11], 2.5)
    split.add(ids[11:], weights[11:], 2.5)
    assert whole.finish() == split.finish()

    duplicate = ids[:1].clone()
    duplicate[0, 1] = duplicate[0, 0]
    with pytest.raises(ValueError, match="duplicate"):
        RoutedMassAccumulator(3).add(duplicate, weights[:1], 2.5)


def test_routed_mass_expected_spelling_is_partition_invariant():
    ids, weights = _routes(13)
    expected = float(torch.tensor(1.3, dtype=torch.float32).item())
    weights = weights * (expected / 2.5)
    whole = RoutedMassAccumulator(3)
    whole.add(ids, weights, expected)
    split = RoutedMassAccumulator(3)
    split.add(ids[:1], weights[:1], expected)
    split.add(ids[1:8], weights[1:8], expected)
    split.add(ids[8:], weights[8:], expected)
    assert whole.finish() == split.finish()


def test_mass_stratification_preserves_hot_and_cold():
    masses = [float(index + 1) for index in range(256)]
    selected = mass_stratified_experts(masses, 16)
    assert len(selected) == 16
    assert 255 in selected
    assert 0 in selected


def test_ldlq_visit_is_descending_not_stored_order():
    diagonal = [1.0, 9.0, 3.0, 7.0]
    permutation = descending_diag_permutation(diagonal)
    visit = ldlq_visit_old_indices(permutation)
    assert [diagonal[index] for index in visit] == [9.0, 7.0, 3.0, 1.0]


def _allocation(bits: dict[str, int], iteration: int) -> LayerAllocation:
    return LayerAllocation(3, bits, ("1",) * 256, 0, 1, 0, iteration, "probe")


def test_fixed_point_compares_full_map_and_detects_cycle():
    controller = FixedPointController(4)
    assert not controller.observe(_allocation({"gate": 4, "down": 3}, 0))
    assert not controller.observe(_allocation({"gate": 4, "down": 4}, 1))
    with pytest.raises(RuntimeError, match="cycle"):
        controller.observe(_allocation({"gate": 4, "down": 3}, 2))


def test_fp16_vector_boundary_rejects_underflow_and_overflow():
    codec = Exl3TrellisCodec(CodecConfig(device="cpu"))
    with pytest.raises(ValueError, match="underflows"):
        codec._validate_vectors([1e-8] * 16, [1.0] * 16, 16, 16)
    with pytest.raises(ValueError, match="overflows"):
        codec._validate_vectors([1e10] * 16, [1.0] * 16, 16, 16)


def test_deterministic_environment_is_exact_and_conflicts_fail(monkeypatch):
    for name in (
        "CUBLAS_WORKSPACE_CONFIG",
        "USE_HUB_KERNELS",
        "TOKENIZERS_PARALLELISM",
    ):
        monkeypatch.delenv(name, raising=False)
    sealed = configure_deterministic_environment()
    assert sealed["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert sealed["USE_HUB_KERNELS"] == "0"
    assert sealed["experts_implementation"] == "eager"
    monkeypatch.setenv("USE_HUB_KERNELS", "1")
    with pytest.raises(RuntimeError, match="deterministic environment mismatch"):
        configure_deterministic_environment()


def test_cuda_ldl_oom_fails_without_calling_an_alternate_factorizer(monkeypatch):
    calls = []

    class FakeCudaMatrix:
        device = SimpleNamespace(type="cuda")
        shape = (16, 16)

    def oom(_):
        calls.append("cuda-cholesky")
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(torch.linalg, "cholesky", oom)
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        cuda_only_block_ldl(FakeCudaMatrix(), 16, {"sigma_reg": 0.025})
    assert calls == ["cuda-cholesky"]


def test_sealed_extension_rejects_ambient_path_and_hash_drift(tmp_path: Path):
    first = tmp_path / "exllamav3_ext.py"
    second_dir = tmp_path / "other"
    second_dir.mkdir()
    second = second_dir / "exllamav3_ext.py"
    atomic_write_bytes(first, b"VALUE = 1\n")
    atomic_write_bytes(second, b"VALUE = 1\n")
    incumbent = sys.modules.pop("exllamav3_ext", None)
    try:
        loaded = load_sealed_extension(first, expected_sha256=sha256_file(first))
        assert loaded.VALUE == 1
        with pytest.raises(RuntimeError, match="ambient exllamav3_ext"):
            load_sealed_extension(second, expected_sha256=sha256_file(second))
        with pytest.raises(ValueError, match="sealed binary"):
            load_sealed_extension(first, expected_sha256="0" * 64)
    finally:
        sys.modules.pop("exllamav3_ext", None)
        if incumbent is not None:
            sys.modules["exllamav3_ext"] = incumbent


def test_independent_routed_reference_matches_official_eager_formula():
    import torch.nn.functional as functional

    generator = torch.Generator().manual_seed(17)
    hidden = torch.randn((6, 4), generator=generator).bfloat16()
    installed = {}
    gate_up = []
    down = []
    for expert in range(4):
        gate = torch.randn((4, 3), generator=generator).bfloat16()
        up = torch.randn((4, 3), generator=generator).bfloat16()
        down_kn = torch.randn((3, 4), generator=generator).bfloat16()
        installed[expert] = {
            "gate_proj": gate,
            "up_proj": up,
            "down_proj": down_kn,
        }
        gate_up.append(torch.cat((gate.T, up.T), dim=0))
        down.append(down_kn.T)
    gate_up = torch.stack(gate_up)
    down = torch.stack(down)
    indices = torch.tensor(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=torch.int64
    )
    weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.2, 0.8], [0.5, 0.5], [0.9, 0.1], [0.4, 0.6]],
        dtype=torch.float32,
    )
    official = torch.zeros_like(hidden)
    mask = torch.nn.functional.one_hot(indices, num_classes=4).permute(2, 1, 0)
    for expert in range(4):
        topk_position, token_index = torch.where(mask[expert])
        current = hidden[token_index]
        gate, up = functional.linear(current, gate_up[expert]).chunk(2, dim=-1)
        current = functional.linear(functional.silu(gate) * up, down[expert])
        current = current * weights[token_index, topk_position, None]
        official.index_add_(0, token_index, current.to(official.dtype))
    reference = TransformersSequentialRuntime._reference_routed_experts(
        hidden, indices, weights, installed
    )
    assert torch.equal(reference, official)


def test_safetensors_reader_rejects_overlap_and_trailing_bytes(tmp_path: Path):
    trailing = tmp_path / "trailing.safetensors"
    write_safetensors_atomic(trailing, [TensorEntry("x", "U8", (1,), b"x")])
    with trailing.open("ab") as handle:
        handle.write(b"stale")
    with pytest.raises(ValueError, match="trailing"):
        SafeTensorReader(trailing)

    overlap = tmp_path / "overlap.safetensors"
    header = {
        "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
        "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]},
    }
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    atomic_write_bytes(overlap, struct.pack("<Q", len(raw)) + raw + b"abc")
    with pytest.raises(ValueError, match="overlap"):
        SafeTensorReader(overlap)


def test_state_partial_resume_and_publication_adoption(tmp_path: Path):
    store = StateStore(tmp_path / "states")
    plan_sha256 = "b" * 64
    domain = {"0000": 1, "0001": 2}
    first = store.begin_transition(
        3, corpus_plan_sha256=plan_sha256, expected_shards=domain
    )
    h0, m0 = first.temporary / "h0", first.temporary / "m0"
    atomic_write_bytes(h0, b"h0")
    atomic_write_json(
        m0,
        {"corpus_plan_sha256": plan_sha256, "shard_id": "0000", "tokens": 1},
    )
    first.add_existing_shard(
        shard_id="0000", hidden_path=h0, metadata_path=m0, tokens=1, hidden_size=6144
    )

    with pytest.raises(ValueError, match="corpus-plan domain drift"):
        store.begin_transition(
            3,
            corpus_plan_sha256="c" * 64,
            expected_shards=domain,
        )

    resumed = store.begin_transition(
        3, corpus_plan_sha256=plan_sha256, expected_shards=domain
    )
    assert resumed.completed_shard_ids == frozenset({"0000"})
    h1, m1 = resumed.temporary / "h1", resumed.temporary / "m1"
    atomic_write_bytes(h1, b"h1")
    atomic_write_json(
        m1,
        {"corpus_plan_sha256": plan_sha256, "shard_id": "0001", "tokens": 2},
    )
    resumed.add_existing_shard(
        shard_id="0001", hidden_path=h1, metadata_path=m1, tokens=2, hidden_size=6144
    )
    digest = resumed.commit(predecessor_sha256="p", backend_fingerprint="b")
    assert (
        store.adopt_existing(3, predecessor_sha256="p", backend_fingerprint="b")
        == digest
    )


def test_successor_state_requires_a_bound_official_repeat_oracle(tmp_path: Path):
    store = StateStore(tmp_path / "states")
    transition = store.begin_transition(
        4, corpus_plan_sha256="b" * 64, expected_shards={"0000": 1}
    )
    hidden = transition.temporary / "hidden"
    metadata = transition.temporary / "metadata.json"
    atomic_write_bytes(hidden, b"hidden")
    atomic_write_json(
        metadata,
        {"corpus_plan_sha256": "b" * 64, "shard_id": "0000", "tokens": 1},
    )
    transition.add_existing_shard(
        shard_id="0000",
        hidden_path=hidden,
        metadata_path=metadata,
        tokens=1,
        hidden_size=6144,
    )
    with pytest.raises(ValueError, match="official repeat oracle"):
        transition.commit(predecessor_sha256="p", backend_fingerprint="b")


def test_checkpoint_inventory_detects_payload_drift(tmp_path: Path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    shard = checkpoint / "model-00001-of-00001.safetensors"
    write_safetensors_atomic(shard, [TensorEntry("x", "U8", (4,), b"data")])
    atomic_write_json(
        checkpoint / "model.safetensors.index.json",
        {"weight_map": {"x": shard.name}},
    )
    atomic_write_json(checkpoint / "config.json", {"model_type": "synthetic"})
    inventory_path = tmp_path / "inventory.json"
    built = build_checkpoint_inventory(checkpoint, inventory_path, role="carrier")
    loaded = load_checkpoint_inventory(inventory_path, role="carrier")
    assert built["inventory_sha256"] == loaded["inventory_sha256"]
    verify_checkpoint_inventory(checkpoint, loaded)
    with shard.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"X")
    with pytest.raises(ValueError, match="changed"):
        verify_checkpoint_inventory(checkpoint, loaded)


def test_checkpoint_inventory_seals_tokenizer_assets_and_stays_outside_model(
    tmp_path: Path,
):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    shard = checkpoint / "model.safetensors"
    write_safetensors_atomic(shard, [TensorEntry("x", "U8", (1,), b"x")])
    atomic_write_json(
        checkpoint / "model.safetensors.index.json",
        {"weight_map": {"x": shard.name}},
    )
    atomic_write_json(checkpoint / "config.json", {"model_type": "synthetic"})
    atomic_write_json(checkpoint / "tokenizer.json", {"version": "sealed"})
    with pytest.raises(ValueError, match="outside read-only models"):
        build_checkpoint_inventory(
            checkpoint, checkpoint / "INVENTORY.json", role="carrier"
        )
    inventory = build_checkpoint_inventory(
        checkpoint, tmp_path / "inventory.json", role="carrier"
    )
    assert "tokenizer.json" in inventory["auxiliary_files_sha256"]
    atomic_write_json(checkpoint / "tokenizer.json", {"version": "mutated"})
    with pytest.raises(ValueError, match="auxiliary/tokenizer assets"):
        verify_checkpoint_inventory(checkpoint, inventory)


def test_exact_35_cannot_claim_unmodified_scalar_r13():
    bit_map = {f"tensor-{index}": 3 + (index % 2) for index in range(768)}
    assert sum(bit_map.values()) == 2688
    with pytest.raises(ValueError, match="cannot be represented"):
        _assert_scalar_bit_compat(bit_map)
    with pytest.raises(ValueError, match="valid 768-entry"):
        _assert_scalar_bit_compat({f"tensor-{index}": 4 for index in range(768)})


def test_concrete_backend_captures_runtime_routes_and_sequence_metadata(tmp_path: Path):
    work = tmp_path / "work"
    store = StateStore(work / "states")
    transition = store.begin_transition(
        3, corpus_plan_sha256="b" * 64, expected_shards={"0000": 10}
    )
    hidden_path = transition.temporary / "hidden-0000.safetensors"
    metadata_path = transition.temporary / "metadata-0000.json"
    hidden = torch.arange(10 * 6144, dtype=torch.float32).reshape(10, 6144).bfloat16()
    previous = torch.zeros((10, 10), dtype=torch.int32)
    write_safetensors_atomic(
        hidden_path,
        [
            torch_tensor_entry("hidden", hidden),
            torch_tensor_entry("prev_topk_indices", previous),
        ],
    )
    atomic_write_json(
        metadata_path,
        {
            "schema": "r7-state-metadata-v2",
            "shard_id": "0000",
            "tokens": 10,
            "global_row_start": 100,
            "sequence_lengths": [10],
            "input_ids": list(range(10)),
            "prev_topk_shape": [10, 10],
            "attention_implementation": "eager",
            "dispatch_audit_sha256": "d" * 64,
            "auxiliary": "prev_topk_indices-int32",
            "corpus_plan_sha256": "b" * 64,
            "producer": "synthetic",
            "carrier_decode_audit_sha256": "c" * 64,
        },
    )
    transition.add_existing_shard(
        shard_id="0000",
        hidden_path=hidden_path,
        metadata_path=metadata_path,
        tokens=10,
        hidden_size=6144,
    )
    transition.commit(predecessor_sha256="p", backend_fingerprint="b")

    runtime = _SyntheticRuntime()
    backend = GLM52Backend.__new__(GLM52Backend)
    backend.runtime = runtime
    backend.work = work
    backend._fingerprint = "concrete-backend-test"
    capture = backend.capture_layer(
        layer=3,
        shards=store.load(3),
        routing_dir=work / "layer-003" / "routing",
    )
    assert capture.mass_audit.tokens == 10
    assert capture.mass_audit.assignments == 80
    assert runtime.metadata_seen["sequence_lengths"] == [10]
    assert torch.equal(runtime.metadata_seen["prev_topk_indices"], previous)
    assert (
        backend.open_capture(layer=3, routing_dir=work / "layer-003" / "routing").digest
        == capture.digest
    )
    capture_path = work / "layer-003" / "routing" / "CAPTURE.json"
    malformed = read_json(capture_path)
    malformed["runtime_arithmetic_audit"]["tensor_count"] = 0
    atomic_write_json(capture_path, malformed)
    with pytest.raises(ValueError, match="arithmetic audit failed"):
        backend.open_capture(layer=3, routing_dir=capture_path.parent)


def test_cold_fallback_tops_up_a_single_weight_bin(tmp_path: Path):
    work = tmp_path / "work"
    store = StateStore(work / "states")
    transition = store.begin_transition(
        3, corpus_plan_sha256="b" * 64, expected_shards={"0000": 100}
    )
    hidden_path = transition.temporary / "hidden-0000.safetensors"
    metadata_path = transition.temporary / "metadata-0000.json"
    tokens = 100
    hidden = (
        torch.arange(tokens * 6144, dtype=torch.float32)
        .reshape(tokens, 6144)
        .bfloat16()
    )
    previous = torch.zeros((tokens, tokens), dtype=torch.int32)
    write_safetensors_atomic(
        hidden_path,
        [
            torch_tensor_entry("hidden", hidden),
            torch_tensor_entry("prev_topk_indices", previous),
        ],
    )
    atomic_write_json(
        metadata_path,
        {
            "schema": "r7-state-metadata-v2",
            "shard_id": "0000",
            "tokens": tokens,
            "global_row_start": 0,
            "sequence_lengths": [tokens],
            "input_ids": list(range(tokens)),
            "prev_topk_shape": [tokens, tokens],
            "attention_implementation": "eager",
            "dispatch_audit_sha256": "d" * 64,
            "auxiliary": "prev_topk_indices-int32",
            "corpus_plan_sha256": "b" * 64,
            "producer": "synthetic",
            "carrier_decode_audit_sha256": "c" * 64,
        },
    )
    transition.add_existing_shard(
        shard_id="0000",
        hidden_path=hidden_path,
        metadata_path=metadata_path,
        tokens=tokens,
        hidden_size=6144,
    )
    transition.commit(predecessor_sha256="p", backend_fingerprint="b")
    backend = GLM52Backend.__new__(GLM52Backend)
    backend.runtime = _SyntheticRuntime()
    backend.work = work
    backend._fingerprint = "concrete-backend-test"
    shards = store.load(3)
    capture = backend.capture_layer(
        layer=3,
        shards=shards,
        routing_dir=work / "layer-003" / "routing",
    )
    first = backend._fallback_locations(capture, shards, 255, "fit", limit=16)
    second = backend._fallback_locations(capture, shards, 255, "fit", limit=16)
    assert first == second
    assert len(first) == 16


def test_row_cache_recovers_after_shard_only_crash(tmp_path: Path, monkeypatch):
    root = tmp_path / "rows"
    original = row_cache.atomic_write_json

    def crash_before_manifest(*_args, **_kwargs):
        raise RuntimeError("injected row-cache crash")

    monkeypatch.setattr(row_cache, "atomic_write_json", crash_before_manifest)
    with pytest.raises(RuntimeError, match="injected"):
        row_cache.write_holdout_cache(
            root,
            expert=7,
            hidden=torch.ones((2, 4), dtype=torch.bfloat16),
            row_ids=(1, 2),
            bindings={"capture": "a" * 64},
            metadata={"rows": 2},
        )
    monkeypatch.setattr(row_cache, "atomic_write_json", original)
    assert (root / "expert-007-holdout.safetensors").is_file()
    assert (
        row_cache.load_holdout_cache(
            root,
            expert=7,
            bindings={"capture": "a" * 64},
            device="cpu",
        )
        is None
    )
    assert not any(root.iterdir())
    row_cache.write_holdout_cache(
        root,
        expert=7,
        hidden=torch.ones((2, 4), dtype=torch.bfloat16),
        row_ids=(1, 2),
        bindings={"capture": "a" * 64},
        metadata={"rows": 2},
    )
    assert row_cache.load_holdout_cache(
        root,
        expert=7,
        bindings={"capture": "a" * 64},
        device="cpu",
    )["row_ids"] == (1, 2)
    atomic_write_bytes(root / "expert-007-holdout.safetensors", b"new-shard")
    assert (
        row_cache.load_holdout_cache(
            root,
            expert=7,
            bindings={"capture": "a" * 64},
            device="cpu",
        )
        is None
    )
    assert not any(root.iterdir())


def test_final_expert_cache_recovers_after_shard_only_crash(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "experts"
    records = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        tensor_id = TensorId(3, 9, projection)
        records.append(
            EncodedTensor(
                tensor_id=tensor_id,
                bits=4,
                trellis=torch.zeros((1,), dtype=torch.int16),
                suh=torch.ones((1,), dtype=torch.float16),
                svh=torch.ones((1,), dtype=torch.float16),
                reconstructed_kn=None,
                proxy_loss=0.0,
                packed_sha256="1" * 64,
                reconstruction_sha256="2" * 64,
                provenance={"test": True},
            )
        )
    original = expert_cache.atomic_write_json

    def crash_before_manifest(*_args, **_kwargs):
        raise RuntimeError("injected expert-cache crash")

    monkeypatch.setattr(expert_cache, "atomic_write_json", crash_before_manifest)
    with pytest.raises(RuntimeError, match="injected"):
        write_cached_expert(
            root,
            encoded=records,
            bindings={"allocation": "a" * 64},
            gate_up_sha256="b" * 64,
            final_loss=0.25,
            holdout_row_ids_sha256="c" * 64,
            permutation_audit={"passed": True},
        )
    monkeypatch.setattr(expert_cache, "atomic_write_json", original)
    assert (root / "expert-009.safetensors").is_file()
    assert (
        load_cached_expert(
            root,
            layer=3,
            expert=9,
            bits={record.tensor_id.key: 4 for record in records},
            bindings={"allocation": "a" * 64},
            codec=object(),
        )
        is None
    )
    assert not any(root.iterdir())
    write_cached_expert(
        root,
        encoded=records,
        bindings={"allocation": "a" * 64},
        gate_up_sha256="b" * 64,
        final_loss=0.25,
        holdout_row_ids_sha256="c" * 64,
        permutation_audit={"passed": True},
    )
    atomic_write_bytes(root / "expert-009.safetensors", b"new-shard")
    assert (
        load_cached_expert(
            root,
            layer=3,
            expert=9,
            bits={record.tensor_id.key: 4 for record in records},
            bindings={"allocation": "a" * 64},
            codec=object(),
        )
        is None
    )
    assert not any(root.iterdir())


def test_shared_full_roundtrip_resumes_at_the_next_sample_expert(tmp_path: Path):
    runner = SearchRunner.__new__(SearchRunner)
    runner.sample = (0, 1, 2)
    runner.capture = SimpleNamespace(
        mass_audit=SimpleNamespace(mass_by_expert=tuple([1.0] * 256))
    )
    runner.progress_path = tmp_path / "search-progress.json"
    runner.progress = {"scores": {}}
    calls = []

    def interrupted(_search, expert):
        calls.append(expert)
        if expert == 2:
            raise RuntimeError("injected shared-score crash")
        return float(expert + 1)

    runner._score_expert = interrupted
    with pytest.raises(RuntimeError, match="injected"):
        runner._score_shared(object(), "candidate/full")
    assert calls == [0, 1, 2]
    persisted = read_json(runner.progress_path)
    assert set(persisted["scores"]) == {
        "candidate/full/sample-expert-000",
        "candidate/full/sample-expert-001",
    }

    resumed = SearchRunner.__new__(SearchRunner)
    resumed.sample = runner.sample
    resumed.capture = runner.capture
    resumed.progress_path = runner.progress_path
    resumed.progress = persisted
    resumed_calls = []

    def finish(_search, expert):
        resumed_calls.append(expert)
        return float(expert + 1)

    resumed._score_expert = finish
    assert resumed._score_shared(object(), "candidate/full") == 2.0
    assert resumed_calls == [2]
    del resumed.progress["scores"]["candidate/full/sample-expert-001"]
    with pytest.raises(ValueError, match="complete subscore domain"):
        resumed._score_shared(object(), "candidate/full")


def test_carried_prefix_progress_reuses_a_hash_bound_prompt(tmp_path: Path):
    runtime = TransformersSequentialRuntime.__new__(TransformersSequentialRuntime)
    runtime._fingerprint = "f" * 64
    runtime.carrier_inventory = {"inventory_sha256": "a" * 64}
    runtime.dispatch_audit_sha256 = "d" * 64
    runtime._corpus_plan_payload = {
        "corpus_plan_sha256": "b" * 64,
        "selected": [
            {
                "shard_id": "prompt-000000-line-000000",
                "tokens": 2,
                "input_ids": [11, 12],
                "global_row_start": 0,
            }
        ],
    }
    progress_path, progress = runtime._open_prefix_progress(
        tmp_path, corpus_plan_sha256="b" * 64
    )
    hidden = tmp_path / "prefix-input-001-hidden-prompt.safetensors"
    metadata = tmp_path / "prefix-input-001-metadata-prompt.json"
    atomic_write_bytes(hidden, b"sealed-hidden")
    atomic_write_json(
        metadata,
        {
            "corpus_plan_sha256": "b" * 64,
            "shard_id": "prompt-000000-line-000000",
            "tokens": 2,
            "input_ids": [11, 12],
            "global_row_start": 0,
        },
    )
    record = (
        "prompt-000000-line-000000",
        hidden,
        metadata,
        2,
        6144,
    )
    runtime._seal_prefix_record(
        progress_path=progress_path,
        progress=progress,
        layer_input=1,
        record=record,
    )
    _, reopened = runtime._open_prefix_progress(tmp_path, corpus_plan_sha256="b" * 64)
    assert set(reopened["records"]) == {"001:prompt-000000-line-000000"}
    atomic_write_json(metadata, {"corpus_plan_sha256": "c" * 64})
    with pytest.raises(ValueError, match="prompt artifact drift"):
        runtime._open_prefix_progress(tmp_path, corpus_plan_sha256="b" * 64)


def test_carried_prefix_restart_skips_every_sealed_layer_prompt(tmp_path: Path):
    carrier = tmp_path / "carrier"
    carrier.mkdir()
    output = tmp_path / "partial"
    output.mkdir()
    plan = {
        "corpus_plan_sha256": "b" * 64,
        "selected": [
            {
                "shard_id": f"prompt-{index:06d}-line-{index:06d}",
                "tokens": 2,
                "input_ids": [10 + index, 20 + index],
                "global_row_start": index * 2,
                "line_index": index,
            }
            for index in range(2)
        ],
    }

    class FakeLoader:
        audit_sha256 = "e" * 64

        @staticmethod
        def weight(name):
            assert name == "model.embed_tokens"
            return (
                torch.arange(32 * 6144, dtype=torch.float32)
                .reshape(32, 6144)
                .bfloat16()
            )

    def make_runtime(call_log, *, fail_after=None):
        runtime = TransformersSequentialRuntime.__new__(TransformersSequentialRuntime)
        runtime._fingerprint = "f" * 64
        runtime.device = torch.device("cpu")
        runtime.owner_config = SimpleNamespace(carrier=carrier)
        runtime.carrier_inventory = {"inventory_sha256": "a" * 64}
        runtime.dispatch_audit_sha256 = "d" * 64
        runtime.loader = FakeLoader()
        runtime._corpus_plan_payload = plan
        runtime.prepare_corpus_plan = lambda **_kwargs: {
            "corpus_plan_sha256": "b" * 64,
            "corpus_plan_artifact_sha256": "c" * 64,
            "expected_shards": {
                str(raw["shard_id"]): int(raw["tokens"]) for raw in plan["selected"]
            },
        }
        runtime._layer_state = lambda *_args, **_kwargs: object()

        def advance(_module, hidden, metadata):
            call_log.append(str(metadata["shard_id"]))
            if fail_after is not None and len(call_log) > fail_after:
                raise RuntimeError("injected carried-prefix crash")
            value = torch.as_tensor(hidden).reshape(-1, 6144).bfloat16().cpu()
            tokens = int(metadata["tokens"])
            return value, torch.zeros((tokens, tokens), dtype=torch.int32)

        runtime._advance = advance
        return runtime

    first_calls = []
    first = make_runtime(first_calls, fail_after=3)
    with pytest.raises(RuntimeError, match="injected carried-prefix crash"):
        tuple(
            first.initialize_carried_state(
                carrier=carrier,
                corpus=tmp_path / "unused.jsonl",
                output_partial=output,
                completed_shard_ids=frozenset(),
            )
        )
    assert len(first_calls) == 4
    progress = read_json(output / "CARRIED_PREFIX_PROGRESS.json")
    assert len(progress["records"]) == 3

    resumed_calls = []
    resumed = make_runtime(resumed_calls)
    records = tuple(
        resumed.initialize_carried_state(
            carrier=carrier,
            corpus=tmp_path / "unused.jsonl",
            output_partial=output,
            completed_shard_ids=frozenset(),
        )
    )
    assert len(records) == 2
    assert len(resumed_calls) == 3
    assert not (output / "CARRIED_PREFIX_PROGRESS.json").exists()


def test_carried_prefix_cleanup_restarts_at_every_unlink_boundary(tmp_path: Path):
    carrier = tmp_path / "carrier"
    carrier.mkdir()
    selected = [
        {
            "shard_id": f"prompt-{index:06d}-line-{index:06d}",
            "tokens": 2,
            "input_ids": [10 + index, 20 + index],
            "global_row_start": index * 2,
            "line_index": index,
        }
        for index in range(2)
    ]
    plan = {"corpus_plan_sha256": "b" * 64, "selected": selected}
    completed = frozenset(str(raw["shard_id"]) for raw in selected)
    boundaries = ["before-progress-unlink", "after-progress-unlink"] + [
        f"after-unlink:{layer:03d}:{raw['shard_id']}:{kind}"
        for layer in range(1, 3)
        for raw in selected
        for kind in ("hidden", "metadata")
    ]

    def make_runtime():
        runtime = TransformersSequentialRuntime.__new__(TransformersSequentialRuntime)
        runtime.owner_config = SimpleNamespace(carrier=carrier)
        runtime._corpus_plan_payload = plan
        runtime.prepare_corpus_plan = lambda **_kwargs: {
            "corpus_plan_sha256": "b" * 64,
            "corpus_plan_artifact_sha256": "c" * 64,
            "expected_shards": {
                str(raw["shard_id"]): int(raw["tokens"]) for raw in selected
            },
        }
        runtime._layer_state = lambda *_args, **_kwargs: pytest.fail(
            "completed prompts must not be forwarded during cleanup"
        )
        return runtime

    for index, fault_boundary in enumerate(boundaries):
        output = tmp_path / f"cleanup-{index:02d}"
        output.mkdir()
        atomic_write_json(
            output / "CARRIED_PREFIX_PROGRESS.json",
            {"schema": "derivative-progress-can-be-retired"},
        )
        derivatives = []
        finals = []
        for raw in selected:
            shard_id = str(raw["shard_id"])
            final_hidden = output / f"hidden-{shard_id}.safetensors"
            final_metadata = output / f"metadata-{shard_id}.json"
            atomic_write_bytes(final_hidden, b"authoritative-hidden")
            atomic_write_bytes(final_metadata, b"authoritative-metadata")
            finals.extend((final_hidden, final_metadata))
            for layer in range(1, 3):
                hidden = output / (
                    f"prefix-input-{layer:03d}-hidden-{shard_id}.safetensors"
                )
                metadata = output / (
                    f"prefix-input-{layer:03d}-metadata-{shard_id}.json"
                )
                atomic_write_bytes(hidden, b"derivative-hidden")
                atomic_write_bytes(metadata, b"derivative-metadata")
                derivatives.extend((hidden, metadata))

        first = make_runtime()

        def crash(boundary):
            if boundary == fault_boundary:
                raise RuntimeError(f"injected cleanup crash: {boundary}")

        first._prefix_cleanup_fault_hook = crash
        with pytest.raises(RuntimeError, match="injected cleanup crash"):
            tuple(
                first.initialize_carried_state(
                    carrier=carrier,
                    corpus=tmp_path / "unused.jsonl",
                    output_partial=output,
                    completed_shard_ids=completed,
                )
            )

        resumed = make_runtime()
        assert (
            tuple(
                resumed.initialize_carried_state(
                    carrier=carrier,
                    corpus=tmp_path / "unused.jsonl",
                    output_partial=output,
                    completed_shard_ids=completed,
                )
            )
            == ()
        )
        assert not (output / "CARRIED_PREFIX_PROGRESS.json").exists()
        assert all(not path.exists() for path in derivatives)
        assert all(path.is_file() for path in finals)


def test_successor_generator_journals_before_advancing_the_next_prompt(
    tmp_path: Path,
):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    shards = []
    for index in range(2):
        shard_id = f"prompt-{index:06d}"
        hidden = inputs / f"hidden-{index}.safetensors"
        metadata = inputs / f"metadata-{index}.json"
        write_safetensors_atomic(
            hidden,
            (
                torch_tensor_entry(
                    "hidden", torch.ones((1, 6144), dtype=torch.bfloat16) * index
                ),
                torch_tensor_entry(
                    "prev_topk_indices", torch.zeros((1, 1), dtype=torch.int32)
                ),
            ),
        )
        atomic_write_json(
            metadata,
            {
                "schema": "r7-state-metadata-v2",
                "shard_id": shard_id,
                "tokens": 1,
                "global_row_start": index,
                "sequence_lengths": [1],
                "input_ids": [index],
                "prev_topk_shape": [1, 1],
                "attention_implementation": "eager",
                "dispatch_audit_sha256": "d" * 64,
                "auxiliary": "prev_topk_indices-int32",
                "corpus_plan_sha256": "b" * 64,
                "producer": "synthetic",
                "carrier_decode_audit_sha256": "c" * 64,
            },
        )
        shards.append(
            StateShard(
                shard_id=shard_id,
                hidden_path=hidden,
                metadata_path=metadata,
                tokens=1,
                hidden_size=6144,
                sha256_hidden=sha256_file(hidden),
                sha256_metadata=sha256_file(metadata),
            )
        )

    def make_runtime(calls):
        runtime = TransformersSequentialRuntime.__new__(TransformersSequentialRuntime)
        runtime.loader = SimpleNamespace(audit_records={})
        runtime._installed = {}
        runtime.dispatch_audit_sha256 = "d" * 64
        runtime._layer_state = lambda *_args, **_kwargs: object()

        def ordinary(_module, hidden, metadata):
            calls.append(str(metadata["shard_id"]))
            return hidden.bfloat16(), torch.zeros((1, 1), dtype=torch.int32)

        runtime._advance = ordinary

        def repeated(_module, hidden, metadata):
            calls.append(str(metadata["shard_id"]))
            return (
                hidden.bfloat16(),
                torch.zeros((1, 1), dtype=torch.int32),
                {
                    "schema": "r7-official-successor-repeat-v1",
                    "hidden_bf16_sha256": "1" * 64,
                    "prev_topk_i32_sha256": "2" * 64,
                    "dispatch_audit_sha256": "d" * 64,
                    "passed": True,
                },
            )

        runtime._advance_repeat_oracle = repeated
        return runtime

    first_calls = []
    first_runtime = make_runtime(first_calls)
    generator = first_runtime.forward_installed_layer(
        layer=3,
        input_shards=tuple(shards),
        output_partial=outputs,
        completed_shard_ids=frozenset(),
    )
    first_record = next(generator)
    assert first_record[0] == "prompt-000000"
    assert first_calls == ["prompt-000000"]
    generator.close()

    resumed_calls = []
    resumed_runtime = make_runtime(resumed_calls)
    remaining = tuple(
        resumed_runtime.forward_installed_layer(
            layer=3,
            input_shards=tuple(shards),
            output_partial=outputs,
            completed_shard_ids=frozenset({"prompt-000000"}),
        )
    )
    assert [record[0] for record in remaining] == ["prompt-000001"]
    assert resumed_calls == ["prompt-000001"]


def test_concrete_backend_rejects_mtp_replacement_and_forward_boundary():
    backend = GLM52Backend.__new__(GLM52Backend)
    backend.runtime = _SyntheticRuntime()
    with pytest.raises(ValueError, match="layers 3..77"):
        backend.load_bf16_expert(layer=78, expert=0)
    with pytest.raises(ValueError, match="layers 3..77"):
        backend.restore_encoded_layer(layer=78, manifest=Path("unused"))
    with pytest.raises(ValueError, match="layers 3..76"):
        tuple(
            backend.forward_installed_layer(
                layer=77,
                input_shards=(),
                output_partial=Path("unused"),
                completed_shard_ids=frozenset(),
            )
        )


def test_backend_validates_install_receipt_and_official_layer_oracle():
    backend = GLM52Backend.__new__(GLM52Backend)
    backend.runtime = _SyntheticRuntime()
    backend._fingerprint = "backend-test"
    encoded = {
        f"L03/E000/{projection}": {
            "packed_sha256": str(index + 1) * 64,
            "reconstruction_sha256": str(index + 4) * 64,
            "reconstructed_kn": object(),
        }
        for index, projection in enumerate(("gate_proj", "up_proj", "down_proj"))
    }
    receipt = backend.install_encoded_expert(layer=3, expert=0, encoded=encoded)
    assert receipt["schema"] == "r7-packed-install-record-v2"
    assert receipt["backend_fingerprint"] == "backend-test"
    audit = backend.audit_installed_layer(layer=3)
    assert audit["schema"] == "r7-official-installed-layer-audit-v1"
    assert audit["backend_fingerprint"] == "backend-test"


def test_assembly_and_conversion_transactions_are_resume_bound(tmp_path: Path):
    assembly = tmp_path / "assembly.partial"
    assembly.mkdir()
    walk = tmp_path / "WALK_COMPLETE.json"
    atomic_write_json(walk, {"walk": 1})
    first = _bind_assembly_transaction(
        assembly,
        carrier=tmp_path / "carrier",
        v2=tmp_path / "v2",
        inventory_sha256="a" * 64,
        walk_manifest=walk,
        layer_manifest_sha256={"3": "b" * 64},
    )
    assert first == _bind_assembly_transaction(
        assembly,
        carrier=tmp_path / "carrier",
        v2=tmp_path / "v2",
        inventory_sha256="a" * 64,
        walk_manifest=walk,
        layer_manifest_sha256={"3": "b" * 64},
    )
    with pytest.raises(ValueError, match="different transaction"):
        _bind_assembly_transaction(
            assembly,
            carrier=tmp_path / "carrier-2",
            v2=tmp_path / "v2",
            inventory_sha256="a" * 64,
            walk_manifest=walk,
            layer_manifest_sha256={"3": "b" * 64},
        )

    conversion = tmp_path / "conversion.partial"
    conversion.mkdir()
    converted = _bind_conversion_transaction(
        conversion,
        source=tmp_path / "source",
        source_manifest_sha256="c" * 64,
        tp_size=4,
    )
    assert converted == _bind_conversion_transaction(
        conversion,
        source=tmp_path / "source",
        source_manifest_sha256="c" * 64,
        tp_size=4,
    )
    with pytest.raises(ValueError, match="different transaction"):
        _bind_conversion_transaction(
            conversion,
            source=tmp_path / "source",
            source_manifest_sha256="c" * 64,
            tp_size=8,
        )


def test_sample_layer_feasibility_report_is_bound_but_not_a_completion_claim(
    tmp_path: Path,
):
    work = tmp_path / "work"
    (work / "v2").mkdir(parents=True)
    (work / "layer-003").mkdir()
    (work / "search").mkdir()
    manifest = work / "v2" / "r7-experts-layer-003.json"
    oracle = work / "layer-003" / "oracle-report.json"
    search = work / "search" / "layer-003.json"
    for path, payload in (
        (manifest, {"manifest": 3}),
        (oracle, {"passed": True}),
        (search, {"search": 3}),
    ):
        atomic_write_json(path, payload)

    walk = SequentialWalk.__new__(SequentialWalk)
    walk.work = work
    walk.config = SimpleNamespace(device="cuda:0")
    walk.carrier_inventory = {"inventory_sha256": "a" * 64}
    walk.source_inventory = {"inventory_sha256": "b" * 64}
    walk.numeric_inventory = {"inventory_sha256": "c" * 64}
    walk.runtime_inventory = {"inventory_sha256": "d" * 64}
    walk.journal = Journal(work / "JOURNAL.json", recipe_config={"test": 1})
    walk.journal.seal(
        StageSeal(
            "layer-003-encoded",
            {},
            {str(manifest.relative_to(work)): sha256_file(manifest)},
            {"layer": 3},
        )
    )
    result = walk._write_pilot_report(
        layer=3,
        elapsed_ns=2_500_000_000,
        initialization_ns=500_000_000,
        layer_elapsed_ns={3: 2_000_000_000},
        initial_seals=(),
    )
    report = read_json(work / "FEASIBILITY_LAYER_003.json")
    assert result["complete"] is False
    assert report["passed"] is True
    assert report["complete"] is False
    assert report["elapsed_seconds"] == "2.500000"
    assert report["initialization_seconds"] == "0.500000"
    assert report["layer_seconds"] == {"3": "2.000000"}
    assert report["timing_valid_for_projection"] is True
    assert report["projection_status"] == "UNVERIFIED"
    assert report["quality_status"] == "UNVERIFIED-not-an-evaluation"
    assert report["bindings"]["layer_manifest_sha256"] == sha256_file(manifest)
