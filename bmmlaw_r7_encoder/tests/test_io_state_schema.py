from __future__ import annotations

from pathlib import Path

from r7_encoder.assemble import is_replaced_routed_tensor
from r7_encoder.constants import TensorId
from r7_encoder.determinism import atomic_write_bytes, atomic_write_json, sha256_file
from r7_encoder.safetensors_io import (
    SafeTensorReader,
    TensorEntry,
    write_safetensors_atomic,
)
from r7_encoder.schema import load_time_tp_slices
from r7_encoder.state import Journal, StageSeal, StateStore


def test_raw_safetensors_roundtrip(tmp_path: Path):
    destination = tmp_path / "x.safetensors"
    entries = [
        TensorEntry("a", "I16", (2, 3), bytes(range(12))),
        TensorEntry("b", "U8", (4,), b"abcd"),
    ]
    hashes, file_hash = write_safetensors_atomic(
        destination, entries, metadata={"format": "pt"}
    )
    reader = SafeTensorReader(destination)
    assert reader.metadata == {"format": "pt"}
    assert reader.tensors["a"].payload.sha256() == hashes["a"]
    assert reader.tensors["b"].payload.sha256() == hashes["b"]
    assert sha256_file(destination) == file_hash


def test_tp_slices_use_128_boundaries():
    gate = load_time_tp_slices(TensorId(3, 0, "gate_proj"), 4)
    assert [
        (item.trellis_axis, item.vector_start, item.vector_end) for item in gate
    ] == [
        (1, 0, 512),
        (1, 512, 1024),
        (1, 1024, 1536),
        (1, 1536, 2048),
    ]
    down = load_time_tp_slices(TensorId(3, 0, "down_proj"), 4)
    assert [
        (item.trellis_axis, item.vector_start, item.vector_end) for item in down
    ] == [
        (0, 0, 512),
        (0, 512, 1024),
        (0, 1024, 1536),
        (0, 1536, 2048),
    ]


def test_every_supported_tp_divisor_and_invalid_boundaries():
    for tp_size in (1, 2, 4, 8, 16):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            slices = load_time_tp_slices(TensorId(3, 0, projection), tp_size)
            assert len(slices) == tp_size
            assert slices[0].vector_start == 0
            assert slices[-1].vector_end == 2048
            assert all(item.vector_start % 128 == 0 for item in slices)
            assert all(item.vector_end % 128 == 0 for item in slices)
    for tp_size in (3, 32):
        with __import__("pytest").raises(ValueError, match="128-element boundary"):
            load_time_tp_slices(TensorId(3, 0, "down_proj"), tp_size)


def test_mtp_layer_is_never_replaced():
    assert is_replaced_routed_tensor(
        "model.layers.77.mlp.experts.2.down_proj.rank0.trellis"
    )
    assert not is_replaced_routed_tensor(
        "model.layers.78.mlp.experts.2.down_proj.rank0.trellis"
    )


def test_state_transition_and_journal_survive_retirement(tmp_path: Path):
    work = tmp_path / "work"
    store = StateStore(work / "states")
    plan_sha256 = "a" * 64
    transition = store.begin_transition(
        3, corpus_plan_sha256=plan_sha256, expected_shards={"0000": 2}
    )
    hidden = transition.temporary / "h.bin"
    metadata = transition.temporary / "m.json"
    atomic_write_bytes(hidden, b"hidden")
    atomic_write_json(
        metadata,
        {"corpus_plan_sha256": plan_sha256, "shard_id": "0000", "tokens": 2},
    )
    transition.add_existing_shard(
        shard_id="0000",
        hidden_path=hidden,
        metadata_path=metadata,
        tokens=2,
        hidden_size=6144,
    )
    digest = transition.commit(
        predecessor_sha256="carry", backend_fingerprint="backend"
    )
    assert sha256_file(store.seal_archive_path(3)) == digest
    assert store.load(3)[0].tokens == 2

    journal = Journal(work / "JOURNAL.json", recipe_config={"x": 1})
    relative = store.seal_archive_path(3).relative_to(work)
    seal = StageSeal("state-input-003", {}, {str(relative): digest}, {"layer": 3})
    journal.seal(seal)
    journal.seal(seal)
    journal.audit_outputs(work)
