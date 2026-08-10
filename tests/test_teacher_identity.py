from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import capture_calibration as capture_driver
from src.calibration_capture import (
    CAPTURE_CODE_FILES,
    TEACHER_IDENTITY_SEAL_SHA256,
    TEACHER_IDENTITY_VALIDATION,
    validate_teacher_identity_evidence,
)
from src.teacher_identity import (
    DEFAULT_POLICY,
    TeacherIdentityError,
    TeacherIdentityPolicy,
    build_teacher_identity_receipt,
    canonical_json_bytes,
    validate_teacher_identity_attestation,
    validate_teacher_identity_receipt,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_default_policy_is_the_exact_glm52_r33_loader_surface() -> None:
    assert len(DEFAULT_POLICY.payload_files) == 156
    assert len(DEFAULT_POLICY.sidecar_files) == 75
    assert len(DEFAULT_POLICY.identity_files) == 8
    assert set(DEFAULT_POLICY.payload_files) == {
        "model-embed.safetensors",
        "model-head.safetensors",
        *(f"model-layer-{layer:03d}.safetensors" for layer in range(79)),
        *(f"r7-experts-layer-{layer:03d}.safetensors" for layer in range(3, 78)),
    }
    assert DEFAULT_POLICY.sidecar_files == tuple(
        f"r7-experts-layer-{layer:03d}.json" for layer in range(3, 78)
    )
    assert "seal_teacher_model.py" in CAPTURE_CODE_FILES
    assert "src/teacher_identity.py" in CAPTURE_CODE_FILES


def test_capture_runtime_requires_exact_full_teacher_evidence() -> None:
    assert validate_teacher_identity_evidence(TEACHER_IDENTITY_VALIDATION) == (
        TEACHER_IDENTITY_VALIDATION
    )
    for key, replacement in (
        ("verification_mode", "metadata"),
        ("all_file_bytes_sha256_validated", False),
        ("receipt_sha256", "0" * 64),
        ("seal_sha256", "0" * 64),
        ("payload_count", 155),
    ):
        drifted = {**TEACHER_IDENTITY_VALIDATION, key: replacement}
        with pytest.raises(ValueError, match="identity validation evidence differs"):
            validate_teacher_identity_evidence(drifted)


@pytest.mark.parametrize("smoke_dcp4", [True, False])
def test_direct_capture_full_hashes_mounted_teacher_before_tokenizer_or_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    smoke_dcp4: bool,
) -> None:
    events: list[object] = []
    monkeypatch.setattr(capture_driver, "load_document_plan", lambda _: {})
    monkeypatch.setattr(capture_driver, "validate_smoke_run_token", lambda _: "token")
    monkeypatch.setattr(
        capture_driver,
        "validate_full_capture_preflight",
        lambda *args, **kwargs: events.append("full_preflight") or {},
    )
    monkeypatch.setattr(
        capture_driver,
        "validate_capture_runtime",
        lambda: events.append("runtime_provenance") or {},
    )

    def fake_teacher(model: Path, receipt: Path, **kwargs: object) -> dict:
        events.append(("teacher_full_hash", model, receipt, kwargs))
        return dict(TEACHER_IDENTITY_VALIDATION)

    monkeypatch.setattr(
        capture_driver,
        "validate_teacher_identity_receipt",
        fake_teacher,
    )

    class TokenizerReached(RuntimeError):
        pass

    def refuse_tokenizer(_: Path) -> object:
        events.append("tokenizer")
        raise TokenizerReached

    monkeypatch.setattr(capture_driver, "_tokenizer", refuse_tokenizer)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=object))
    model = tmp_path / "model"
    model.mkdir()
    args = SimpleNamespace(
        model=model,
        plan_file=tmp_path / "plan.json",
        smoke_dir=tmp_path / "smoke",
        jit_cache_dir=tmp_path / "jit",
    )
    with pytest.raises(TokenizerReached):
        capture_driver.run_capture(args, smoke_dcp4=smoke_dcp4)

    teacher_event = next(
        value for value in events if isinstance(value, tuple) and value[0] == "teacher_full_hash"
    )
    assert events.index(teacher_event) < events.index("tokenizer")
    assert teacher_event[1] == model
    assert teacher_event[2] == Path("/work/evidence/teacher_model_identity.json")
    assert teacher_event[3] == {
        "expected_seal_sha256": TEACHER_IDENTITY_SEAL_SHA256,
        "verify_mode": "full",
        "workers": 4,
    }


def _json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _fixture(tmp_path: Path) -> tuple[Path, TeacherIdentityPolicy]:
    root = tmp_path / "teacher"
    root.mkdir()
    payloads = (
        "model-embed.safetensors",
        "r7-experts-layer-003.safetensors",
    )
    sidecars = ("r7-experts-layer-003.json",)
    policy = TeacherIdentityPolicy(payload_files=payloads, sidecar_files=sidecars)

    (root / payloads[0]).write_bytes(b"tiny embed payload\x00")
    (root / payloads[1]).write_bytes(b"tiny routed payload\x01")
    quantization = {
        "r7_routed_experts": {
            "bit_map_manifests": list(sidecars),
            "payload_hash_verification": False,
        }
    }
    _json(
        root / "config.json",
        {
            "hybrid_tr3_tail": {"tier_bitmap": "tier_bitmap.json"},
            "quantization_config": quantization,
        },
    )
    _json(root / "quantization_config.json", quantization)
    _json(root / "tier_bitmap.json", {"schema": "tiny"})
    _json(root / "tokenizer.json", {"model": {"type": "WordLevel"}})
    _json(root / "tokenizer_config.json", {"tokenizer_class": "Tiny"})
    (root / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")
    _json(root / "generation_config.json", {"temperature": 1.0})
    _json(
        root / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 2},
            "weight_map": {
                "model.embed_tokens.weight": payloads[0],
                "model.layers.3.mlp.experts.0.gate_proj.trellis": payloads[1],
            },
        },
    )
    _json(
        root / sidecars[0],
        {
            "layer": 3,
            "schema_version": 2,
            "shard": payloads[1],
            "shard_sha256": _sha256(root / payloads[1]),
        },
    )
    return root, policy


def _build(tmp_path: Path) -> tuple[Path, Path, TeacherIdentityPolicy, dict]:
    root, policy = _fixture(tmp_path)
    receipt = tmp_path / "teacher_identity.json"
    result = build_teacher_identity_receipt(
        root,
        receipt,
        workers=2,
        policy=policy,
    )
    return root, receipt, policy, result


def test_build_and_full_validation_close_exact_allowlist(tmp_path: Path) -> None:
    root, receipt, policy, built = _build(tmp_path)
    assert built["verification_mode"] == "metadata"
    assert built["all_file_bytes_sha256_validated"] is False
    assert built["non_payload_file_bytes_sha256_validated"] is True
    assert built["payload_count"] == 2
    assert built["loader_sidecar_count"] == 1
    assert built["loader_identity_count"] == 8
    assert built["total_file_count"] == 11

    full = validate_teacher_identity_receipt(
        root,
        receipt,
        expected_seal_sha256=built["seal_sha256"],
        verify_mode="full",
        workers=2,
        policy=policy,
    )
    assert full["all_file_bytes_sha256_validated"] is True
    assert full["receipt_sha256"] == _sha256(receipt)
    parsed = json.loads(receipt.read_text())
    assert parsed["seal_sha256"] == full["seal_sha256"]
    assert [item["path"] for item in parsed["seal"]["files"]] == sorted(
        set(policy.payload_files)
        | set(policy.sidecar_files)
        | set(policy.identity_files)
    )


def test_build_is_create_once_and_never_overwrites(tmp_path: Path) -> None:
    root, receipt, policy, _ = _build(tmp_path)
    before = receipt.read_bytes()
    with pytest.raises(TeacherIdentityError, match="overwrite"):
        build_teacher_identity_receipt(root, receipt, workers=1, policy=policy)
    assert receipt.read_bytes() == before


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    root, policy = _fixture(tmp_path)
    (root / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":2},"weight_map":'
        '{"same":"model-embed.safetensors",'
        '"same":"r7-experts-layer-003.safetensors"}}',
        encoding="utf-8",
    )
    with pytest.raises(TeacherIdentityError, match="duplicate key"):
        build_teacher_identity_receipt(
            root,
            tmp_path / "receipt.json",
            workers=1,
            policy=policy,
        )


def test_unsafe_index_path_is_rejected(tmp_path: Path) -> None:
    root, policy = _fixture(tmp_path)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    index["weight_map"]["model.embed_tokens.weight"] = "../model-embed.safetensors"
    _json(root / "model.safetensors.index.json", index)
    with pytest.raises(TeacherIdentityError, match="safe top-level"):
        build_teacher_identity_receipt(
            root,
            tmp_path / "receipt.json",
            workers=1,
            policy=policy,
        )


def test_symlink_and_unreferenced_payloads_are_rejected(tmp_path: Path) -> None:
    root, policy = _fixture(tmp_path)
    payload = root / "model-embed.safetensors"
    payload.unlink()
    payload.symlink_to(root / "r7-experts-layer-003.safetensors")
    with pytest.raises(TeacherIdentityError, match="symlink"):
        build_teacher_identity_receipt(
            root,
            tmp_path / "receipt-a.json",
            workers=1,
            policy=policy,
        )

    payload.unlink()
    payload.write_bytes(b"tiny embed payload\x00")
    nested = root / "ambiguous"
    nested.mkdir()
    (nested / "extra.safetensors").write_bytes(b"extra")
    with pytest.raises(TeacherIdentityError, match="unreferenced"):
        build_teacher_identity_receipt(
            root,
            tmp_path / "receipt-b.json",
            workers=1,
            policy=policy,
        )


def test_same_size_payload_drift_fails_full_sha256(tmp_path: Path) -> None:
    root, receipt, policy, _ = _build(tmp_path)
    payload = root / "model-embed.safetensors"
    raw = payload.read_bytes()
    payload.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(TeacherIdentityError, match="SHA256 differs"):
        validate_teacher_identity_receipt(
            root,
            receipt,
            verify_mode="full",
            workers=2,
            policy=policy,
        )


def test_sidecar_claim_is_independently_cross_checked(tmp_path: Path) -> None:
    root, policy = _fixture(tmp_path)
    sidecar = root / "r7-experts-layer-003.json"
    value = json.loads(sidecar.read_text())
    value["shard_sha256"] = "0" * 64
    _json(sidecar, value)
    with pytest.raises(TeacherIdentityError, match="sidecar claim"):
        build_teacher_identity_receipt(
            root,
            tmp_path / "receipt.json",
            workers=2,
            policy=policy,
        )


def test_receipt_digest_tamper_is_rejected_before_file_trust(tmp_path: Path) -> None:
    root, receipt, policy, _ = _build(tmp_path)
    value = json.loads(receipt.read_text())
    value["seal"]["files"][0]["sha256"] = "0" * 64
    _json(receipt, value)
    with pytest.raises(TeacherIdentityError, match="canonical seal digest"):
        validate_teacher_identity_receipt(
            root,
            receipt,
            verify_mode="full",
            workers=1,
            policy=policy,
        )


def test_full_attestation_binds_to_current_receipt_and_metadata(tmp_path: Path) -> None:
    root, receipt, policy, built = _build(tmp_path)
    full = validate_teacher_identity_receipt(
        root,
        receipt,
        expected_seal_sha256=built["seal_sha256"],
        verify_mode="full",
        workers=2,
        policy=policy,
    )
    attestation = tmp_path / "preflight.json"
    _json(attestation, full)
    assert validate_teacher_identity_attestation(
        attestation,
        root,
        receipt,
        expected_seal_sha256=built["seal_sha256"],
        workers=2,
        policy=policy,
    ) == full

    bad = dict(full)
    bad["total_bytes"] += 1
    _json(attestation, bad)
    with pytest.raises(TeacherIdentityError, match="does not bind"):
        validate_teacher_identity_attestation(
            attestation,
            root,
            receipt,
            expected_seal_sha256=built["seal_sha256"],
            workers=1,
            policy=policy,
        )
