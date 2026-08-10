from __future__ import annotations

import os
from pathlib import Path

import pytest

import src.calibration_recovery as recovery
from src.calibration_capture import (
    FILE_ABI,
    SELECTED_LAYERS,
    atomic_json,
    validate_capture,
)


def _router_audit() -> dict:
    return {
        "router_routed_scaling_factor": 1.0,
        "runner_output_scale": 2.5,
    }


def test_lost_worker_values_are_explicitly_unavailable_not_synthesized() -> None:
    value = recovery._unavailable_route_evidence(6, _router_audit())
    assert value["raw_router_return_weights"]["sha256"] is None
    assert value["raw_router_return_weights"]["sum"] is None
    assert value["reference_route_diagnostics"]["boundary_ambiguous_rows"] is None
    assert value["reference_route_diagnostics"]["max_abs_weight_error"] is None
    recovery._validate_unavailable_route_evidence(
        value, layer=6, router_audit=_router_audit()
    )

    fabricated = {
        **value,
        "reference_route_diagnostics": {
            **value["reference_route_diagnostics"],
            "boundary_ambiguous_rows": 0,
        },
    }
    with pytest.raises(ValueError, match="unavailable-route evidence"):
        recovery._validate_unavailable_route_evidence(
            fabricated, layer=6, router_audit=_router_audit()
        )


def test_archived_jit_inventory_is_self_sealed_without_live_tree() -> None:
    body = {
        "directories": ["torchinductor", "triton/cache"],
        "files": {
            "torchinductor/kernel.so": {"bytes": 17, "sha256": "a" * 64},
        },
    }
    value = {**body, "inventory_sha256": recovery._canonical_digest(body)}
    assert recovery._validate_inventory(value) == value

    drifted = {**value, "inventory_sha256": "0" * 64}
    with pytest.raises(ValueError, match="inventory digest"):
        recovery._validate_inventory(drifted)


def test_recovery_rejects_links_and_cross_name_inode_aliases(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.write_bytes(b"payload")
    assert recovery._identity(payload)[2] == 7

    link = tmp_path / "link"
    link.symlink_to(payload)
    with pytest.raises(ValueError, match="not a regular file"):
        recovery._identity(link)

    alias = tmp_path / "alias"
    os.link(payload, alias)
    with pytest.raises(ValueError, match="another hard link"):
        recovery._identity(payload)


def test_recovery_layout_requires_only_final_payload_names(tmp_path: Path) -> None:
    (tmp_path / "document_plan.json").write_text("{}", encoding="utf-8")
    for layer in SELECTED_LAYERS:
        directory = tmp_path / f"layer_{layer:03d}"
        directory.mkdir()
        for name in FILE_ABI:
            (directory / name).touch()
    recovery._require_payload_layout(tmp_path, with_manifests=False)

    partial = tmp_path / "layer_006" / "hidden.bf16.bin.partial"
    partial.touch()
    with pytest.raises(ValueError, match="file set differs"):
        recovery._require_payload_layout(tmp_path, with_manifests=False)


def test_validate_capture_dispatches_recovered_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = {"complete": True, "capture_run_uuid": "recovered"}
    atomic_json(
        tmp_path / "capture_manifest.json",
        {"schema": recovery.RECOVERED_CAPTURE_SCHEMA},
    )
    monkeypatch.setattr(
        recovery,
        "validate_recovered_capture",
        lambda path, *, verify_hashes: marker,
    )
    assert validate_capture(tmp_path, verify_hashes=False) == marker
