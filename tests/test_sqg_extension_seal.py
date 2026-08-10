from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import src.sqg_extension_seal as seal_module
from src.fresh_pipeline_common import (
    atomic_json,
    canonical_sha256,
    code_tree_manifest,
    sha256_file,
)
from src.sqg_extension_seal import (
    BUILD_CONTRACT,
    R33_IMAGE_ID,
    SQG_EXTENSION_MODULE,
    SQG_EXTENSION_SEAL_SCHEMA,
    validate_sqg_extension_seal,
)


def _fixture_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, dict]:
    kquant_root = tmp_path / "kquant-root"
    csrc = kquant_root / "kquant/csrc"
    csrc.mkdir(parents=True)
    (csrc / "sqg_quantize.cpp").write_text("cpp", encoding="utf-8")
    (csrc / "sqg_quantize.cu").write_text("cu", encoding="utf-8")
    extension_dir = tmp_path / "sealed-extension"
    extension_dir.mkdir()
    extension = extension_dir / seal_module._expected_extension_name()
    extension.write_bytes(b"sealed-extension")
    provenance = {"revision": "fixture"}
    monkeypatch.setattr(seal_module, "git_provenance", lambda root: provenance)
    smoke = {
        "device": "cuda:0",
        "visible_device_count": 1,
        "records": [
            {
                "bits": bits,
                "output_sha256": str(bits) * 64,
                "indices_sha256": str(bits + 1) * 64,
                "finite": True,
            }
            for bits in (3, 4)
        ],
        "passed": True,
    }
    smoke["smoke_id"] = canonical_sha256(smoke)
    value = {
        "schema": SQG_EXTENSION_SEAL_SCHEMA,
        "complete": True,
        "module_name": SQG_EXTENSION_MODULE,
        "runtime_image_id": R33_IMAGE_ID,
        "extension": {
            "path": str(extension),
            "bytes": extension.stat().st_size,
            "sha256": sha256_file(extension),
        },
        "build_contract": BUILD_CONTRACT,
        "kquant": provenance,
        "csrc_tree": code_tree_manifest(csrc),
        "runtime_environment": seal_module._runtime_environment(),
        "compiler_environment": {"fixture": True},
        "build_receipt": {"fixture": True},
        "smoke": smoke,
        "jit_allowed_in_workers": False,
    }
    value["seal_id"] = canonical_sha256(value)
    seal_path = extension_dir / "sqg-extension-seal.json"
    atomic_json(seal_path, value)
    return seal_path, kquant_root, extension, value


def test_static_extension_seal_validation_rehashes_code_and_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seal_path, kquant_root, extension, value = _fixture_seal(
        tmp_path, monkeypatch
    )

    observed = validate_sqg_extension_seal(
        seal_path,
        kquant_root=kquant_root,
        require_runtime_environment=False,
    )

    assert observed == value
    assert observed["extension"]["sha256"] == sha256_file(extension)


def test_extension_seal_rejects_binary_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seal_path, kquant_root, extension, _ = _fixture_seal(tmp_path, monkeypatch)
    extension.write_bytes(b"changed")

    with pytest.raises(ValueError, match="extension bytes differ"):
        validate_sqg_extension_seal(
            seal_path,
            kquant_root=kquant_root,
            require_runtime_environment=False,
        )


def test_extension_seal_rejects_symlinked_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seal_path, kquant_root, extension, value = _fixture_seal(
        tmp_path, monkeypatch
    )
    target = tmp_path / "outside.so"
    target.write_bytes(extension.read_bytes())
    extension.unlink()
    extension.symlink_to(target)

    with pytest.raises(ValueError, match="non-symlink regular file"):
        validate_sqg_extension_seal(
            seal_path,
            kquant_root=kquant_root,
            require_runtime_environment=False,
        )
    assert value["extension"]["path"] == str(extension)


def test_runtime_extension_seal_requires_exact_prebuilt_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seal_path, kquant_root, extension, value = _fixture_seal(
        tmp_path, monkeypatch
    )
    monkeypatch.setenv("FRESH_SQG_RUNTIME_IMAGE_ID", R33_IMAGE_ID)
    monkeypatch.setenv("KQUANT_SQG_REQUIRE_PREBUILT", "1")
    monkeypatch.setenv("KQUANT_SQG_EXTENSION_PATH", str(extension))
    monkeypatch.setenv(
        "KQUANT_SQG_EXTENSION_SHA256", value["extension"]["sha256"]
    )

    validate_sqg_extension_seal(seal_path, kquant_root=kquant_root)
    monkeypatch.setenv("KQUANT_SQG_EXTENSION_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="override SHA256 differs"):
        validate_sqg_extension_seal(seal_path, kquant_root=kquant_root)


def test_extension_seal_canonical_id_covers_smoke_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seal_path, kquant_root, _, value = _fixture_seal(tmp_path, monkeypatch)
    changed = deepcopy(value)
    changed["smoke"]["records"][0]["finite"] = False
    atomic_json(seal_path, changed, overwrite=True)

    with pytest.raises(ValueError, match="canonical seal_id binding"):
        validate_sqg_extension_seal(
            seal_path,
            kquant_root=kquant_root,
            require_runtime_environment=False,
        )
