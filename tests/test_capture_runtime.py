from __future__ import annotations

from pathlib import Path

import pytest

from src.calibration_capture import validate_runtime_provenance_evidence
from src.calibration_plan import sha256_file
from src.capture_runtime import (
    IMAGE_DECLARATION_ENV,
    RUNTIME_ENV,
    RUNTIME_FILES,
    RUNTIME_IMAGE_ID,
    RUNTIME_IMAGE_REFERENCE,
    RUNTIME_PYTHON,
    validate_capture_runtime,
)


def _environment() -> dict[str, str]:
    return {
        **RUNTIME_ENV,
        IMAGE_DECLARATION_ENV: RUNTIME_IMAGE_ID,
    }


def test_live_runtime_gate_hashes_files_and_rejects_host_python(tmp_path: Path) -> None:
    marker = tmp_path / ".dockerenv"
    marker.write_bytes(b"")
    runtime_file = tmp_path / "override.py"
    runtime_file.write_bytes(b"exact override\n")
    files = {str(runtime_file): sha256_file(runtime_file)}

    evidence = validate_capture_runtime(
        executable=RUNTIME_PYTHON,
        environ=_environment(),
        container_marker=marker,
        runtime_files=files,
    )
    assert evidence["files"][str(runtime_file)]["bytes"] == 15

    with pytest.raises(RuntimeError, match="host vLLM is invalid"):
        validate_capture_runtime(
            executable="/usr/bin/python3",
            environ=_environment(),
            container_marker=marker,
            runtime_files=files,
        )

    with pytest.raises(RuntimeError, match="runtime file drift"):
        validate_capture_runtime(
            executable=RUNTIME_PYTHON,
            environ=_environment(),
            container_marker=marker,
            runtime_files={str(runtime_file): "0" * 64},
        )

    drifted = _environment()
    drifted["VLLM_EXL3_PREFILL_CAPACITY"] = "1024"
    with pytest.raises(RuntimeError, match="environment differs"):
        validate_capture_runtime(
            executable=RUNTIME_PYTHON,
            environ=drifted,
            container_marker=marker,
            runtime_files=files,
        )

    aliased = _environment()
    aliased["VLLM_DCP_UNSEALED_ALIAS"] = "1"
    with pytest.raises(RuntimeError, match="unsealed DCP/EXL3 control aliases"):
        validate_capture_runtime(
            executable=RUNTIME_PYTHON,
            environ=aliased,
            container_marker=marker,
            runtime_files=files,
        )


def test_sealed_manifest_runtime_provenance_requires_every_exact_hash() -> None:
    evidence = {
        "image_reference": RUNTIME_IMAGE_REFERENCE,
        "image_id_declared": RUNTIME_IMAGE_ID,
        "image_identity_observation": (
            "host docker inspect declaration; independently bound below by exact "
            "mounted execution-file hashes"
        ),
        "python_executable": RUNTIME_PYTHON,
        "environment": RUNTIME_ENV,
        "files": {
            filename: {"sha256": digest, "bytes": 1}
            for filename, digest in RUNTIME_FILES.items()
        },
    }
    assert validate_runtime_provenance_evidence(evidence) == evidence
    drifted = {**evidence, "files": {**evidence["files"]}}
    filename = next(iter(RUNTIME_FILES))
    drifted["files"][filename] = {"sha256": "0" * 64, "bytes": 1}
    with pytest.raises(ValueError, match="provenance differs"):
        validate_runtime_provenance_evidence(drifted)
