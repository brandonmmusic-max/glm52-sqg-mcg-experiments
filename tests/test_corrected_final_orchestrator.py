from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts.encode_final_shard import _validated_sealed_shard_identity


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/run_corrected_final_encode.sh"


def test_corrected_final_orchestrator_parses() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_corrected_final_orchestrator_uses_proven_parallel_shape() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "layers=(6 28 52 77)" in source
    assert "starts=(0 64 128 192)" in source
    assert "ends=(64 128 192 256)" in source
    assert "for shard in 0 1 2 3" in source
    assert "scripts/encode_final_shard.py" in source
    assert "--device cuda:0 --threads 3" in source
    assert "--full-revalidate" not in source
    assert "sha256sum" not in source


def test_corrected_final_orchestrator_seals_zero_mcg_and_materializes_new_path() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "scripts/consolidate_final_shards.py" in source
    assert "-m src.run_fresh_sqg seal-run" in source
    assert "3,072" in source and "zero MCG" in source
    assert "scripts/materialize_fast_directional.py" in source
    assert '[[ ! -e "$CANDIDATE_OUTPUT" && ! -L "$CANDIDATE_OUTPUT" ]]' in source
    assert 'src=$CANDIDATE_PARENT,dst=$CANDIDATE_PARENT' in source
    assert 'src=$SOURCE_MODEL,dst=$SOURCE_MODEL,readonly' not in source
    assert "distinct source/output paths" in source
    assert "EXDEV" in source
    assert "glm-r33-fixed" not in source
    assert "profile_search_shard.py" not in source


def test_fast_final_worker_rejects_bf16_file_identity_drift(tmp_path: Path) -> None:
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"sealed")
    stat = shard.stat()
    expected = {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    identity = _validated_sealed_shard_identity(shard, expected)
    assert identity[:4] == (
        expected["device"],
        expected["inode"],
        expected["bytes"],
        expected["mtime_ns"],
    )

    shard.write_bytes(b"drifted")
    with pytest.raises(ValueError, match="identity drift without payload hash"):
        _validated_sealed_shard_identity(shard, expected)
