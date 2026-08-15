from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_wave_bf16_manifest_from_source_seal.py"
)


def digest(character: str) -> str:
    return character * 64


def source_seal() -> dict[str, object]:
    shards = {
        "model-00001-of-00282.safetensors": {
            "bytes": 100,
            "sha256": digest("a"),
            "header_sha256": digest("b"),
            "selected_tensor_count": 1536,
            "total_tensor_count": 1600,
        },
        "model-00002-of-00282.safetensors": {
            "bytes": 200,
            "sha256": digest("c"),
            "header_sha256": digest("d"),
            "selected_tensor_count": 1536,
            "total_tensor_count": 1600,
        },
    }
    return {
        "schema": "glm52-fresh-sqg-bf16-source-v2",
        "repo": "zai-org/GLM-5.2",
        "revision": "b4734de4facf877f85769a911abafc5283eab3d9",
        "layers": [3, 4, 5, 6],
        "index": {
            "sha256": "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e"
        },
        "complete_index_header_binding_validated": True,
        "shards": shards,
        "shard_count": 2,
        "target_tensor_count": 3072,
        "total_shard_bytes": 300,
    }


def run(source: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-seal",
            str(source),
            "--layers",
            "3",
            "4",
            "5",
            "6",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )


def test_derives_exact_legacy_manifest_without_payload_files(tmp_path: Path) -> None:
    source = tmp_path / "source-seal.json"
    output = tmp_path / "wave-manifest.json"
    source.write_text(json.dumps(source_seal()))
    first = run(source, output)
    assert first.returncode == 0, first.stderr
    manifest = json.loads(output.read_text())
    assert manifest["schema"] == "glm52-fresh-sqg-bf16-shard-manifest-v1"
    assert manifest["layers"] == [3, 4, 5, 6]
    assert manifest["selected_tensor_count"] == 3072
    assert manifest["total_shard_bytes"] == 300
    assert manifest["shards"]["model-00001-of-00282.safetensors"]["tensor_count"] == 1600
    assert "selected_tensor_count" not in manifest["shards"]["model-00001-of-00282.safetensors"]
    assert run(source, output).returncode == 0


def test_rejects_changed_existing_output_and_bad_census(tmp_path: Path) -> None:
    source = tmp_path / "source-seal.json"
    output = tmp_path / "wave-manifest.json"
    value = source_seal()
    source.write_text(json.dumps(value))
    output.write_text("{}\n")
    assert run(source, output).returncode != 0
    output.unlink()
    value["target_tensor_count"] = 1
    source.write_text(json.dumps(value))
    assert run(source, output).returncode != 0
