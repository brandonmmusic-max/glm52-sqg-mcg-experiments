from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from evaluation.analyze_native_mcg_pair import analyze_pair


POSITIONS = 2047
REFERENCE_SHA256 = "8" * 64
TOKEN_SHA256 = "9" * 64
IMAGE_ID = "sha256:" + "a" * 64
OVERLAY_SHA256 = "b" * 64
REGIME = {
    "kv_cache_dtype": "fp8",
    "rope": "bfloat16",
    "tensor_parallel_size": 4,
    "decode_context_parallel_size": 4,
    "dcp_comm_backend": "a2a",
    "dcp_kv_cache_interleave_size": 64,
    "context_tokens": 2048,
    "scored_positions": 2047,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _position_entry(path: Path, values: torch.Tensor) -> tuple[dict[str, object], float]:
    save_file(
        {"kld_ref_to_model": values},
        path,
        metadata={
            "schema": "glm52-paired-position-kld-v1",
            "direction": "KL(ref||model)",
            "positions": str(POSITIONS),
        },
    )
    mean = float(values.to(torch.float64).mean().item())
    return (
        {
            "host_path": str(path.resolve()),
            "sha256": _sha256(path),
            "positions": POSITIONS,
            "tensor": "kld_ref_to_model",
            "independently_validated": True,
        },
        mean,
    )


def _write_summaries(root: Path) -> tuple[Path, Path]:
    candidate_entries = []
    control_entries = []
    candidate_means = []
    control_means = []
    position = torch.linspace(-0.001, 0.001, POSITIONS, dtype=torch.float32)
    for run in range(5):
        control_values = 0.06 + position + run * 0.0001
        candidate_values = control_values - 0.002 + position * 0.25
        entry, mean = _position_entry(root / f"candidate-{run}.safetensors", candidate_values)
        candidate_entries.append(entry)
        candidate_means.append(mean)
        entry, mean = _position_entry(root / f"control-{run}.safetensors", control_values)
        control_entries.append(entry)
        control_means.append(mean)

    candidate = {
        "schema": "glm52-fresh-sqg-candidate-kld-result-v2",
        "reference_sha256": REFERENCE_SHA256,
        "reference_token_ids_u32le_sha256": TOKEN_SHA256,
        "regime": REGIME,
        "runtime": {
            "candidate_image_id": IMAGE_ID,
            "runtime_overlay": {"manifest_sha256": OVERLAY_SHA256},
        },
        "candidate_result": {
            "runs": 5,
            "values": candidate_means,
            "paired_per_position_outputs": candidate_entries,
        },
    }
    control = {
        "schema": "glm52-native-mcg-control-kld-result-v1",
        "reference_sha256": REFERENCE_SHA256,
        "reference_token_ids_u32le_sha256": TOKEN_SHA256,
        "regime": REGIME,
        "selected_layers": [6, 28, 52, 77],
        "runtime": {
            "image_id": IMAGE_ID,
            "runtime_overlay": {"manifest_sha256": OVERLAY_SHA256},
        },
        "runtime_dispatch": {"native_mcg_dispatch_proved": True},
        "control_result": {
            "runs": 5,
            "values": control_means,
            "paired_per_position_outputs": control_entries,
        },
    }
    candidate_path = root / "candidate-summary.json"
    control_path = root / "control-summary.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    control_path.write_text(json.dumps(control), encoding="utf-8")
    return candidate_path, control_path


def test_position_paired_analysis_publishes_bound_delta_tensor(tmp_path: Path) -> None:
    candidate, control = _write_summaries(tmp_path)
    tensor_output = tmp_path / "paired.safetensors"
    report = analyze_pair(
        candidate,
        control,
        tensor_output=tensor_output,
        bootstrap_iterations=100,
        bootstrap_block_size=32,
    )
    paired = report["paired_analysis"]
    assert paired["mean_delta"] == pytest.approx(-0.002, abs=1e-9)
    assert paired["mean_direction"] == "lower"
    assert paired["candidate_lower_fraction"] == 1.0
    assert len(paired["run_pair_mean_deltas"]) == 5
    assert report["positions"] == POSITIONS
    assert report["runs_per_arm"] == 5
    assert report["paired_tensor"]["sha256"] == _sha256(tensor_output)
    with safe_open(tensor_output, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {
            "candidate_minus_native_mcg_by_run",
            "candidate_minus_native_mcg_mean_by_position",
        }
        assert tuple(handle.get_tensor("candidate_minus_native_mcg_by_run").shape) == (
            5,
            POSITIONS,
        )
        assert tuple(
            handle.get_tensor("candidate_minus_native_mcg_mean_by_position").shape
        ) == (POSITIONS,)


def test_rejects_mismatched_reference_or_nonfive_arm(tmp_path: Path) -> None:
    candidate_path, control_path = _write_summaries(tmp_path)
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control["reference_sha256"] = "7" * 64
    control_path.write_text(json.dumps(control), encoding="utf-8")
    with pytest.raises(ValueError, match="reference_sha256 differs"):
        analyze_pair(
            candidate_path,
            control_path,
            tensor_output=tmp_path / "mismatch.safetensors",
            bootstrap_iterations=100,
        )
    with pytest.raises(ValueError, match="exactly five runs"):
        analyze_pair(
            candidate_path,
            control_path,
            tensor_output=tmp_path / "four.safetensors",
            expected_runs=4,
            bootstrap_iterations=100,
        )
