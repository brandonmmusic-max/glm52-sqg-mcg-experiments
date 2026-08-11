from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from scripts.analyze_same_checkpoint_tail_null import analyze


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(root: Path, name: str, offset: float) -> Path:
    outputs = []
    values = []
    for run in range(1, 6):
        vector = np.linspace(0.0, 0.2, 2047, dtype=np.float32) + offset
        path = root / f"{name}-run{run}.safetensors"
        save_file(
            {"kld_ref_to_model": vector},
            path,
            metadata={
                "schema": "glm52-paired-position-kld-v1",
                "direction": "KL(ref||model)",
                "positions": "2047",
            },
        )
        values.append(float(vector.astype(np.float64).mean()))
        outputs.append(
            {
                "host_path": str(path),
                "sha256": _sha256(path),
                "positions": 2047,
                "tensor": "kld_ref_to_model",
                "independently_validated": True,
            }
        )
    payload = {
        "schema": "glm52-fresh-sqg-candidate-kld-result-v2",
        "candidate": "/candidate",
        "selected_sqg_layers": [74, 75, 76, 77],
        "runtime_image_id": "sha256:image",
        "runtime_overlay": {"sha256": "overlay"},
        "reference_sha256": "reference",
        "reference_token_ids_u32le_sha256": "tokens",
        "candidate_construction": {
            "run_seal_sha256": "seal",
            "candidate_manifest_sha256": "manifest",
        },
        "primary_same_base_image_baseline": {"summary_sha256": "baseline"},
        "regime": {
            "kv_cache_dtype": "fp8",
            "decode_context_parallel_size": 4,
            "dcp_comm_backend": "a2a",
            "dcp_kv_cache_interleave_size": 64,
        },
        "candidate_result": {
            "runs": 5,
            "values": values,
            "mean_kld": float(np.mean(values)),
            "paired_per_position_outputs": outputs,
        },
    }
    path = root / f"{name}-summary.json"
    path.write_text(json.dumps(payload))
    return path


def test_same_checkpoint_null_enumerates_all_balanced_partitions(tmp_path: Path) -> None:
    summary_a = _summary(tmp_path, "a", 0.01)
    summary_b = _summary(tmp_path, "b", 0.01)
    result = analyze(summary_a, summary_b)
    assert result["method"]["unique_unoriented_balanced_partitions"] == 126
    assert result["method"]["directional_partition_orientations"] == 252
    assert (
        result["observed_arbitrary_group_comparison"]["a_minus_b"]["mean_delta"]
        == 0.0
    )
    assert (
        result["balanced_partition_null_envelopes"]
        ["positive_delta_cvar_1pct"]["maximum"]
        == 0.0
    )


def test_same_checkpoint_null_reports_known_group_offset(tmp_path: Path) -> None:
    summary_a = _summary(tmp_path, "a", 0.011)
    summary_b = _summary(tmp_path, "b", 0.010)
    result = analyze(summary_a, summary_b)
    delta = result["observed_arbitrary_group_comparison"]["a_minus_b"]["mean_delta"]
    assert abs(delta - 0.001) < 1.0e-7
    assert result["observed_arbitrary_group_comparison"]["a_minus_b"][
        "fraction_left_lower"
    ] == 0.0
