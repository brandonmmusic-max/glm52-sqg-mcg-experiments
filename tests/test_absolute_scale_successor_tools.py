from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts.profile_search_shard import assigned_cell_ids
from scripts.rebind_absolute_gate_scale_fix import (
    EXPECTED_LAST_FIX_FILES,
    _rewrite_binding,
    _validate_last_fix,
)


PROJECT = Path(__file__).resolve().parents[1]


def _provenance(files: dict[str, dict[str, object]]) -> dict[str, object]:
    src: dict[str, object] = {}
    bmmlaw: dict[str, object] = {}
    for path, record in files.items():
        if path.startswith("src/"):
            src[path.removeprefix("src/")] = record
        else:
            bmmlaw[path.removeprefix("bmmlaw_r7_encoder/")] = record
    return {
        "src_tree": {"files": src},
        "bmmlaw_r7_encoder_tree": {"files": bmmlaw},
    }


def test_reviewed_last_fix_pair_is_exact_and_rejects_extra_code_change() -> None:
    old_files = {
        path: {"bytes": 1, "sha256": hashes["old_sha256"]}
        for path, hashes in EXPECTED_LAST_FIX_FILES.items()
    }
    new_files = {
        path: {"bytes": 2, "sha256": hashes["new_sha256"]}
        for path, hashes in EXPECTED_LAST_FIX_FILES.items()
    }
    changes = _validate_last_fix(_provenance(old_files), _provenance(new_files))
    assert {item["path"] for item in changes} == set(EXPECTED_LAST_FIX_FILES)
    old_files["src/unreviewed.py"] = {"bytes": 1, "sha256": "a" * 64}
    new_files["src/unreviewed.py"] = {"bytes": 2, "sha256": "b" * 64}
    with pytest.raises(ValueError, match="unexpected code changes"):
        _validate_last_fix(_provenance(old_files), _provenance(new_files))


def test_preparation_rebind_keeps_run_id_and_changes_only_preflight_id() -> None:
    value = {
        "binding": {
            "layer": 28,
            "run_id": "glm52-fresh-sqg-r1",
            "preflight_id": "old",
            "capture_manifest_sha256": "capture",
        }
    }
    _rewrite_binding(
        value,
        layer=28,
        old_preflight_id="old",
        new_preflight_id="new",
        run_id="glm52-fresh-sqg-r1",
    )
    assert value["binding"] == {
        "layer": 28,
        "run_id": "glm52-fresh-sqg-r1",
        "preflight_id": "new",
        "capture_manifest_sha256": "capture",
    }


def test_profile_workers_partition_exactly_16_cells_without_overlap() -> None:
    families = ("identity", "aggregate_rms", "quarter_rms", "inverse_quarter_rms")
    cells = [
        {"draw": draw, "cell_id": f"draw-{draw:02d}__{family}"}
        for draw in range(8)
        for family in families
    ]
    assignments = [assigned_cell_ids(cells, worker) for worker in range(4)]
    assert all(len(item) == 4 for item in assignments)
    assert len(set().union(*(set(item) for item in assignments))) == 16
    assert not any(
        set(assignments[left]) & set(assignments[right])
        for left in range(4)
        for right in range(left + 1, 4)
    )


def test_importer_validation_is_cpu_binding_only_and_orchestrator_stops_at_selection() -> None:
    importer = (PROJECT / "scripts/rebind_absolute_gate_scale_fix.py").read_text()
    orchestration = (PROJECT / "scripts/run_corrected_profile_search.sh").read_text()
    smoke = (PROJECT / "scripts/smoke_absolute_gate_scale.py").read_text()
    assert "_open_fast_sealed_runtime" not in importer
    assert 'device="cpu"' in importer
    assert "full_bf16_payload_hashing_performed\": False" in importer
    assert "SMOKE_LAYER = 28" in smoke and "SMOKE_EXPERT = 0" in smoke
    assert 'expected_bits = {"gate_proj": 4, "up_proj": 3}' in smoke
    assert "for worker in 0 1 2 3" in orchestration
    assert "Corrected profile search complete. Full final encoding was not launched." in orchestration
    assert "encode_final_shard.py" not in orchestration


def test_shell_orchestrators_parse() -> None:
    subprocess.run(
        [
            "bash",
            "-n",
            str(PROJECT / "scripts/prepare_corrected_successor.sh"),
            str(PROJECT / "scripts/run_corrected_profile_search.sh"),
        ],
        check=True,
    )

