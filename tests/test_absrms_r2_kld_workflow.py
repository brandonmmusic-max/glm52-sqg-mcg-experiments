from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "scripts/run_absrms_r2_kld.sh"
CANDIDATE = ROOT / "evaluation/run_fresh_kld.sh"
NATIVE = ROOT / "evaluation/run_native_mcg_control.sh"


def test_absrms_r2_workflow_has_valid_shell_syntax() -> None:
    for script in (WORKFLOW, CANDIDATE, NATIVE):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_workflow_pins_only_rejected_compiled_code_cache_as_seed() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "fresh-sqg-encode-absfix-r2" in source
    assert "fresh-sqg-encode-absrms-v2" not in source
    assert "fresh-sqg-evaluation-r1/candidate/fresh-sqg4-r1" in source
    assert 'REJECTED_CACHE="$REJECTED_OUT/run1-candidate-runtime-cache"' in source
    assert 'REJECTED_RECEIPT="$REJECTED_OUT/run1-runtime-cache-receipt.json"' in source
    assert 'REJECTED_RECORD="$REJECTED_OUT/run1-record.accepted.json"' in source
    assert "REJECTED_RECEIPT_SHA256=" in source
    assert "REJECTED_RECORD_SHA256=" in source
    assert ".file_count_after_boot == 13684" in source
    assert "compiled executable code only; no model, logits, KV, RNG, or process state" in source
    assert "cache overlay ABI differs" in source
    assert 'CANDIDATE_BOOT_CACHE_SEED="$REJECTED_CACHE"' in source
    assert 'NATIVE_BOOT1_CACHE_SEED="$REJECTED_CACHE"' in source


def test_candidate_uses_independent_seed_snapshot_for_every_fresh_boot() -> None:
    source = CANDIDATE.read_text(encoding="utf-8")
    loop = source[source.index('for run in $(seq 1 "$RUNS"); do') :]
    snapshot = loop.index('"$CANDIDATE_BOOT_CACHE_SEED" "$run_cache"')
    launch = loop.index("  docker run --rm", snapshot)
    assert snapshot < launch
    assert 'source_arm:"rejected_sqg_candidate_compiled_code_cache"' in loop
    assert "fresh_container_python_engine_workers_model_load:true" in loop
    assert "cache_payload_byte_hashing_skipped: true" in source
    assert "independent_snapshot_per_boot: true" in source
    assert "source_file_count != $expected_files" in source
    assert 'sudo -n cp -a --reflink=auto -- "$source_cache/."' in source
    assert 'cd "$run_cache"' not in loop
    assert "runtime-cache-files.sha256" not in loop


def test_both_arms_resume_from_consecutive_accepted_boots() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert 'resume_from_for_arm "$CANDIDATE_OUT" candidate' in workflow
    assert 'resume_from_for_arm "$NATIVE_OUT" native-mcg' in workflow
    assert 'quarantine_incomplete_boot "$output"' in workflow
    for source in (candidate, native):
        assert 'RESUME_FROM_RUN="${RESUME_FROM_RUN:-1}"' in source
        assert '[[ "$run" -ge "$RESUME_FROM_RUN" ]] || continue' in source
        assert "preserved_completed_runs" in source
        assert "$(((RESUME_FROM_RUN - 1) * 5))" in source


def test_workflow_can_stop_after_candidate_against_existing_baseline() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert '${CANDIDATE_ONLY:-0}' in source
    assert "existing_native_baseline" in source
    assert "fresh_native_boots_run:0" in source


def test_workflow_requires_five_by_five_dispatch_and_position_evidence() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert source.count("RUNS=5") == 2
    assert ".candidate_result.runs == 5" in source
    assert ".control_result.runs == 5" in source
    assert ".runtime_dispatch.native_mcg_dispatch_proved == true" in source
    assert ".runtime_dispatch.sqg_dispatch_records == 0" in source
    assert source.count(".positions == 2047") >= 4
    assert source.count(".independently_validated == true") >= 3
    assert "analyze_native_mcg_pair.py" in source
    assert "--expected-runs 5" in source
    assert "existing paired tensor hash differs" in source


def test_workflow_never_launches_or_restores_production_directly() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "docker run" not in source
    assert "docker start" not in source
    assert "glm-r33-fixed" not in source
    assert "reference_hf/reference-logits" not in source
    assert "bf16" not in source.lower()
