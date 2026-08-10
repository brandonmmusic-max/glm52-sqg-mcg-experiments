from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "evaluation/run_fresh_kld.sh"
NATIVE = ROOT / "evaluation/run_native_mcg_control.sh"
SALVAGE = ROOT / "evaluation/salvage_completed_candidate_run1.sh"
CONTINUE = ROOT / "scripts/continue_cached_kld_after_run1.sh"


def test_cached_continuation_scripts_have_valid_shell_syntax() -> None:
    for path in (CANDIDATE, NATIVE, SALVAGE, CONTINUE):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_candidate_resume_preserves_run1_and_snapshots_each_warm_cache() -> None:
    source = CANDIDATE.read_text(encoding="utf-8")
    assert 'RESUME_FROM_RUN="${RESUME_FROM_RUN:-1}"' in source
    assert '[[ "$run" -ge "$RESUME_FROM_RUN" ]] || continue' in source
    assert 'ACCEPTED_RECORDS+=( "$previous_record" )' in source
    assert '"$OUT/run1-candidate-runtime-cache" "$run_cache"' in source
    assert 'sudo -n cp -a --reflink=auto -- "$source_cache/."' in source
    assert "independently_mutable_destination:true" in source
    assert "fresh_container_python_engine_workers_model_load:true" in source
    assert "byte_hashing_skipped:true" in source
    loop = source[source.index('for run in $(seq 1 "$RUNS"); do') :]
    assert "runtime-cache-files.sha256" not in loop
    assert 'cd "$run_cache"' not in loop


def test_native_uses_truthful_cross_arm_then_native_run1_cache_seeds() -> None:
    source = NATIVE.read_text(encoding="utf-8")
    assert 'NATIVE_BOOT1_CACHE_SEED="${NATIVE_BOOT1_CACHE_SEED:-}"' in source
    assert 'cache_seed_source="$NATIVE_BOOT1_CACHE_SEED"' in source
    assert 'cache_seed_source="$OUT/run1-native-mcg-runtime-cache"' in source
    assert "cross_arm_candidate_cache_seed:$source_arm_candidate" in source
    assert 'sudo -n cp -a --reflink=auto -- "$source_cache/."' in source
    assert "byte_hashing_skipped:true" in source
    assert "fresh_container_python_engine_workers_model_load:true" in source


def test_salvage_keeps_kld_dispatch_and_position_validation_as_gates() -> None:
    source = SALVAGE.read_text(encoding="utf-8")
    assert "fallback_prefill_kld_done" in source
    assert 'python3 "$POSITION_VALIDATOR" "$position_host"' in source
    assert "every_required_line_observed == true" in source
    assert "docker_exit_status_proved_zero_before_failed_cache_scan:true" in source
    assert ".docker_exit_status = 0" in source
    assert "byte_hashing_skipped:true" in source
    assert "run1-runtime-cache-files.sha256" in source
    assert "partial_byte_manifest_ignored" in source


def test_top_level_continuation_requires_exact_five_plus_five_and_pair() -> None:
    source = CONTINUE.read_text(encoding="utf-8")
    assert "RESUME_FROM_RUN=2" in source
    assert "SNAPSHOT_SEED_FROM_RUN1=1" in source
    assert "SNAPSHOT_SEED_RUNTIME_CACHE=1" in source
    assert 'NATIVE_BOOT1_CACHE_SEED="$CANDIDATE_OUT/run1-candidate-runtime-cache"' in source
    assert source.count("RUNS=5") == 2
    assert "analyze_native_mcg_pair.py" in source
    assert "--expected-runs 5" in source
