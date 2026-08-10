from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CONTROL_RUNNER = ROOT / "evaluation/run_native_mcg_control.sh"
CANDIDATE_RUNNER = ROOT / "evaluation/run_fresh_kld.sh"
CONTROL_VALIDATOR = ROOT / "evaluation/validate_native_mcg_control.py"
POSITION_VALIDATOR = ROOT / "evaluation/validate_per_position_kld.py"
NATIVE_ARGS = ROOT / "evaluation/r33_native_mcg_control.args"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _docker_env_keys(source: str) -> set[str]:
    # Compare only the model-inference launch.  The candidate runner also has
    # a separate, CPU-only containerized artifact validator whose sealed-code
    # environment is intentionally unrelated to the paired inference regime.
    run_loop = source.index('for run in $(seq 1 "$RUNS"); do')
    return set(
        re.findall(r'-e\s+(?:"?)([A-Z][A-Z0-9_]+)=', source[run_loop:])
    )


def test_native_control_runner_has_valid_shell_syntax_and_pinned_code() -> None:
    subprocess.run(["bash", "-n", str(CONTROL_RUNNER)], check=True)
    source = CONTROL_RUNNER.read_text(encoding="utf-8")
    assert f"CONTROL_VALIDATOR_SHA256={_digest(CONTROL_VALIDATOR)}" in source
    assert f"POSITION_VALIDATOR_SHA256={_digest(POSITION_VALIDATOR)}" in source
    assert f"NATIVE_ARGS_SHA256={_digest(NATIVE_ARGS)}" in source


def test_native_control_mounts_only_protected_model_read_only() -> None:
    source = CONTROL_RUNNER.read_text(encoding="utf-8")
    assert "[[ $# -eq 0 ]]" in source
    assert (
        "PROTECTED_MODEL=/home/brandonmusic/models/"
        "GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED" in source
    )
    assert '-v "$PROTECTED_MODEL:/model:ro"' in source
    assert "FRESH_ARTIFACTS_ROOT" not in source
    assert "FRESH_RUN_SEAL" not in source
    assert "candidate_mounted: false" in source
    assert '"$PRODUCTION_CONTAINER"' in source
    assert "production container is running" in source
    assert "NATIVE_MCG_RESULTS_ROOT is required" in source
    assert "glm52-native-mcg-kld-" in source


def test_native_control_is_exact_five_boot_same_regime_arm() -> None:
    control = CONTROL_RUNNER.read_text(encoding="utf-8")
    candidate = CANDIDATE_RUNNER.read_text(encoding="utf-8")
    assert 'RUNS="${RUNS:-5}"' in control
    assert '[[ "$RUNS" == 5 ]]' in control
    assert "run${run}-native-mcg-runtime-cache" in control
    assert "run${run}-native-mcg-hf-cache" in control
    assert "--context-length 2048 --stride 512 --max-windows 1" in control
    assert "--tensor-parallel-size 4" in control
    assert '--kv-cache-dtype "$KV_DTYPE"' in control
    assert "--attention-backend B12X_MLA_SPARSE" in control
    assert _docker_env_keys(control) == _docker_env_keys(candidate)


def test_native_control_requires_runtime_and_per_position_dispatch_proofs() -> None:
    source = CONTROL_RUNNER.read_text(encoding="utf-8")
    assert "VLLM_EXL3_NATIVE_MCG_CONTROL_LAYERS=6,28,52,77" in NATIVE_ARGS.read_text(
        encoding="utf-8"
    )
    assert "selected_mcg_markers == 3072" in source
    assert "selected_sqg_markers == 0" in source
    assert "retaining 768 exclusive MCG tensors on native nonfused dispatch" in source
    assert "reserving preserved R7 fused slot" in source
    assert (
        "R7 preserved fused accounting closed: actual=45 reserved=3 total=48 "
        "reserved_layers=[6, 28, 52]" in source
    )
    assert "entered SQG dispatch" in source
    assert '--per-position-output "$position_container"' in source
    validate_at = source.index('python3 "$POSITION_VALIDATOR" "$position_host"')
    accept_at = source.index('ACCEPTED_RECORDS+=( "$accepted_record" )')
    assert validate_at < accept_at
    assert '.native_mcg_dispatch_proved == true' in source
    assert '.sqg_dispatch_records == 0' in source


def test_runner_cleans_only_its_cid_and_never_adopts_named_container() -> None:
    source = CONTROL_RUNNER.read_text(encoding="utf-8")
    assert '--cidfile "$run_cidfile"' in source
    assert 'docker rm -f "$owned_cid"' in source
    assert 'docker rm -f "$CONTAINER_NAME"' not in source
    assert 'docker container inspect "$CONTAINER_NAME"' in source
