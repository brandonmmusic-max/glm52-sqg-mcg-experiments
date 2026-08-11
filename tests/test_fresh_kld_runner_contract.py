from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "evaluation/run_fresh_kld.sh"
PREFILL = ROOT / "evaluation/prefill_kld_paired.py"
OVERLAY = ROOT / "evaluation/runtime_overlay"
RUNTIME_ARGS = ROOT / "evaluation/r33_exact_runtime.args"
RUNTIME_MOUNTS = ROOT / "evaluation/r33_exact_mounts.sha256"
WIKITEXT_CACHE = ROOT / "evaluation/wikitext_cache.sha256"
PURE_SQG_SOURCES = ROOT / "evaluation/pure_sqg_validator_sources.sha256"
PYDEPS = Path(
    "/home/brandonmusic/klc-linux/glm52_hybrid_opt/kld_eval_current/pydeps"
)


def test_fresh_kld_runner_has_valid_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_incomplete_run1_retry_is_fail_closed() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'RETRY_INCOMPLETE_RUN1="${RETRY_INCOMPLETE_RUN1:-0}"' in source
    assert "incomplete run-1 retry found accepted evidence" in source
    assert "incomplete run-1 retry found nonempty runs.jsonl" in source
    assert "refusing to retry a completed candidate output" in source


def test_fresh_kld_runner_binds_every_treatment_check_to_exact_eval_layers() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'SQG_EVAL_LAYERS="${SQG_EVAL_LAYERS:-6,28,52,77}"' in source
    assert "SQG_EVAL_LAYERS must be exactly four comma-separated integers" in source
    assert "SQG_EVAL_LAYERS must contain four unique ascending layers" in source
    assert 'SQG_EVAL_LAYERS_JSON="[$SQG_EVAL_LAYERS]"' in source
    assert source.count(
        'for sqg_layer in "${SQG_EVAL_LAYER_ARRAY[@]}"; do'
    ) == 2
    assert '-e FRESH_SQG_SELECTED_LAYERS="$SQG_EVAL_LAYERS"' in source
    assert 'SQG_TAIL_TRACE_LAYERS="${SQG_TAIL_TRACE_LAYERS:-$SQG_EVAL_LAYERS}"' in source
    assert "selected_overrides:$selected_overrides" in source
    assert "selected_sqg_layers: $selected_layers" in source
    assert "required_layers: $selected_layers" in source
    assert "records: $records" in source
    assert "selected_layers: $selected_layers" in source
    # The literal remains only as the backwards-compatible default and as the
    # provenance check for the original compiled-code cache seed.
    assert source.count("6,28,52,77") == 2
    assert "historical compiled-code cache" in source
    assert "SQG_EXPECTED_RESERVED_LAYERS_CSV=none" in source
    assert (
        "VLLM_EXL3_R7_EXPECT_RESERVED_LAYERS="
        "$SQG_EXPECTED_RESERVED_LAYERS_CSV" in source
    )


def test_fresh_kld_runner_is_bound_to_sealed_candidate_abi() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    invocation = source.index(
        'docker run --rm --network none --read-only',
    )
    manifest = source.index('> "$OUT/candidate-pure-sqg-preflight.json"', invocation)
    validation_block = source[invocation:manifest]
    for required in (
        '--source "$PRODUCTION_MODEL"',
        '--teacher-receipt "$TEACHER_RECEIPT_CONTAINER"',
        '--run-seal "$RUN_SEAL_CONTAINER"',
        '--artifacts-root /output',
        '--bit-contract "$BIT_CONTRACT_CONTAINER"',
    ):
        assert required in validation_block
    for required_mount in (
        '-v "$PROJECT_DIR:/work:ro"',
        '-v "$BF16_LAYERS_SOURCE:$BF16_LAYERS_CONTAINER:ro"',
        '-v "$CAPTURE_SOURCE:/capture:ro"',
        '-v "$SQG_EXTENSION_SOURCE:/sqg-extension:ro"',
        '-v "$R33_EXL3_EXT_DIR:/opt/exllamav3-r7ext:ro"',
        '-v "$ARTIFACTS_ROOT:/output:ro"',
        '-v "$MODELS_ROOT:$MODELS_ROOT:ro"',
    ):
        assert required_mount in validation_block
    assert '"$IMAGE" "$PURE_SQG_VALIDATOR_CONTAINER" "$CANDIDATE"' in validation_block
    assert "--network none" in validation_block
    assert "--read-only" in validation_block
    assert "FRESH_SQG_STATIC_EXLLAMAV3_ROOT" not in source
    assert "--layers" not in validation_block
    assert "CANDIDATE_PROVENANCE" not in source
    assert "candidate_provenance" not in source
    assert "LOCAL_CORRECTED_BUILD" not in source
    assert "EXPECTED_SQG_LAYERS" not in source
    assert '.selected_layer_codebook == "sqg_xor_cheb_t12"' in source
    assert ".selected_mcg_markers == 0" in source
    assert ".selected_legacy_inodes_reused == 0" in source
    assert ".selected_payload_hashes_verified == true" in source
    assert ".construction_exclusions == {" in source
    assert 'reject_result_overlap "candidate" "$CANDIDATE"' in source
    assert 'reject_result_overlap "production model" "$PRODUCTION_MODEL"' in source
    assert 'reject_result_overlap "fresh artifact tree" "$ARTIFACTS_ROOT"' in source
    assert 'BF16_LAYERS_CONTAINER="$BF16_LAYERS_SOURCE"' in source
    assert '-e FRESH_SQG_BIT_CONTRACT_SHA256=' in validation_block
    assert (
        "for validator_path_env in FRESH_SQG_BF16_MANIFEST "
        "FRESH_SQG_PLAN_CONTRACT; do" in source
    )
    assert '"${VALIDATOR_DYNAMIC_ENV_ARGS[@]}"' in validation_block


def test_every_accepted_run_requires_hashed_per_position_kld() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '-v "$run_output:/results:rw"' in source
    assert '--per-position-output "$position_container"' in source
    validate_at = source.index('python3 "$POSITION_VALIDATOR" "$position_host"')
    append_at = source.index('>> "$OUT/runs.jsonl"', validate_at)
    assert validate_at < append_at
    validation_block = source[validate_at:append_at]
    assert '--expected-sha256 "$position_sha256"' in validation_block
    assert "--expected-positions 2047" in validation_block
    assert '--expected-mean-kld "$position_mean"' in validation_block
    assert ".independently_validated == true" in source
    assert 'sha256sum -c "$OUT/per-position-evidence.sha256"' in source
    assert '"$run_log_sha256" "$run_log"' in source
    assert '"$dispatch_proof_sha256" "$dispatch_proof"' in source
    assert "sqg_dispatch_proved: true" in source
    assert "$((RUNS * 5))" in source
    assert "paired_per_position_outputs: map(.per_position)" in source
    assert "' \"${ACCEPTED_RECORDS[@]}\" > \"$OUT/summary.json.partial\"" in source
    assert "used_as_summary_input: false" in source
    assert "SELECTED_TREATMENT_FILES=()" in source
    assert 'DIRECTIONAL_TEST_FAST="${DIRECTIONAL_TEST_FAST:-0}"' in source
    assert "selected_treatment_files_rehashed_before_every_launch:" in source
    assert "($directional_test_fast | not)" in source
    assert 'sha256sum "${SELECTED_TREATMENT_FILES[$index]}"' in source
    loop_at = source.index('for run in $(seq 1 "$RUNS"); do')
    launch_at = source.index("  docker run --rm", loop_at)
    assert 'verify_sealed_eval_inputs' in source[loop_at:launch_at]


def test_pinned_prefill_runner_is_the_per_position_implementation() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    expected = hashlib.sha256(PREFILL.read_bytes()).hexdigest()
    assert f"FALLBACK_SHA256={expected}" in source
    prefill = PREFILL.read_text(encoding="utf-8")
    replace_at = prefill.index("os.replace(partial, output_path)")
    final_at = prefill.index('"fallback_prefill_kld_done"', replace_at)
    assert replace_at < final_at
    assert '"schema": "glm52-paired-position-kld-v1"' in prefill
    assert '"direction": "KL(ref||model)"' in prefill
    prompt_match = re.search(
        r"REFERENCE_TOKEN_IDS_U32LE_SHA256\s*=\s*\(\s*\"([0-9a-f]{64})\"",
        prefill,
    )
    assert prompt_match is not None
    assert (
        f"REFERENCE_TOKEN_IDS_U32LE_SHA256={prompt_match.group(1)}" in source
    )
    assert ".token_ids_u32le_sha256 == $token_sha256" in source


def test_pinned_runtime_overlay_manifest_is_current_and_self_consistent() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    manifest = OVERLAY / "SHA256SUMS.runtime-overlay"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert f"OVERLAY_MANIFEST_SHA256={digest}" in source
    subprocess.run(
        ["sha256sum", "-c", manifest.name],
        cwd=OVERLAY,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = {
        line.split(maxsplit=1)[1].removeprefix("./")
        for line in manifest.read_text(encoding="utf-8").splitlines()
    }
    expected.add(manifest.name)
    observed = {
        path.relative_to(OVERLAY).as_posix()
        for path in OVERLAY.rglob("*")
        if path.is_file()
    }
    assert observed == expected
    assert not any(path.is_symlink() for path in OVERLAY.rglob("*"))
    assert "runtime overlay exact file inventory differs" in source


def test_pinned_exact_runtime_args_are_current() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    digest = hashlib.sha256(RUNTIME_ARGS.read_bytes()).hexdigest()
    assert f"R33_EXTRA_ARGS_SHA256={digest}" in source


def test_pinned_exact_runtime_mount_bytes_are_current() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    digest = hashlib.sha256(RUNTIME_MOUNTS.read_bytes()).hexdigest()
    assert f"R33_MOUNT_EVIDENCE_SHA256={digest}" in source
    completed = subprocess.run(
        ["sha256sum", "-c", str(RUNTIME_MOUNTS)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.count(": OK\n") == 5
    assert "extension_directory_exact_single_file: true" in source


def test_paired_kld_dependency_tree_is_fully_pinned_and_offline() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "find . -type f -print0 | sort -z | xargs -0 sha256sum",
        ],
        cwd=PYDEPS,
        check=True,
        capture_output=True,
    )
    digest = hashlib.sha256(completed.stdout).hexdigest()
    count = completed.stdout.count(b"\n")
    assert f"PYDEPS_TREE_SHA256={digest}" in source
    assert f"PYDEPS_FILE_COUNT={count}" in source
    assert '-v "$PYDEPS:/deps:ro"' in source
    for setting in (
        "HF_HUB_OFFLINE=1",
        "HF_DATASETS_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
    ):
        assert setting in source


def test_runner_cannot_delete_or_adopt_an_existing_container() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '[[ "$CONTAINER_NAME" != "$PRODUCTION_CONTAINER" ]]' in source
    assert 'docker container inspect "$CONTAINER_NAME"' in source
    assert '--cidfile "$run_cidfile"' in source
    assert 'docker rm -f "$owned_cid"' in source
    assert 'docker rm -f "$CONTAINER_NAME"' not in source


def test_each_run_uses_isolated_runtime_and_seeded_hf_caches() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'run_cache="$OUT/run${run}-candidate-runtime-cache"' in source
    assert 'run_hf_cache="$OUT/run${run}-candidate-hf-cache"' in source
    assert '-v "$run_cache:/cache:rw"' in source
    assert '-v "$run_hf_cache:/hf-datasets:rw"' in source
    assert (
        '-v /home/brandonmusic/.cache/huggingface:'
        '/root/.cache/huggingface:ro' in source
    )
    assert '-e HF_DATASETS_CACHE=/hf-datasets' in source
    assert 'cp -a --reflink=auto' in source
    wikitext_digest = hashlib.sha256(WIKITEXT_CACHE.read_bytes()).hexdigest()
    assert f"WIKITEXT_CACHE_EVIDENCE_SHA256={wikitext_digest}" in source
    assert '/home/brandonmusic/.cache/glm52-tr3-release:/cache:rw' not in source
    assert '/home/brandonmusic/.cache/huggingface:/root/.cache/huggingface:rw' not in source


def test_validator_and_evidence_chain_are_fail_closed() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    pure_digest = hashlib.sha256(
        (ROOT / "evaluation/validate_pure_sqg_candidate.py").read_bytes()
    ).hexdigest()
    position_digest = hashlib.sha256(
        (ROOT / "evaluation/validate_per_position_kld.py").read_bytes()
    ).hexdigest()
    assert f"PURE_SQG_VALIDATOR_SHA256={pure_digest}" in source
    assert f"POSITION_VALIDATOR_SHA256={position_digest}" in source
    source_manifest_digest = hashlib.sha256(PURE_SQG_SOURCES.read_bytes()).hexdigest()
    assert f"PURE_SQG_SOURCE_EVIDENCE_SHA256={source_manifest_digest}" in source
    assert 'sha256sum -c "$PURE_SQG_SOURCE_EVIDENCE"' in source
    subprocess.run(
        ["sha256sum", "-c", str(PURE_SQG_SOURCES)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert 'if [[ "$docker_status" -ne 0 ]]; then' in source
    assert '"$docker_status" -ne 139' not in source
    assert "printf '%s  %s\\n' \"$position_sha256\" \"$position_host\"" in source
    assert '"$position_validation_sha256" "$position_validation"' in source
    assert 'sha256sum -c "$OUT/per-position-evidence.sha256"' in source
    assert 'sha256sum -c "$CANDIDATE_RUNTIME_EVIDENCE"' in source
    assert '-e "PYTHONPATH=/sqg-runtime-overlay"' in source
    assert 'PYTHONPATH=/sqg-runtime-overlay:/deps' not in source
