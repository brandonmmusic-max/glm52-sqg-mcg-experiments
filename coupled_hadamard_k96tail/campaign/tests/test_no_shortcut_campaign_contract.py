from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _text(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_full_campaign_is_exact_k96_no_shortcut_and_parses() -> None:
    path = SCRIPTS / "run_full_coupled_3p0625_campaign.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text(encoding="utf-8")
    assert 'histogram == {"3": 672, "4": 96}' in text
    assert ".bpw == 3.125" in text
    assert ".kld-route-v1-k096.allocation.json" in text
    assert "SEALED_LAYER3_ROOT" in text
    assert "layer3_seal_passes" in text
    assert 'bit_census == {"k3": 720, "k4": 48, "total": 768}' in text
    assert "--worst-count 40" in text
    assert ".kld_route_policy.end_to_end_quality_claim == false" in text
    assert "no_b300_owner_speed_rescue == true" in text
    assert "run_coupled_no_shortcut_recipe.py" not in text
    assert "run_coupled_recipe_wave.sh" in text
    assert "validate_coupled_score_encode_parity.py" in text
    assert "run_finalize_k96tail_model.sh" in text


def test_recipe_launcher_uses_frozen_sqg_and_rank_private_triton_contract() -> None:
    text = _text("run_coupled_recipe_wave.sh")
    assert "src=$SOURCE_SQG_ROOT,dst=/source-sqg,readonly" in text
    assert "--source-sqg-root /source-sqg" in text
    assert "FRESH_SQG_RANK_PRIVATE_TRITON=1" in text
    assert "--qsrt-root /qsrt" in text
    assert "--final-profile-root /recipe/final_profiles" in text


def test_finalizer_requires_untrimmed_dcp1_kld_and_mtp3_inference() -> None:
    path = SCRIPTS / "run_finalize_k96tail_model.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text(encoding="utf-8")
    assert "--tensor-parallel-size 4" in text
    assert "--pipeline-parallel-size 1" in text
    assert "--decode-context-parallel-size 1" in text
    assert ".statistics.trim_fraction_per_side == 0.0" in text
    assert "--profile prod up -d server-prod" in text
    assert "MTP3_SMOKE" in text
    assert 'TOPOLOGY_ATTESTATION=tp1pp4dcp1' in text
    assert '--mtp3-smoke "$MTP3_SMOKE"' in text


def test_acceptance_gates_mean_p99_cvar_and_no_trimming() -> None:
    text = _text("seal_coupled_k96tail_release.py")
    for gate in (
        "full_untrimmed_mean_below_source",
        "p99_below_source",
        "cvar_worst_1pct_below_source",
        "no_nonfinite_positions",
        "no_positions_trimmed",
    ):
        assert gate in text
    assert '"mtp3_smoke_sha256"' in text
