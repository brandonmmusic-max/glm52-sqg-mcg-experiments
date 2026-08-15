from __future__ import annotations

import json
from pathlib import Path
import py_compile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reproduction" / "k96tail-distributed-campaign.json"


def test_distributed_assignment_is_exact_and_nonoverlapping() -> None:
    campaign = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert campaign["schema"] == "glm52-coupled-k96tail-distributed-campaign-v1"
    local: list[int] = []
    remote: list[int] = []
    for assignment in campaign["assignments"]:
        start, end = assignment["layers"]
        target = local if assignment["worker"] == "local" else remote
        target.extend(range(start, end + 1))
    assert local == list(range(47, 51))
    assert sorted(remote) == list(range(51, 78))
    assert len(remote) == len(set(remote)) == 27


def test_rate_and_preservation_contract_is_explicit() -> None:
    campaign = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rates = campaign["rate_policy"]
    assert rates["layer_3"]["k3"] == 720
    assert rates["layer_3"]["k4"] == 48
    assert rates["layer_3"]["bits_per_weight"] == 3.0625
    assert rates["layers_4_through_77"]["k3"] == 672
    assert rates["layers_4_through_77"]["k4"] == 96
    assert rates["layers_4_through_77"]["bits_per_weight"] == 3.125
    assert rates["layer_78"]["policy"] == "preserve_source_mtp_unchanged"
    assert rates["layer_78"]["bits_per_weight"] == 3.5
    assert campaign["model"]["original_bf16_model_downloaded_for_reencode"] is False


def test_live_contract_has_explicit_final_placeholders() -> None:
    campaign = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert campaign["complete"] is False
    required = {
        "hub_model_commit",
        "assembly_manifest_id",
        "candidate_kld_sha256",
        "candidate_mean_kld",
        "candidate_p99_kld",
        "candidate_cvar_worst_1pct",
        "full_acceptance_sha256",
    }
    assert required <= campaign["final_results"].keys()
    assert all(campaign["final_results"][key] is None for key in required)


def test_audit_entrypoint_parses() -> None:
    py_compile.compile(
        str(ROOT / "scripts" / "audit_k96tail_campaign.py"), doraise=True
    )


def test_release_sealer_uses_target_layers_and_labels_mtp78_correctly() -> None:
    path = ROOT / "scripts" / "seal_coupled_k96tail_release.py"
    py_compile.compile(str(path), doraise=True)
    text = path.read_text(encoding="utf-8")
    assert "for layer in COUPLED_TARGET_LAYERS" in text
    assert "for layer in ROUTED_LAYERS" not in text
    assert '"layers_4_through_77_census"' in text
    assert '"layer_78_preserved_census"' in text
    assert "MTP layer 78: preserved source 384 K3 + 384 K4" in text


def test_method_document_is_linked_from_project_and_hub_cards() -> None:
    document = ROOT / "docs" / "K96_COUPLED_DISTRIBUTED_REPRODUCTION_20260814.md"
    assert document.is_file()
    text = document.read_text(encoding="utf-8")
    for required in (
        "Layer 3",
        "4--77",
        "MTP layer 78",
        "H512",
        "H128",
        "No-shortcut",
        "PENDING FINAL",
        "Unsafe remote score run-ahead overlap",
    ):
        assert required in text
    project_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    hub_readme = (ROOT / "hub" / "k96tail-staging" / "README.md").read_text(
        encoding="utf-8"
    )
    assert document.name in project_readme
    assert "PENDING FINAL" in hub_readme
