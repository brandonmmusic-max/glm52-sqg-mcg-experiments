from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.fresh_pipeline_calibration import ProfileScaleEvidence, make_shared_profiles
from src.fresh_pipeline_common import HADAMARD_BLOCK, HIDDEN, rademacher
from src.fresh_pipeline_evaluation import (
    document_scores_from_routed_aggregates,
    paired_document_bootstrap,
)
from src.fresh_pipeline_runner import (
    BONFERRONI_CONFIDENCE,
    PipelineSettings,
    PROFILE_FAMILIES,
    _build_profile_preregistration,
)
from src.run_fresh_sqg import main


def _scales() -> ProfileScaleEvidence:
    return ProfileScaleEvidence(
        gate_input_base=torch.linspace(0.015, 0.019, HIDDEN),
        down_output_base=torch.linspace(0.020, 0.014, HIDDEN),
        gate_block_quarter=torch.linspace(
            0.8, 1.2, HIDDEN // HADAMARD_BLOCK
        ),
        down_block_quarter=torch.linspace(
            1.2, 0.8, HIDDEN // HADAMARD_BLOCK
        ),
        evidence={"evidence_id": "fit-scale-evidence"},
    )


def test_profile_factorial_has_fresh_identity_control() -> None:
    scales = _scales()
    families = scales.families()
    assert tuple(families) == PROFILE_FAMILIES
    gate_magnitude, down_magnitude = families["identity"]
    gate_reference = scales.gate_input_base.mean()
    assert torch.allclose(gate_magnitude, torch.full((HIDDEN,), gate_reference))
    assert gate_magnitude.mean().item() == pytest.approx(0.017, abs=1e-7)
    assert not torch.allclose(gate_magnitude, torch.ones(HIDDEN))
    assert torch.equal(down_magnitude, torch.ones(HIDDEN))
    gate, down = make_shared_profiles(
        layer=6,
        draw=3,
        family="identity",
        scales=scales,
        sign_factory=lambda length, *parts: rademacher(
            length, "test-run", *parts
        ),
    )
    assert gate.side == "input" and down.side == "output"
    assert bool((gate.signs.abs() == 1).all())
    assert bool((down.signs.abs() == 1).all())
    assert "legacy_input=false" in gate.derivation


def test_gate_families_preserve_absolute_rms_and_down_families_stay_relative() -> None:
    scales = _scales()
    families = scales.families()
    gate_reference = scales.gate_input_base.mean().item()

    for family, (gate, down) in families.items():
        assert gate.mean().item() == pytest.approx(gate_reference, rel=1e-6)
        assert gate.max().item() < 0.03
        assert down.mean().item() == pytest.approx(1.0, rel=1e-6)
        if family != "identity":
            assert gate.std().item() > 0
            assert down.std().item() > 0

    assert torch.allclose(
        families["aggregate_rms"][0], scales.gate_input_base, rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (("profile_draws", 7), ("panel_experts", 15)),
)
def test_pipeline_settings_freeze_full_factorial_protocol(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="preregistered treatment"):
        PipelineSettings(run_id="test", **{field: value})


def _score(values: list[float]) -> dict[str, object]:
    return {
        "document_scores": [
            {"document_epoch": index, "relative_error": value}
            for index, value in enumerate(values)
        ]
    }


def test_paired_document_bootstrap_detects_uniform_improvement() -> None:
    result = paired_document_bootstrap(
        _score([0.5, 0.4, 0.3, 0.2]),
        _score([0.4, 0.3, 0.2, 0.1]),
        seed=17,
        iterations=1_000,
    )
    assert result["lower_bound_gt_zero"] is True
    assert result["observed_mean_improvement"] == pytest.approx(0.1)


def test_routed_score_squares_after_expert_sum_and_keeps_cross_terms() -> None:
    first_delta = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    second_delta = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    reference = np.ones((2, 2), dtype=np.float32)
    result = document_scores_from_routed_aggregates(
        first_delta + second_delta,
        reference,
        np.array([10, 10], dtype=np.int64),
    )
    # Row zero cancels across experts. Row one sums to 3, retaining the +4
    # cross term that would be lost by independently squaring 2 and 1.
    assert result["aggregate_error_energy"] == pytest.approx(9.0)
    assert result["aggregate_error_energy"] != pytest.approx(
        float(np.square(first_delta).sum() + np.square(second_delta).sum())
    )


def test_bootstrap_rejects_document_drift_and_preserves_improvement_sign() -> None:
    with pytest.raises(ValueError, match="document domains differ"):
        paired_document_bootstrap(
            _score([0.3, 0.2]),
            {"document_scores": [{"document_epoch": 9, "relative_error": 0.1}]},
            seed=1,
            iterations=1_000,
        )
    worse = paired_document_bootstrap(
        _score([0.1, 0.2, 0.3]),
        _score([0.2, 0.3, 0.4]),
        seed=2,
        iterations=1_000,
        confidence=BONFERRONI_CONFIDENCE,
    )
    assert worse["observed_mean_improvement"] < 0
    assert worse["lower_bound_gt_zero"] is False
    assert worse["confidence"] == pytest.approx(1.0 - 0.05 / 31)


def test_preregistration_uses_fit_panel_and_contains_all_32_exact_cells() -> None:
    class Capture:
        def __init__(self) -> None:
            self.roles: list[str] = []

        def role_gate_square_mass_by_expert(self, role: str) -> torch.Tensor:
            self.roles.append(role)
            return torch.arange(1, 257, dtype=torch.float64)

    capture = Capture()
    settings = PipelineSettings(run_id="preregistered-test")
    runtime = SimpleNamespace(
        layer=6,
        settings=settings,
        capture=capture,
        preflight={
            "preflight_id": "preflight",
            "source_seal": {"sha256": "source"},
            "capture": {"sha256": "capture"},
            "bit_contract": {"sha256": "bits"},
            "kquant": {"revision": "kquant"},
        },
    )
    prereg = _build_profile_preregistration(
        runtime,
        _scales(),
        SimpleNamespace(evidence_id="fit-h13"),
    )
    assert capture.roles == ["fit"]
    assert prereg["selection_panel_construction"] == (
        "fit_gate_square_mass_stratified_v1"
    )
    assert prereg["selection_used_for_panel_construction"] is False
    assert prereg["holdout_available_to_selection"] is False
    assert prereg["factorial"]["expected_cells"] == 32
    assert prereg["factorial"]["proxy_pruning_allowed"] is False
    assert len(prereg["cells"]) == 32
    assert len({cell["cell_id"] for cell in prereg["cells"]}) == 32
    assert prereg["bootstrap"]["multiplicity_correction"] == "bonferroni"
    assert prereg["bootstrap"]["comparisons"] == 31


def test_dry_run_is_inert_and_declares_all_32_cells(capsys) -> None:
    assert main(["dry-run", "--run-id", "unit-test"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["profile_factorial"]["cells_per_layer"] == 32
    assert value["profile_factorial"]["proxy_pruning"] is False
    assert value["model_workload_launched"] is False
    assert value["production_container_touched"] is False
    assert value["runnable_model_materialized"] is False
