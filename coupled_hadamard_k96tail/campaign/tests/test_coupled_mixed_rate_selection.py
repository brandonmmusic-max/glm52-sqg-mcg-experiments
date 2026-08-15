from __future__ import annotations

import torch

from qsrt.qsrt_coupled import (
    CoupledHadamardSpec,
    coupled_execution,
    encode_coupled_weights,
)
from scripts.score_coupled_tail_triplet_candidates import (
    _execute_down,
    _execute_upstream,
)
from scripts.score_glm52_w4a8_activation_quality import (
    NativeProjection,
    apply_output_transform,
    native_label_gemm,
    prepare_down_operand,
    prepare_gate_up_operand,
)
from scripts.score_select_coupled_mixed_rate import (
    NUM_EXPERTS,
    _execute_full_w4a8_coupled,
    _select_draws,
)
from src.glm52_fresh_sqg.reference import normalized_hadamard


def _report(fit: list[tuple[float, float]], selection: list[tuple[float, float]]):
    reports = {}
    for draw, column in ((0, 0), (6, 1)):
        reports[draw] = {
            "roles": {
                "fit": {
                    "individual_experts": {
                        str(expert): {"relative_error": values[column]}
                        for expert, values in enumerate(fit)
                    }
                },
                "selection": {
                    "individual_experts": {
                        str(expert): {"relative_error": values[column]}
                        for expert, values in enumerate(selection)
                    }
                },
            }
        }
    return reports


def test_nonzero_draw_requires_fit_proposal_and_selection_confirmation() -> None:
    fit = [(1.0, 1.0)] * NUM_EXPERTS
    selection = [(1.0, 1.0)] * NUM_EXPERTS
    fit = list(fit)
    selection = list(selection)

    # Expert 0 wins on both folds and may select draw 6.
    fit[0] = (1.0, 0.8)
    selection[0] = (1.0, 0.9)
    # Expert 1 wins fit but loses confirmation, so it must fall back.
    fit[1] = (1.0, 0.8)
    selection[1] = (1.0, 1.1)
    # Expert 2 loses fit even though it wins confirmation, so it must fall back.
    fit[2] = (1.0, 1.1)
    selection[2] = (1.0, 0.8)

    selected, decisions = _select_draws(
        _report(fit, selection), draws=(0, 6)
    )

    assert selected[0] == 6
    assert selected[1] == 0
    assert selected[2] == 0
    assert decisions["0"]["proposed_draw"] == 6
    assert decisions["1"]["selected_draw"] == 0
    assert decisions["2"]["proposed_draw"] == 0


def test_ties_conservatively_select_identity_draw() -> None:
    tied = [(1.0, 1.0)] * NUM_EXPERTS
    selected, _ = _select_draws(_report(tied, tied), draws=(0, 6))
    assert set(selected.values()) == {0}


def test_selected_draw_scorer_uses_exact_coupled_native_w4a8_path() -> None:
    generator = torch.Generator().manual_seed(5202)
    hidden_width, intermediate = 512, 128
    spec = CoupledHadamardSpec(
        residual_block_size=512,
        preactivation_block_size=128,
        postactivation_block_size=128,
        residual_draw=0,
        intermediate_draw=6,
        activation="silu",
    )
    source = (
        torch.randn(intermediate, hidden_width, generator=generator),
        torch.randn(intermediate, hidden_width, generator=generator),
        torch.randn(hidden_width, intermediate, generator=generator),
    )
    execution = coupled_execution(encode_coupled_weights(source, spec), spec)

    def labels(rows: int, columns: int) -> torch.Tensor:
        values = torch.randn(rows, columns, generator=generator) * 0.05
        return values.to(torch.float8_e4m3fn).float()

    shared_suh = torch.ones(hidden_width, dtype=torch.float16)
    projections = {
        "gate_proj": NativeProjection(
            labels(hidden_width, intermediate),
            shared_suh,
            torch.ones(intermediate, dtype=torch.float16),
            3,
        ),
        "up_proj": NativeProjection(
            labels(hidden_width, intermediate),
            shared_suh.clone(),
            torch.ones(intermediate, dtype=torch.float16),
            4,
        ),
        "down_proj": NativeProjection(
            labels(intermediate, hidden_width),
            torch.ones(intermediate, dtype=torch.float16),
            torch.ones(hidden_width, dtype=torch.float16),
            3,
        ),
    }
    rows = torch.randn(3, hidden_width, generator=generator) * 0.1
    hadamard = normalized_hadamard(
        device=torch.device("cpu"), dtype=torch.float32, size=128
    )
    pair = (3, 4)
    middle = _execute_upstream(
        rows,
        execution,
        {pair: {key: projections[key] for key in ("gate_proj", "up_proj")}},
        hadamard,
        prepare_gate_up_operand,
        native_label_gemm,
        apply_output_transform,
    )[pair]
    expected = _execute_down(
        middle,
        projections["down_proj"],
        execution,
        hadamard,
        prepare_down_operand,
        native_label_gemm,
        apply_output_transform,
    )
    actual = _execute_full_w4a8_coupled(
        rows,
        execution,
        projections,
        hadamard,
        prepare_gate_up_operand=prepare_gate_up_operand,
        prepare_down_operand=prepare_down_operand,
        native_label_gemm=native_label_gemm,
        apply_output_transform=apply_output_transform,
        torch=torch,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
