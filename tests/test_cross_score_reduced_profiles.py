from __future__ import annotations

import numpy as np
import pytest

from scripts.cross_score_reduced_profiles import (
    FilteredLayerCapture,
    external_document_epochs,
)


def _plan(documents):
    return {"documents": documents}


def test_external_document_epochs_excludes_all_reduced_roles() -> None:
    full = _plan(
        [
            {"epoch": 0, "role": "selection", "document_sha256": "a"},
            {"epoch": 1, "role": "fit", "document_sha256": "b"},
            {"epoch": 2, "role": "selection", "document_sha256": "c"},
            {"epoch": 3, "role": "selection", "document_sha256": "d"},
        ]
    )
    reduced = _plan(
        [
            {"epoch": 100, "role": "fit", "document_sha256": "a"},
            {"epoch": 101, "role": "holdout", "document_sha256": "b"},
        ]
    )
    assert external_document_epochs(full, reduced, role="selection") == (2, 3)


def test_external_document_epochs_rejects_non_subset() -> None:
    full = _plan([{"epoch": 0, "role": "selection", "document_sha256": "a"}])
    reduced = _plan([{"epoch": 0, "role": "selection", "document_sha256": "x"}])
    with pytest.raises(ValueError, match="not a subset"):
        external_document_epochs(full, reduced, role="selection")


class _FakeCapture:
    layer = 77
    rows = 4
    role_ids = np.array([1, 1, 1, 2], dtype=np.uint8)
    doc_epochs = np.array([10, 11, 12, 13], dtype=np.uint32)
    topk_ids = np.array(
        [
            [5, 1, 2, 3, 4, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [5, 2, 3, 4, 6, 7, 8, 9],
            [5, 1, 2, 3, 4, 6, 7, 8],
        ],
        dtype=np.uint8,
    )
    topk_weights = np.full((4, 8), 0.25, dtype=np.float32)

    def binding(self):
        return {"fake": True}

    def load_hidden(self, rows, **kwargs):
        raise AssertionError("not used")


def test_filtered_capture_preserves_exact_rows_routes_and_epochs() -> None:
    view = FilteredLayerCapture(
        _FakeCapture(),
        source_role="selection",
        document_epochs=(10, 12),
        full_plan_sha256="f" * 64,
        reduced_plan_sha256="r" * 64,
    )
    assert view.role_rows("selection").tolist() == [0, 2]
    routed = view.routed_rows(5, "selection")
    assert routed.row_indices.tolist() == [0, 2]
    assert routed.route_slots.tolist() == [0, 0]
    assert routed.document_epochs.tolist() == [10, 12]
    assert view.binding()["document_count"] == 2


def test_filtered_capture_rejects_missing_epoch() -> None:
    with pytest.raises(ValueError, match="absent from capture"):
        FilteredLayerCapture(
            _FakeCapture(),
            source_role="selection",
            document_epochs=(10, 99),
            full_plan_sha256="f" * 64,
            reduced_plan_sha256="r" * 64,
        )
