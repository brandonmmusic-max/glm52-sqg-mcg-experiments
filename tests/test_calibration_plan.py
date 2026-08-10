from __future__ import annotations

import hashlib
from pathlib import Path

from src.calibration_plan import (
    CORPUS_SHA256,
    EXPECTED_SPLIT,
    OWNER_CAPTURE_FINGERPRINT,
    OWNER_DOCUMENTS,
    OWNER_MANIFEST_SHA256,
    OWNER_TOKENS,
    _split_summary,
    document_role,
    load_owner_documents,
)


MODEL_ROOT = Path(
    "/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED"
)


def test_actual_owner_selection_and_whole_document_split_close() -> None:
    owner, documents = load_owner_documents(
        MODEL_ROOT / "calibration_manifest.json",
        MODEL_ROOT / "calibration" / "reap_recall_calib.jsonl",
    )

    assert owner["capture_fingerprint"] == OWNER_CAPTURE_FINGERPRINT
    assert owner["corpus_sha256"] == CORPUS_SHA256
    assert len(documents) == OWNER_DOCUMENTS
    assert sum(int(item["tokens"]) for item in documents) == OWNER_TOKENS
    assert _split_summary(documents) == EXPECTED_SPLIT
    assert len({item["document_sha256"] for item in documents}) == OWNER_DOCUMENTS
    assert len({item["corpus_line"] for item in documents}) == OWNER_DOCUMENTS
    assert [item["epoch"] for item in documents] == list(range(OWNER_DOCUMENTS))


def test_frozen_owner_and_corpus_digests_are_the_expected_files() -> None:
    assert hashlib.sha256(
        (MODEL_ROOT / "calibration_manifest.json").read_bytes()
    ).hexdigest() == OWNER_MANIFEST_SHA256
    assert hashlib.sha256(
        (MODEL_ROOT / "calibration" / "reap_recall_calib.jsonl").read_bytes()
    ).hexdigest() == CORPUS_SHA256


def test_document_split_is_content_based_and_deterministic() -> None:
    digest = hashlib.sha256(b"one immutable complete document").hexdigest()
    assert document_role(digest) == document_role(digest)
    role, bucket = document_role(digest)
    assert role in EXPECTED_SPLIT
    assert 0 <= bucket < 5

