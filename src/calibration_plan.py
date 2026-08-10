"""Sealed, document-disjoint calibration plan for the fresh SQG pilot.

The owner calibration selection is already fixed.  This module does not sample
another corpus.  It verifies that selection byte-for-byte, derives stable
document identities from the selected text, and assigns each complete document
to fit, selection, or holdout using the KQuant-style hash partition described
in :data:`SPLIT_POLICY`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence


OWNER_MANIFEST_SHA256 = (
    "b14c763fbc8feca6539f1411139fbfae58a7a906df5cbd5b9d9e104ab0475699"
)
OWNER_CAPTURE_FINGERPRINT = (
    "2efd10279b8c953e3e46a469d9ec9970593795859c7a2cebc98c9ea707115b51"
)
CORPUS_SHA256 = "cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4"
OWNER_DOCUMENTS = 4_497
OWNER_TOKENS = 1_050_468
MAX_SAMPLE_TOKENS = 4_096
MIN_SAMPLE_TOKENS = 8

ROLES = ("fit", "selection", "holdout")
ROLE_TO_ID = {name: index for index, name in enumerate(ROLES)}
EXPECTED_SPLIT = {
    "fit": {"documents": 2_638, "tokens": 601_343},
    "selection": {"documents": 892, "tokens": 219_650},
    "holdout": {"documents": 967, "tokens": 229_475},
}
SPLIT_POLICY = (
    "document_sha256_hex -> blake2b(digest_size=8) -> little-endian uint64 "
    "mod 5; buckets 0,1,2=fit, 3=selection, 4=holdout"
)
PLAN_SCHEMA = "glm52-fresh-sqg-document-plan-v1"


def sha256_file(path: str | Path, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def document_role(document_sha256: str) -> tuple[str, int]:
    """Return the fixed KQuant-style role and bucket for one document hash."""

    if (
        not isinstance(document_sha256, str)
        or len(document_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in document_sha256)
    ):
        raise ValueError("document_sha256 must be a lowercase SHA256 hex digest")
    digest = hashlib.blake2b(document_sha256.encode("ascii"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "little") % 5
    if bucket <= 2:
        return "fit", bucket
    return ("selection", bucket) if bucket == 3 else ("holdout", bucket)


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash token IDs in an explicit unsigned little-endian 32-bit ABI."""

    digest = hashlib.sha256()
    for token_id in token_ids:
        value = int(token_id)
        if not 0 <= value <= 0xFFFFFFFF:
            raise ValueError(f"token id {value} is outside uint32")
        digest.update(value.to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _validate_owner_manifest(path: str | Path) -> dict:
    path = Path(path)
    actual_sha = sha256_file(path)
    if actual_sha != OWNER_MANIFEST_SHA256:
        raise ValueError(
            f"owner calibration manifest SHA256 differs: {actual_sha} != "
            f"{OWNER_MANIFEST_SHA256}"
        )
    manifest = _read_json(path)
    claimed = manifest.get("capture_fingerprint")
    unhashed = dict(manifest)
    unhashed.pop("capture_fingerprint", None)
    actual_fingerprint = canonical_sha256(unhashed)
    if claimed != OWNER_CAPTURE_FINGERPRINT or claimed != actual_fingerprint:
        raise ValueError("owner calibration manifest fingerprint does not close")
    expected_scalars = {
        "schema": "glm52-b300-capture-plan-v1",
        "corpus_sha256": CORPUS_SHA256,
        "corpus_rows": 12_228,
        "max_sample_tokens": MAX_SAMPLE_TOKENS,
        "min_sample_tokens": MIN_SAMPLE_TOKENS,
        "total_tokens": OWNER_TOKENS,
        "owner_corpus_only": True,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"owner calibration manifest {key}={manifest.get(key)!r}, "
                f"expected {expected!r}"
            )
    if manifest.get("routing") != {
        "natural": True,
        "forced_expert_activation": False,
        "scoring_func": "sigmoid",
        "top_k": 8,
        "n_group": 1,
        "topk_group": 1,
    }:
        raise ValueError("owner calibration routing contract differs")
    passes = manifest.get("passes")
    if not isinstance(passes, list) or len(passes) != 4:
        raise ValueError("owner calibration manifest must contain four passes")
    return manifest


def load_owner_documents(
    owner_manifest_path: str | Path,
    corpus_path: str | Path,
) -> tuple[dict, list[dict]]:
    """Load and validate all 4,497 selected owner documents without tokenizing."""

    owner = _validate_owner_manifest(owner_manifest_path)
    corpus_path = Path(corpus_path)
    corpus_digest = sha256_file(corpus_path)
    if corpus_digest != CORPUS_SHA256:
        raise ValueError(f"owner corpus SHA256 differs: {corpus_digest}")
    raw_lines = corpus_path.read_text(encoding="utf-8").splitlines()
    if len(raw_lines) != int(owner["corpus_rows"]):
        raise ValueError(
            f"owner corpus line count {len(raw_lines)} != {owner['corpus_rows']}"
        )

    documents: list[dict] = []
    selected_lines: set[int] = set()
    selected_hashes: set[str] = set()
    for pass_index, pass_info in enumerate(owner["passes"]):
        if not isinstance(pass_info, dict):
            raise ValueError(f"owner pass {pass_index} is not an object")
        axis = pass_info.get("axis")
        samples = pass_info.get("samples")
        if not isinstance(axis, str) or not isinstance(samples, list):
            raise ValueError(f"owner pass {pass_index} has invalid axis/samples")
        pass_tokens = 0
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, dict) or set(sample) != {"line", "ntok"}:
                raise ValueError(
                    f"owner pass {pass_index} sample {sample_index} has invalid schema"
                )
            line = int(sample["line"])
            ntok = int(sample["ntok"])
            if not 0 <= line < len(raw_lines):
                raise ValueError(f"selected corpus line {line} is outside the corpus")
            if line in selected_lines:
                raise ValueError(f"owner calibration line {line} is selected twice")
            if not MIN_SAMPLE_TOKENS <= ntok <= MAX_SAMPLE_TOKENS:
                raise ValueError(f"owner calibration line {line} has invalid ntok={ntok}")
            selected_lines.add(line)
            try:
                record = json.loads(raw_lines[line])
            except Exception as exc:
                raise ValueError(f"owner corpus line {line} is invalid JSON") from exc
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"owner corpus line {line} has no string text")
            if record.get("axis") != axis:
                raise ValueError(
                    f"owner corpus line {line} axis {record.get('axis')!r} != {axis!r}"
                )
            text = record["text"]
            doc_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if doc_sha in selected_hashes:
                raise ValueError(
                    "owner selection contains duplicate document text; a whole-document "
                    f"split would leak {doc_sha}"
                )
            selected_hashes.add(doc_sha)
            role, bucket = document_role(doc_sha)
            documents.append(
                {
                    "epoch": len(documents),
                    "owner_pass_index": pass_index,
                    "owner_pass": str(pass_info.get("name")),
                    "axis": axis,
                    "sample_index": sample_index,
                    "corpus_line": line,
                    "tokens": ntok,
                    "document_sha256": doc_sha,
                    "split_bucket": bucket,
                    "role": role,
                    "role_id": ROLE_TO_ID[role],
                }
            )
            pass_tokens += ntok
        if pass_tokens != int(pass_info.get("tokens", -1)):
            raise ValueError(
                f"owner pass {pass_index} token sum {pass_tokens} != "
                f"{pass_info.get('tokens')}"
            )

    _validate_split(documents)
    return owner, documents


def _split_summary(documents: Iterable[dict]) -> dict[str, dict[str, int]]:
    result = {role: {"documents": 0, "tokens": 0} for role in ROLES}
    for document in documents:
        role = document.get("role")
        if role not in result:
            raise ValueError(f"invalid document role {role!r}")
        result[role]["documents"] += 1
        result[role]["tokens"] += int(document["tokens"])
    return result


def _validate_split(documents: Sequence[dict]) -> None:
    if len(documents) != OWNER_DOCUMENTS:
        raise ValueError(f"owner document count {len(documents)} != {OWNER_DOCUMENTS}")
    if sum(int(document["tokens"]) for document in documents) != OWNER_TOKENS:
        raise ValueError("owner selected token total differs")
    if _split_summary(documents) != EXPECTED_SPLIT:
        raise ValueError(
            f"whole-document split differs: {_split_summary(documents)} != "
            f"{EXPECTED_SPLIT}"
        )
    hashes = [str(document["document_sha256"]) for document in documents]
    lines = [int(document["corpus_line"]) for document in documents]
    epochs = [int(document["epoch"]) for document in documents]
    if len(set(hashes)) != len(hashes) or len(set(lines)) != len(lines):
        raise ValueError("document hashes and corpus lines must be unique")
    if epochs != list(range(len(documents))):
        raise ValueError("document epochs must be contiguous owner-order integers")
    for document in documents:
        role, bucket = document_role(str(document["document_sha256"]))
        if (
            document.get("role") != role
            or int(document.get("role_id", -1)) != ROLE_TO_ID[role]
            or int(document.get("split_bucket", -1)) != bucket
        ):
            raise ValueError(f"document epoch {document['epoch']} split assignment drifted")


def tokenizer_identity(model_root: str | Path, tokenizer: object) -> dict:
    root = Path(model_root)
    files: dict[str, str] = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    ):
        path = root / name
        if path.is_file():
            files[name] = sha256_file(path)
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "files_sha256": files,
    }


def build_document_plan(
    owner_manifest_path: str | Path,
    corpus_path: str | Path,
    model_root: str | Path,
    tokenizer: object,
) -> dict:
    """Tokenize and seal the fixed owner document list for later capture."""

    owner, documents = load_owner_documents(owner_manifest_path, corpus_path)
    identity = tokenizer_identity(model_root, tokenizer)
    if identity != owner.get("tokenizer"):
        raise ValueError(
            "capture tokenizer identity differs from the tokenizer that created "
            "the sealed owner plan"
        )
    raw_lines = Path(corpus_path).read_text(encoding="utf-8").splitlines()
    for document in documents:
        line = int(document["corpus_line"])
        text = json.loads(raw_lines[line])["text"]
        token_ids = list(tokenizer.encode(text))[:MAX_SAMPLE_TOKENS]
        if len(token_ids) != int(document["tokens"]):
            raise ValueError(
                f"tokenization drift at corpus line {line}: {len(token_ids)} != "
                f"{document['tokens']}"
            )
        document["token_ids_sha256_u32le"] = token_ids_sha256(token_ids)

    source_root = Path(model_root).resolve()
    source_identity = {
        "path": str(source_root),
        "config_sha256": sha256_file(source_root / "config.json"),
        "manifest_json_sha256": (
            sha256_file(source_root / "MANIFEST.json")
            if (source_root / "MANIFEST.json").is_file()
            else None
        ),
        "manifest_sha256_file_sha256": (
            sha256_file(source_root / "MANIFEST.sha256")
            if (source_root / "MANIFEST.sha256").is_file()
            else None
        ),
        "role": "read-only quantized calibration activation generator",
    }
    plan = {
        "schema": PLAN_SCHEMA,
        "owner_manifest_sha256": OWNER_MANIFEST_SHA256,
        "owner_capture_fingerprint": OWNER_CAPTURE_FINGERPRINT,
        "corpus_sha256": CORPUS_SHA256,
        "split_policy": SPLIT_POLICY,
        "split": EXPECTED_SPLIT,
        "documents_total": OWNER_DOCUMENTS,
        "tokens_total": OWNER_TOKENS,
        "token_id_abi": "uint32-little-endian",
        "tokenizer": identity,
        "activation_source": source_identity,
        "documents": documents,
    }
    plan["plan_fingerprint"] = canonical_sha256(plan)
    validate_document_plan(plan)
    return plan


def validate_document_plan(plan: dict) -> None:
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported fresh-SQG document plan schema")
    claimed = plan.get("plan_fingerprint")
    unhashed = dict(plan)
    unhashed.pop("plan_fingerprint", None)
    if claimed != canonical_sha256(unhashed):
        raise ValueError("fresh-SQG document plan fingerprint does not close")
    expected = {
        "owner_manifest_sha256": OWNER_MANIFEST_SHA256,
        "owner_capture_fingerprint": OWNER_CAPTURE_FINGERPRINT,
        "corpus_sha256": CORPUS_SHA256,
        "split_policy": SPLIT_POLICY,
        "split": EXPECTED_SPLIT,
        "documents_total": OWNER_DOCUMENTS,
        "tokens_total": OWNER_TOKENS,
        "token_id_abi": "uint32-little-endian",
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"fresh-SQG document plan {key} differs")
    documents = plan.get("documents")
    if not isinstance(documents, list):
        raise ValueError("fresh-SQG document plan has no document list")
    _validate_split(documents)
    for document in documents:
        token_hash = document.get("token_ids_sha256_u32le")
        if (
            not isinstance(token_hash, str)
            or len(token_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in token_hash)
        ):
            raise ValueError(
                f"document epoch {document['epoch']} lacks a sealed token-ID hash"
            )


def load_document_plan(path: str | Path) -> dict:
    plan = _read_json(path)
    validate_document_plan(plan)
    return plan


def load_plan_tokens(plan: dict, corpus_path: str | Path, tokenizer: object) -> list[list[int]]:
    """Retokenize every document and fail if a single ID or length drifted."""

    validate_document_plan(plan)
    corpus_path = Path(corpus_path)
    if sha256_file(corpus_path) != CORPUS_SHA256:
        raise ValueError("capture corpus differs from the sealed owner corpus")
    raw_lines = corpus_path.read_text(encoding="utf-8").splitlines()
    result: list[list[int]] = []
    for document in plan["documents"]:
        record = json.loads(raw_lines[int(document["corpus_line"])])
        text = record["text"]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != document["document_sha256"]:
            raise ValueError(f"document epoch {document['epoch']} text hash drifted")
        ids = list(tokenizer.encode(text))[:MAX_SAMPLE_TOKENS]
        if (
            len(ids) != int(document["tokens"])
            or token_ids_sha256(ids) != document["token_ids_sha256_u32le"]
        ):
            raise ValueError(f"document epoch {document['epoch']} tokenization drifted")
        result.append(ids)
    return result
