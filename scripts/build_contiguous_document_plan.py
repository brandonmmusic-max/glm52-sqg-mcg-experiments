#!/usr/bin/env python3
"""Build a deterministic whole-document subset of the sealed owner plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROLES = ("fit", "selection", "holdout")
TARGETS = {"fit": 150_000, "selection": 50_000, "holdout": 50_000}
SCHEMA = "glm52-fresh-sqg-contiguous-document-plan-v1"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-tokens", type=int, default=TARGETS["fit"])
    parser.add_argument("--selection-tokens", type=int, default=TARGETS["selection"])
    parser.add_argument("--holdout-tokens", type=int, default=TARGETS["holdout"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    target = {
        "fit": args.fit_tokens,
        "selection": args.selection_tokens,
        "holdout": args.holdout_tokens,
    }
    if any(value <= 0 for value in target.values()):
        raise ValueError("role token targets must be positive")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    selected: list[dict] = []
    totals = {role: 0 for role in ROLES}
    for document in source["documents"]:
        role = str(document["role"])
        if totals[role] >= target[role]:
            continue
        selected.append(dict(document))
        totals[role] += int(document["tokens"])
        if all(totals[name] >= target[name] for name in ROLES):
            break
    if any(totals[role] < target[role] for role in ROLES):
        raise ValueError(f"source plan cannot meet role targets: {totals} < {target}")
    # The runtime requires an exact monotonic request epoch.  Reassigning the
    # epoch changes no document identity or role; it only describes the order
    # of this preregistered subset capture.
    for epoch, document in enumerate(selected):
        document["epoch"] = epoch
    split = {
        role: {
            "documents": sum(document["role"] == role for document in selected),
            "tokens": sum(
                int(document["tokens"])
                for document in selected
                if document["role"] == role
            ),
        }
        for role in ROLES
    }
    plan = {
        key: source[key]
        for key in (
            "owner_manifest_sha256",
            "owner_capture_fingerprint",
            "corpus_sha256",
            "split_policy",
            "token_id_abi",
            "tokenizer",
            "activation_source",
        )
    }
    plan.update(
        {
            "schema": SCHEMA,
            "split": split,
            "documents_total": len(selected),
            "tokens_total": sum(int(document["tokens"]) for document in selected),
            "documents": selected,
            "subset_policy": {
                "source_plan_fingerprint": source["plan_fingerprint"],
                "selection": "source order, accept whole documents until each role target is met",
                "requested_role_token_minima": target,
                "holdout_used_for_encoding": False,
            },
        }
    )
    plan["plan_fingerprint"] = canonical_sha256(plan)
    atomic_json(args.output.resolve(), plan)
    print(json.dumps({"output": str(args.output.resolve()), "split": split, "tokens": plan["tokens_total"], "documents": len(selected), "fingerprint": plan["plan_fingerprint"]}, sort_keys=True))


if __name__ == "__main__":
    main()
