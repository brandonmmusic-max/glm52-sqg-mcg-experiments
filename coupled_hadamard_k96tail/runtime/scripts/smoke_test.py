#!/usr/bin/env python3
"""Deterministic 16-token API smoke test with a finite-output gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

PROMPT = "The Commonwealth of Kentucky's highest court is called the"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()

    body = {
        "model": args.model,
        "prompt": PROMPT,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "logprobs": 1,
    }
    request = urllib.request.Request(
        f"{args.url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)

    choice = payload["choices"][0]
    text = choice.get("text", "")
    logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
    tokens = (choice.get("logprobs") or {}).get("tokens") or []
    problems = []
    if len(tokens) != args.max_tokens and choice.get("finish_reason") == "length":
        problems.append(f"expected {args.max_tokens} tokens, got {len(tokens)}")
    if not text.strip():
        problems.append("empty completion text")
    nonfinite = [value for value in logprobs if value is None or not math.isfinite(value)]
    if nonfinite:
        problems.append(f"{len(nonfinite)} non-finite token logprobs")

    receipt = {
        "schema": "glm52-sqg-w4a8-sm120-smoke-v1",
        "prompt": PROMPT,
        "temperature": 0.0,
        "seed": 0,
        "completion_text": text,
        "tokens": tokens,
        "token_logprobs": logprobs,
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage"),
        "problems": problems,
        "pass": not problems,
    }
    if args.result_json:
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
