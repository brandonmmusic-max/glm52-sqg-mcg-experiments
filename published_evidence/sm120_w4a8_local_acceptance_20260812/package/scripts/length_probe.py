#!/usr/bin/env python3
"""Probe the live server with increasing prompt lengths to localize the
long-prefill fault (suspected index_topk-vs-seqlen boundary)."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9418"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "GLM-5.2-SQG-W4A8"
LENGTHS = [int(v) for v in (sys.argv[3].split(",") if len(sys.argv) > 3 else
           ["512", "1024", "1800", "2040", "2046", "2047", "2049", "2200", "3000"])]

WORD = "the "


def probe(target_tokens: int) -> tuple[str, int, str]:
    prompt = WORD * target_tokens
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        f"{URL}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            payload = json.load(response)
        return ("OK", payload["usage"]["prompt_tokens"], payload["choices"][0].get("text", "")[:20])
    except urllib.error.HTTPError as error:
        return (f"HTTP{error.code}", -1, error.read()[:120].decode(errors="replace"))
    except Exception as error:  # noqa: BLE001
        return (type(error).__name__, -1, str(error)[:120])


for target in LENGTHS:
    status, actual, detail = probe(target)
    print(f"target~{target:5d} actual={actual:5d} {status} {detail!r}", flush=True)
    if status != "OK":
        print("STOPPING: server likely dead at this length", flush=True)
        break
