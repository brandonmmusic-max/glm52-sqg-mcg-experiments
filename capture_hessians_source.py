#!/usr/bin/env python3
"""
capture_hessians.py — Pass A of the CALIBRATED GLM-5.2 tr3 encode.

Runs Luke's GLM-5.2-NVFP4 (ModelOpt, vLLM-loadable) over the OWNER'S calibration
corpus (/root/calib/calibration data/reap_recall_calib.jsonl — no substitutes)
and streams, per MoE layer L in 3..77:

    {capture_dir}/layer_{L:03d}/x.bin    raw bf16 bits  [N_tok, 6144]  MoE block input
    {capture_dir}/layer_{L:03d}/ids.bin  uint8          [N_tok, 8]     top-8 routed expert ids
    {capture_dir}/layer_{L:03d}/layer_manifest.json     tokens/bytes/sha256/routed counts
    {capture_dir}/capture_manifest.json                 corpus selection + run metadata

These are exactly sufficient for per-expert LDLQ in encode_tr3.py:
    gate/up:  H_e = X_e^T X_e  (X_e = x rows routed to expert e; [6144,6144])
    down:     I_e = silu(X_e Wg^T) * (X_e Wu^T)  ->  H_down_e = I_e^T I_e  [2048,2048]
              (per-K-slice r uses the [512r,512r+512) diagonal block)

DESIGN
  * vLLM 0.25 in its own venv (/root/venv-vllm) — the encode env (/venv/main,
    torch 2.11 + exllamav3) is never touched.  TP=4, enforce_eager=True so
    torch forward hooks fire on every token (no cudagraph replay).
  * Hooks: forward_pre_hook on every `model.layers.{L}.mlp.experts` (FusedMoE)
    module, TP rank 0 only (hidden states + gate are replicated across TP
    ranks).  The hook recomputes routing IN-WORKER with the model's own gate
    weights using the exact transformers-reference math for glm_moe_dsa
    (GlmMoeDsaTopkRouter): top8( sigmoid(x_f32 @ Wgate_f32^T) + e_score_bias ).
    n_group == topk_group == 1, so group masking is a no-op (asserted).
    This is deliberately independent of the `router_logits` kwarg, which in
    vLLM 0.25 may be the raw hidden states when FusedMoE.is_internal_router.
  * Hook installation goes through --worker-extension-cls (methods invoked by
    NAME via collective_rpc; no pickled callables, no insecure serialization).
  * Corpus subsampling is deterministic: seed 20260711, shuffle the full
    12,228-sample file, tokenize each `text` field RAW (some are JSON-encoded
    chat strings — fed as-is per owner directive), truncate to 4,096 tokens,
    accumulate until >= 1,048,576 tokens.
  * enable_prefix_caching=False so every prompt token is truly forwarded;
    max_tokens=1 so captured tokens == sum(prompt lens) EXACTLY (asserted
    per layer — any mismatch is a loud failure).
  * Disk: ~1M tok x 6144 x 2B x 75 layers ~= 922 GB + ids ~0.6 GB.

USAGE (box, inside /root/venv-vllm — see RUNBOOK_TR3.md):
    python /root/capture_hessians.py --verify            # mandatory preflight (~15 min, model load dominated)
    python /root/capture_hessians.py --capture           # full pass (resumable at whole-pass granularity)
    python /root/capture_hessians.py --status            # capture dir state

No KLD / quality evaluation anywhere.  HF token never read or logged here
(local model dir only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

# ----------------------------------------------------------------------------
# constants (owner-pinned)
# ----------------------------------------------------------------------------

DEF_SRC = "/root/models/luke-nvfp4"
DEF_CORPUS = "/root/calib/calibration data/reap_recall_calib.jsonl"
DEF_CAPTURE = "/root/tr3_capture"

SEED = 20260711
TARGET_TOKENS = 1_048_576
MAX_SAMPLE_TOKENS = 4_096
MIN_SAMPLE_TOKENS = 8

HIDDEN = 6144
NUM_EXPERTS = 256
TOPK = 8
FIRST_MOE_LAYER = 3
NUM_LAYERS = 78                      # main model layers 0..77 (78 = MTP, not loaded by vLLM)
MOE_LAYERS = list(range(FIRST_MOE_LAYER, NUM_LAYERS))

TP = 4
MAX_MODEL_LEN = 4_352                # 4096-token samples + margin
MAX_NUM_BATCHED_TOKENS = 8_192


def log(msg: str, logfile: str | None = None):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a") as f:
            f.write(line + "\n")


def sha256_file(path: str, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# vLLM worker extension — methods run INSIDE worker processes, invoked by name
# via llm.collective_rpc("tr3_*").  TP rank 0 does all capture work.
# ----------------------------------------------------------------------------

class CaptureWorkerExtension:

    def _tr3_rank(self) -> int:
        try:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            return get_tensor_model_parallel_rank()
        except Exception:
            return int(getattr(self, "rank", 0))

    def _tr3_model(self):
        mr = getattr(self, "model_runner", None)
        if mr is None:
            raise RuntimeError("worker has no model_runner")
        if hasattr(mr, "get_model"):
            return mr.get_model()
        return mr.model

    def tr3_capture_init(self, out_dir: str, verify_mode: bool) -> dict:
        """Install capture hooks on rank 0.  Returns discovery/audit info."""
        import torch
        import torch.nn.functional as F

        rank = self._tr3_rank()
        if rank != 0:
            return {"rank": rank, "hooks": 0}

        model = self._tr3_model()

        # --- discover MoE expert modules + their gates ------------------------
        experts_mods: dict[int, torch.nn.Module] = {}
        mlp_mods: dict[int, torch.nn.Module] = {}
        pat = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")
        for name, mod in model.named_modules():
            m = pat.search(name)
            if m:
                L = int(m.group(1))
                experts_mods[L] = mod
        pat_mlp = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp$")
        for name, mod in model.named_modules():
            m = pat_mlp.search(name)
            if m and int(m.group(1)) in experts_mods:
                mlp_mods[int(m.group(1))] = mod

        if sorted(experts_mods.keys()) != MOE_LAYERS:
            raise RuntimeError(
                f"expected MoE expert modules for layers {MOE_LAYERS[0]}..{MOE_LAYERS[-1]}, "
                f"found {sorted(experts_mods.keys())}")

        # --- per-layer gate weight (fp32) + e_score_correction_bias ----------
        self._tr3_gate_w: dict[int, torch.Tensor] = {}
        self._tr3_gate_b: dict[int, torch.Tensor] = {}
        seq_par = []
        for L, mlp in mlp_mods.items():
            gate = getattr(mlp, "gate", None)
            if gate is None or not hasattr(gate, "weight"):
                raise RuntimeError(f"layer {L}: mlp.gate with weight not found")
            w = gate.weight
            assert tuple(w.shape) == (NUM_EXPERTS, HIDDEN), (L, tuple(w.shape))
            b = getattr(gate, "e_score_correction_bias", None)
            if b is None:
                raise RuntimeError(f"layer {L}: gate.e_score_correction_bias missing")
            self._tr3_gate_w[L] = w.detach().to(torch.float32)
            self._tr3_gate_b[L] = b.detach().to(torch.float32).flatten()
            assert self._tr3_gate_b[L].shape == (NUM_EXPERTS,)
            if bool(getattr(mlp, "is_sequence_parallel", False)):
                seq_par.append(L)
        if seq_par:
            raise RuntimeError(
                f"sequence-parallel MoE enabled on layers {seq_par[:4]}... — rank-0 capture "
                f"would only see 1/TP of tokens; disable use_sequence_parallel_moe")

        # n_group / topk_group sanity (group masking must be a no-op)
        cfg = None
        for attr in ("config", "model_config"):
            cfg = getattr(model, attr, None) or cfg
        hf = getattr(cfg, "hf_config", cfg)
        ng = getattr(hf, "n_group", 1)
        tg = getattr(hf, "topk_group", 1)
        if not (ng in (None, 1) and tg in (None, 1)):
            raise RuntimeError(f"n_group={ng} topk_group={tg}: group-masked routing not "
                               f"implemented in capture (GLM-5.2 config says 1/1)")

        # --- open output files ------------------------------------------------
        os.makedirs(out_dir, exist_ok=True)
        self._tr3_files_x = {}
        self._tr3_files_i = {}
        self._tr3_counts = {L: 0 for L in MOE_LAYERS}
        self._tr3_routed = {L: [0] * NUM_EXPERTS for L in MOE_LAYERS}
        self._tr3_verify = verify_mode
        self._tr3_verify_cmp = {"checked_tokens": 0, "mismatch_tokens": 0, "kwarg_logits_layers": 0}
        for L in MOE_LAYERS:
            d = os.path.join(out_dir, f"layer_{L:03d}")
            os.makedirs(d, exist_ok=True)
            self._tr3_files_x[L] = open(os.path.join(d, "x.bin"), "wb", buffering=16 << 20)
            self._tr3_files_i[L] = open(os.path.join(d, "ids.bin"), "wb", buffering=1 << 20)

        # --- hook -------------------------------------------------------------
        self._tr3_handles = []

        def make_hook(L: int):
            gw = self._tr3_gate_w[L]
            gb = self._tr3_gate_b[L]

            def hook(module, args, kwargs):
                x = kwargs.get("hidden_states", None)
                if x is None and args:
                    x = args[0]
                assert x is not None and x.dim() == 2 and x.shape[1] == HIDDEN, \
                    f"layer {L}: unexpected MoE input {None if x is None else tuple(x.shape)}"
                n = x.shape[0]
                # routing: exact transformers GlmMoeDsaTopkRouter math (n_group=1)
                logits = F.linear(x.to(torch.float32), gw)
                sel = torch.sigmoid(logits) + gb
                ids = torch.topk(sel, TOPK, dim=-1, sorted=False).indices
                # verify-mode cross-check vs the router_logits kwarg when it is
                # real logits (shape [n, NUM_EXPERTS]) rather than hidden states
                if self._tr3_verify:
                    rl = kwargs.get("router_logits", None)
                    if rl is None and len(args) > 1:
                        rl = args[1]
                    if rl is not None and rl.dim() == 2 and rl.shape[1] == NUM_EXPERTS \
                            and rl.data_ptr() != x.data_ptr():
                        sel2 = torch.sigmoid(rl.to(torch.float32)) + gb
                        ids2 = torch.topk(sel2, TOPK, dim=-1, sorted=False).indices
                        s1, _ = ids.sort(dim=-1)
                        s2, _ = ids2.sort(dim=-1)
                        bad = (s1 != s2).any(dim=-1).sum().item()
                        self._tr3_verify_cmp["checked_tokens"] += n
                        self._tr3_verify_cmp["mismatch_tokens"] += bad
                        self._tr3_verify_cmp["kwarg_logits_layers"] += 1
                ids_u8 = ids.to(torch.uint8).cpu().contiguous()
                x_bf = x.detach().to(torch.bfloat16).cpu().contiguous()
                self._tr3_files_x[L].write(x_bf.view(torch.int16).numpy().tobytes())
                self._tr3_files_i[L].write(ids_u8.numpy().tobytes())
                self._tr3_counts[L] += n
                rc = self._tr3_routed[L]
                binc = torch.bincount(ids.flatten().to(torch.int64), minlength=NUM_EXPERTS)
                bl = binc.cpu().tolist()
                for e in range(NUM_EXPERTS):
                    rc[e] += bl[e]
                return None  # do not modify args

            return hook

        for L, mod in experts_mods.items():
            h = mod.register_forward_pre_hook(make_hook(L), with_kwargs=True)
            self._tr3_handles.append(h)

        return {
            "rank": 0,
            "hooks": len(self._tr3_handles),
            "layers": [MOE_LAYERS[0], MOE_LAYERS[-1]],
            "experts_module": type(next(iter(experts_mods.values()))).__name__,
            "gate_dtype": str(next(iter(mlp_mods.values())).gate.weight.dtype),
        }

    def tr3_capture_status(self) -> dict:
        if self._tr3_rank() != 0 or not hasattr(self, "_tr3_counts"):
            return {}
        c = self._tr3_counts
        return {"min": min(c.values()), "max": max(c.values()),
                "layer3": c.get(FIRST_MOE_LAYER, 0)}

    def tr3_capture_finalize(self) -> dict:
        if self._tr3_rank() != 0 or not hasattr(self, "_tr3_counts"):
            return {}
        for h in self._tr3_handles:
            h.remove()
        for f in self._tr3_files_x.values():
            f.flush(); os.fsync(f.fileno()); f.close()
        for f in self._tr3_files_i.values():
            f.flush(); os.fsync(f.fileno()); f.close()
        return {
            "counts": {str(L): self._tr3_counts[L] for L in MOE_LAYERS},
            "routed": {str(L): self._tr3_routed[L] for L in MOE_LAYERS},
            "verify_cmp": self._tr3_verify_cmp,
        }


# ----------------------------------------------------------------------------
# corpus selection (deterministic, owner's corpus only)
# ----------------------------------------------------------------------------

def build_prompts(corpus_path: str, tokenizer, target_tokens: int, logfile: str,
                  max_sample_tokens: int = MAX_SAMPLE_TOKENS):
    import random
    records = []
    with open(corpus_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                records.append((i, line))
    rng = random.Random(SEED)
    order = list(range(len(records)))
    rng.shuffle(order)

    chosen = []        # (line_idx, ntok)
    token_lists = []
    total = 0
    skipped_short = 0
    axis_hist: dict[str, int] = {}
    t0 = time.time()
    for oi in order:
        if total >= target_tokens:
            break
        li, raw = records[oi]
        rec = json.loads(raw)
        text = rec["text"]              # raw calibration text per owner directive
        ids = tokenizer.encode(text)
        if len(ids) < MIN_SAMPLE_TOKENS:
            skipped_short += 1
            continue
        ids = ids[:max_sample_tokens]
        token_lists.append(ids)
        chosen.append((li, len(ids)))
        axis_hist[rec.get("axis", "?")] = axis_hist.get(rec.get("axis", "?"), 0) + 1
        total += len(ids)
    if total < target_tokens:
        raise RuntimeError(f"corpus exhausted at {total} tokens < target {target_tokens}")
    log(f"corpus: {len(token_lists)} samples, {total} tokens "
        f"(target {target_tokens}, cap {max_sample_tokens}/sample, seed {SEED}, "
        f"skipped_short={skipped_short}, axes={axis_hist}, {time.time()-t0:.1f}s tokenize)", logfile)
    manifest = {
        "corpus_path": corpus_path,
        "corpus_sha256": sha256_file(corpus_path),
        "seed": SEED,
        "target_tokens": target_tokens,
        "max_sample_tokens": max_sample_tokens,
        "min_sample_tokens": MIN_SAMPLE_TOKENS,
        "num_samples": len(token_lists),
        "total_tokens": total,
        "skipped_short": skipped_short,
        "axis_histogram": axis_hist,
        "samples": [{"line": li, "ntok": nt} for li, nt in chosen],
    }
    return token_lists, manifest


# ----------------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------------

def make_llm(src: str):
    # workers must be able to import this module by name for the extension cls
    here = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = here + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    from vllm import LLM
    return LLM(
        model=src,
        tensor_parallel_size=TP,
        enforce_eager=True,                 # hooks must fire — no cudagraphs
        enable_prefix_caching=False,        # every prompt token truly forwarded
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        gpu_memory_utilization=0.90,
        worker_extension_cls="capture_hessians.CaptureWorkerExtension",
        seed=SEED,
    )


def run_pass(llm, token_lists, out_dir: str, verify_mode: bool, logfile: str,
             max_tokens: int = 1):
    from vllm import SamplingParams, TokensPrompt
    r = llm.collective_rpc("tr3_capture_init", args=(out_dir, verify_mode))
    info = [x for x in r if x.get("hooks")]
    assert len(info) == 1 and info[0]["hooks"] == len(MOE_LAYERS), f"hook install: {r}"
    log(f"hooks installed: {info[0]}", logfile)

    # ignore_eos so decode-forward count is exactly (max_tokens - 1) per seq —
    # token accounting below must be able to assert equality
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in token_lists]
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    n_prompt_tok = sum(len(t) for t in token_lists)
    exp = n_prompt_tok + len(token_lists) * (max_tokens - 1)
    log(f"generate done: {len(outs)} seqs, {n_prompt_tok} prompt tokens in {dt:.1f}s "
        f"({n_prompt_tok/max(dt,1e-9):.0f} tok/s incl. capture)", logfile)

    fin = llm.collective_rpc("tr3_capture_finalize")
    fin = [x for x in fin if x][0]
    counts = {int(k): v for k, v in fin["counts"].items()}
    bad = {L: c for L, c in counts.items() if c != exp}
    if bad:
        raise RuntimeError(
            f"captured token count mismatch (expected {exp}/layer): {dict(list(bad.items())[:5])} "
            f"— scheduler double-processed or skipped tokens; capture is INVALID")
    log(f"token accounting OK: {exp} tokens on all {len(counts)} layers", logfile)
    return outs, fin, exp


def write_layer_manifests(out_dir: str, fin: dict, tokens: int, logfile: str,
                          do_hash: bool):
    routed = {int(k): v for k, v in fin["routed"].items()}
    jobs = []
    for L in MOE_LAYERS:
        d = os.path.join(out_dir, f"layer_{L:03d}")
        xp, ip = os.path.join(d, "x.bin"), os.path.join(d, "ids.bin")
        assert os.path.getsize(xp) == tokens * HIDDEN * 2, f"{xp}: size mismatch"
        assert os.path.getsize(ip) == tokens * TOPK, f"{ip}: size mismatch"
        jobs.append((L, xp, ip))
    hashes = {}
    if do_hash:
        from multiprocessing import Pool
        t0 = time.time()
        with Pool(processes=16) as pool:
            res = pool.map(_hash_pair, jobs)
        hashes = {L: (hx, hi) for L, hx, hi in res}
        log(f"sha256 over capture files done in {time.time()-t0:.0f}s", logfile)
    for L, xp, ip in jobs:
        rc = routed[L]
        man = {
            "layer": L,
            "tokens": tokens,
            "hidden": HIDDEN,
            "x_dtype": "bfloat16",
            "x_bytes": tokens * HIDDEN * 2,
            "ids_topk": TOPK,
            "routed_counts": rc,
            "routed_min": min(rc), "routed_max": max(rc),
            "cold_experts_lt1024": [e for e, c in enumerate(rc) if c < 1024],
            "sha256_x": hashes.get(L, (None, None))[0],
            "sha256_ids": hashes.get(L, (None, None))[1],
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(os.path.join(out_dir, f"layer_{L:03d}", "layer_manifest.json"), "w") as f:
            json.dump(man, f)
    cold = {L: len([e for e, c in enumerate(routed[L]) if c < 1024]) for L in MOE_LAYERS}
    ncold = sum(cold.values())
    log(f"routed-count audit: {ncold} cold (<1024 tokens) expert slots across "
        f"{len(MOE_LAYERS)} layers (they will use layer-level H fallback)", logfile)


def _hash_pair(job):
    L, xp, ip = job
    return L, sha256_file(xp), sha256_file(ip)


def capture_complete(out_dir: str) -> bool:
    if not os.path.exists(os.path.join(out_dir, "capture_manifest.json")):
        return False
    for L in MOE_LAYERS:
        if not os.path.exists(os.path.join(out_dir, f"layer_{L:03d}", "layer_manifest.json")):
            return False
    return True


# ----------------------------------------------------------------------------
# modes
# ----------------------------------------------------------------------------

def mode_verify(args):
    logfile = args.log or "/root/tr3_capture_verify.log"
    out_dir = args.capture_dir + "_verify"
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    log(f"VERIFY: loading model TP{TP} eager (this is the vLLM-on-box check)", logfile)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.src)
    texts = [
        "The capital of France is",
        "Under Kentucky law, a claim for negligence requires proof of duty, breach,",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    ]
    token_lists = [tok.encode(t)[:64] for t in texts]

    llm = make_llm(args.src)
    outs, fin, exp = run_pass(llm, token_lists, out_dir, verify_mode=True,
                              logfile=logfile, max_tokens=32)
    for t, o in zip(texts, outs):
        log(f"  greedy: {t!r} -> {o.outputs[0].text!r}", logfile)

    import numpy as np
    import torch
    for L in (FIRST_MOE_LAYER, 40, NUM_LAYERS - 1):
        d = os.path.join(out_dir, f"layer_{L:03d}")
        x = np.fromfile(os.path.join(d, "x.bin"), dtype=np.int16).reshape(-1, HIDDEN)
        xt = torch.from_numpy(x).view(torch.bfloat16).float()
        ids = np.fromfile(os.path.join(d, "ids.bin"), dtype=np.uint8).reshape(-1, TOPK)
        assert xt.shape[0] == exp and ids.shape[0] == exp
        rms = xt.square().mean().sqrt().item()
        assert 1e-3 < rms < 1e3 and torch.isfinite(xt).all(), f"layer {L}: x rms {rms}"
        assert (np.sort(ids, axis=1)[:, 1:] != np.sort(ids, axis=1)[:, :-1]).all(), \
            f"layer {L}: duplicate expert id within a token"
        log(f"  layer {L}: x[{xt.shape[0]},{HIDDEN}] rms={rms:.4f}, ids OK "
            f"(uniq-8/token), routed min/max="
            f"{min(fin['routed'][str(L)])}/{max(fin['routed'][str(L)])}", logfile)

    cmp = fin["verify_cmp"]
    if cmp["checked_tokens"]:
        frac = 1.0 - cmp["mismatch_tokens"] / cmp["checked_tokens"]
        log(f"  routing cross-check vs vLLM router_logits kwarg: "
            f"{cmp['checked_tokens']} token-layer checks, match={frac:.6f}", logfile)
        assert frac >= 0.99, "routing recompute disagrees with vLLM router logits"
    else:
        log("  routing cross-check: router_logits kwarg was not real logits "
            "(is_internal_router) — recompute path is the only source (by design)", logfile)
    log("VERIFY PASSED", logfile)


def mode_capture(args):
    logfile = args.log or "/root/tr3_capture.log"
    out_dir = args.capture_dir
    if capture_complete(out_dir) and not args.fresh:
        log(f"capture already complete in {out_dir} (use --fresh to redo)", logfile)
        return
    if os.path.exists(out_dir):
        if not args.fresh and any(os.scandir(out_dir)):
            raise SystemExit(f"{out_dir} exists but is incomplete — rerun with --fresh "
                             f"(capture is all-or-nothing; ~1-2 h)")
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.src)
    token_lists, manifest = build_prompts(args.corpus, tok, args.target_tokens, logfile)

    llm = make_llm(args.src)
    outs, fin, exp = run_pass(llm, token_lists, out_dir, verify_mode=False,
                              logfile=logfile, max_tokens=1)
    del llm

    write_layer_manifests(out_dir, fin, exp, logfile, do_hash=not args.no_hash)
    manifest.update({
        "model": args.src,
        "captured_tokens_per_layer": exp,
        "moe_layers": [MOE_LAYERS[0], MOE_LAYERS[-1]],
        "tp": TP, "enforce_eager": True, "enable_prefix_caching": False,
        "vllm": _vllm_version(),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    with open(os.path.join(out_dir, "capture_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    sz = sum(os.path.getsize(os.path.join(r, fn))
             for r, _, fns in os.walk(out_dir) for fn in fns)
    log(f"CAPTURE COMPLETE -> {out_dir} ({sz/2**30:.1f} GiB, {exp} tokens/layer, "
        f"{len(MOE_LAYERS)} layers)", logfile)


def mode_status(args):
    out_dir = args.capture_dir
    if capture_complete(out_dir):
        with open(os.path.join(out_dir, "capture_manifest.json")) as f:
            m = json.load(f)
        print(f"COMPLETE: {m['captured_tokens_per_layer']} tokens/layer, "
              f"{m['num_samples']} samples, corpus {m['corpus_path']}")
    else:
        n = sum(os.path.exists(os.path.join(out_dir, f"layer_{L:03d}", "layer_manifest.json"))
                for L in MOE_LAYERS)
        print(f"INCOMPLETE: {n}/{len(MOE_LAYERS)} layer manifests in {out_dir}")


def _vllm_version():
    try:
        import vllm
        return vllm.__version__
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser(description="GLM-5.2 calibration capture (Pass A)")
    ap.add_argument("--verify", action="store_true", help="mandatory preflight on the box")
    ap.add_argument("--capture", action="store_true", help="full capture pass")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--src", default=DEF_SRC)
    ap.add_argument("--corpus", default=DEF_CORPUS)
    ap.add_argument("--capture-dir", default=DEF_CAPTURE)
    ap.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    ap.add_argument("--fresh", action="store_true", help="wipe an incomplete capture dir")
    ap.add_argument("--no-hash", action="store_true", help="skip post-run sha256 (not recommended)")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    if args.verify:
        mode_verify(args)
    elif args.capture:
        mode_capture(args)
    elif args.status:
        mode_status(args)
    else:
        raise SystemExit("pick one of --verify / --capture / --status")


if __name__ == "__main__":
    main()
