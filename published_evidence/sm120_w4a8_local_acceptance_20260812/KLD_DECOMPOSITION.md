# GLM-5.2 SQG W4A8 — KLD decomposition (2026-08-12)

Measured full-vocabulary paired KLD(BF16 ref || model), 2,047 sealed positions,
vocab 154,880, TP4/DCP1, B12X_MLA_SPARSE, fp8 KV, block 64, eager, MTP off:

**mean 0.075833** | median 0.000612 | trimmed(0.5%/side) 0.062377 |
p95 0.380 | p99 1.398 | CVaR1% 2.207 | max 5.978 | 2047/2047 finite

## 1. Harness is exonerated (three independent confirmations)

| Measurement path | mean KLD |
|---|---|
| this package's `kld/run_kld_sm120.py` | 0.07583317451 |
| authentic historical `prefill_kld_fallback_hybrid.py` (sha `4fc1d276…` math) run unmodified in this image | 0.07583317439 |
| same, with `logprobs_mode="raw_logits"` (the exact r33 convention) | 0.07583317388 |

Agreement to 9 significant figures; residual is float summation order. Also:
100% full-vocabulary coverage (317,039,360/317,039,360 finite), zero non-finite
rows, no mod-64 chunk structure, no positional clustering.

Note: the on-disk `kld_eval_current/prefill_kld_fallback_hybrid.py` had drifted
from the hash the historical wrapper pins (`304c47bd…` vs `4fc1d276…`). The
authentic copy was recovered from
`glm52-mdispatch/hf_publish_local_corrected_v1/reproducibility/local-corrected-v1/eval/`
and diffed: the drift only ADDS `--result-json` receipt writing, per-position
collection, and a PP-size arg. **The KLD math is byte-identical.**

`return_prompt_logits` does not exist on this branch, so both runners use the
flat-logprobs path; the `raw_logits` run above proves that is equivalent.

## 2. Codec is correct — verified against the ORIGINAL published BF16

The `glm52_fresh_sqg.reference` decoder is NOT a usable oracle here: it misses
the original BF16 weight by **161%** on a tensor the runtime reproduces to
**4.07%**. Ground truth is therefore `zai-org/GLM-5.2` itself.

| Path | vs original BF16 (rel-RMSE) | receipt |
|---|---|---|
| K6 dense non-routed (`layers.3.mlp.shared_experts.down_proj`) | **0.0407** | `ground_truth_oracle.json` |
| routed expert FFN, A16 activations (K4/K4/K3) | **0.1321** | `a8_vs_a16_oracle` |
| routed expert FFN, production A8 activations | **0.1374** | `routed_ground_truth_oracle` |

Output RMS matches BF16 to 0.4% (0.07085 vs 0.07116), so there is no scale,
layout, or epilogue defect. The K6 endpoint also agrees **bit-exactly** with the
independent generic W4A16 dense path.

## 3. Where the 0.0758 comes from

Owner/codex campaign history on the same sealed reference:

* MCG r33 (comparable dense-K6 EXL3 artifact): **0.0624499**
* **SQG A16: 0.0624265 — parity with MCG.** The SQG weight codebook is not the problem.
* Full-W4A8 activation quantization measured **+10.22%/+10.86%** vs SQG A16
  → predicts ≈ **0.0692**.

Measured here: **0.0758**, i.e. ~9.5% above that prediction. Local A8 cost
measurement at the expert level is consistent and small: A8 adds only **4.0%**
relative on top of A16 (0.1374 vs 0.1321; direct A8-vs-A16 delta 4.05%), which
compounds across 75 routed layers plus the router.

Residual candidates for the remaining ~9.5% (none yet isolated):
1. **380 non-routed matrices converted to SQG K6**, a different dense/non-routed
   path than the MCG artifact used.
2. **`down_target_beta = 0.25` uniformly for every routed layer** — the expedited
   rescue path, not the documented per-layer beta selection (layer 77 had
   selected 0.0625). Encode-side; not fixable without re-encoding.
3. Runtime differences vs the r33 measurement stack (cu133/torch213 vs cu132)
   and DCP1-vs-DCP4 topology.

## 4. Branch defect: BF16 KV cache yields NaN

`B12X_MLA_SPARSE` advertises `bfloat16` in `supported_kv_cache_dtypes`, but a
full KLD run with `--kv-cache-dtype bfloat16` returns **mean_kld = NaN** over all
2,047 positions (receipt `kld_historical/kld_rawlogits_bf16kv.json`). fp8
(`fp8_ds_mla`) is therefore the only usable KV format on this path, and the
fp8-KV contribution to KLD could not be isolated by substitution.

## 5. Options to improve KLD (no re-encode required)

* **Keep `act` at A16 while leaving `h` at A8** (`h-A8/act-A16`). History
  attributes the dominant W4A8 damage to quantizing `act = SiLU(gate)*up` to
  E4M3. This requires editing the runtime's activation endpoint, which the
  checkpoint contract currently seals as `full-w4a8` — an explicit
  owner decision, not something to change silently.
* Ablate the 380 SQG-K6 non-routed matrices against the established dense
  carrier to size contributor (1).
* Re-measure once DCP4 is correct, to remove topology as a variable.

Encode-side (requires re-encode, explicitly out of scope for this run):
genuine per-layer beta/profile selection instead of the uniform 0.25.
