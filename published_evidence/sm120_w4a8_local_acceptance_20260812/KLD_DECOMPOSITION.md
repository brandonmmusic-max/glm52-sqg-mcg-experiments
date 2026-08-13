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

## 6. Determinism: verified real, and falsified against a frozen-output artifact

Repeat boots at fixed config reproduce the mean to all 17 digits
(`0.07583317451217256` x 5 boots, SD 0.0), which raised a fair suspicion of a
cached or frozen result. Two checks resolve it:

1. **Separate executions:** wall-clock differs per boot (57.273 s, 58.570 s ...)
   and `enable_prefix_caching=False` in the runner.
2. **Perturbation test:** forcing two prefill chunks instead of one
   (`max_num_batched_tokens` 2304 -> 1152) MOVES the result:

   | prefill | mean KLD | p99 | max |
   |---|---|---|---|
   | single chunk (2304) | 0.07583317451 | 1.3976 | 5.978 |
   | two chunks (1152) | 0.07650368874 | 1.4137 | 6.113 |

   Delta **+0.00067 (+0.88%)**. Receipt: `kld/chunkperturb_batched1152.json`.

Conclusion: computation is live; determinism is a property of the TP4/**DCP1**
configuration (no cross-rank LSE merge, batch=1, eager, no async scheduling),
whereas the historical SD ~0.0014 came from DCP4 where the cross-rank merge
varies reduction order. Note also that chunked prefill costs ~0.9% KLD, and the
reported baseline uses the single-chunk path — matching the historical runs
(`max_num_batched_tokens=2048` >= 2047 positions), so the comparison is not
flattered by chunking.

## 7. CORRECTIONS (adversarial review, 2026-08-13) — errors in sections above

Independent review found the following errors in MY earlier analysis. They are
corrected here rather than edited away.

1. **The "MCG r33 gate" row was a chimera.** I reported the target as
   `mean 0.0624499 / p99 1.0797 / CVaR-1% 1.8384`. Only the MEAN is MCG r33
   (`docs/fast_parallel_sqg_encoding.md:329`). The p99 and CVaR figures are the
   **SQG alpha=0 arm's** tails from `results/h13_blend_final_kld_r1.md`. I was
   gating against SQG's own tails while labelling them MCG's.
2. **Those tail thresholds are not discriminating.** The same-checkpoint null
   envelope (`results/contiguous_late_same_checkpoint_tail_null_r1.json`) reaches
   p99 **1.1146** and CVaR-1% **1.8736** — both ABOVE the thresholds I used. So
   my claim that this checkpoint "fails all three gates" is unsupported on the
   two tail criteria. **The defensible statement is the mean only: 0.0758 vs
   0.0624 = +21%.**
3. **`down_target_beta = 0.0625` was NOT a frozen fleet value.** It is a declared
   bootstrap *prior*, layer-77 scoped
   (`FULL_W4A8_BUILD_BINDING_AND_BETA_POLICY.md:5-11`;
   `SQG_REPRODUCIBILITY_BUILD_BINDING.json -> beta_policy.fleet_wide_layer77_beta
   = false`). The policy states "a bare numeric beta is not an admissible
   production input", which condemns 0.0625-for-all-76 as much as 0.25. The
   measured panel gap is **0.372%** on layer-77 fit SSE over 16 experts, and
   **beta=0 won more experts than 0.0625**. Beta cannot carry a material share of
   the ~9.5% residual.
4. **H13 alpha=0 makes the MEAN worse**, not better: 0.0628499 (alpha=0) vs
   0.0624617 (alpha=0.25). It buys tails (p99 -11.76%, CVaR -3.57%) at ~0.62%
   mean cost. Stated more clearly than before.
5. **A GLM coupled arm was already measured and LOST.** Selection NMSE
   0.00488459 vs 0.00366551 (**+33.26%**), holdout +35.10%, recorded verdict
   "NO-GO for promotion into the full build"
   (`docs/sqg_mcg_experiment_record.md:2794-2811`). My citation of
   -9.58%/-9.30% NMSE as support for the coupled arm was evidence substitution:
   those numbers belong to the **uncoupled** coordinate-corrected xterm repair
   that ALREADY SHIPS in this checkpoint.
6. **Disk figure wrong:** `/` has **460 GB** free (not 829 GB); `/mnt/toshiba`
   has 2.2 TB.
7. **My "no BF16 needed" verification was wrong.** `load_layer_hessians` returns
   only Gram matrices (`w13`, `w2`) — **there is no `B` in the Hessian bundle**,
   so `W* = H^-1 B` is not constructible from what was saved. The QSRT packer's
   quantization target is a dequantized official weight matrix, and the
   `--official-repo-dir` line I cited as an optionality guard is subprocess argv
   forwarding. A re-encode needs the official BF16 routed experts (~19.3 GB/layer).
8. I inspected a **stale QSRT tree** (local master 2 commits behind origin/master,
   557 lines changed in `scripts/pack_qsrt_candidates.py`), so my flag and
   line-number claims came from the wrong revision.
