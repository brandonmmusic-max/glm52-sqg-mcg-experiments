# GLM-5.2 SQG full-W4A8 — SM120 local acceptance run (2026-08-12)

First end-to-end serving + KLD evidence for the published full-W4A8 checkpoint
`brandonmusic/GLM-5.2-SQG-W4A8` @ `593dd0d2de6f79ce4e65303930c22c75e1359d44`
(tag `accepted-glm52-sqg-w4a8-b300-r1`) on 4x RTX PRO 6000 Blackwell
(SM120, CC 12.0, PCIe, no NVLink).

## Headline result — this checkpoint does NOT beat MCG

| Metric | this W4A8 checkpoint | MCG r33 baseline | delta |
|---|---|---|---|
| **mean KLD** | **0.0758332** | 0.0624499 | **+21.4% worse** |
| p99 | 1.3976 | 1.0797 | +29.5% worse |
| CVaR worst-1% | 2.2071 | 1.8384 | +20.1% worse |
| median | 0.0006116 | 0.0006010 | +1.7% |
| max | 5.9780 | 3.7252 | +60.5% worse |

The campaign gate is `mean <= baseline AND p99 <= baseline AND CVaR-1% <=
baseline`. **This artifact fails all three.** Reported numbers are arithmetic
means over all 2,047 sealed positions, untrimmed, matching how every historical
receipt on this reference was computed. (The receipts also carry a
symmetrically-trimmed mean because the acceptance spec required that field; it
is NOT used for comparison here, and must not be — the regression *is* the
tail, so trimming deletes the very damage being measured.)

Measurement config: TP4/DCP1, `B12X_MLA_SPARSE`, fp8 (`fp8_ds_mla`) KV,
`block_size=64`, enforce-eager, MTP off, 2,047 positions, vocab 154,880, FP32
KLD math, reference `logits_sha256 = 87f992a689…fec063` (identical to all
EXL3-era receipts).

**Determinism:** repeat boots reproduce the mean to all 17 digits
(`0.07583317451217256` in `boot1` and `boot2`), so boot spread here is 0 —
unlike the historical 5-boot references which carried SD ≈ 0.0014.

## What is NOT the cause (each independently disproven)

1. **The harness.** Three independent paths agree to 9 significant figures:
   this package's runner (`0.07583317451`), the authentic historical
   `prefill_kld_fallback_hybrid.py` at its pinned hash `4fc1d276…` run
   unmodified in this image (`0.07583317439`), and the same with
   `logprobs_mode="raw_logits"` reproducing the exact r33 convention
   (`0.07583317388`). 100% full-vocabulary coverage
   (317,039,360/317,039,360 finite), zero non-finite rows, no mod-64 chunk
   structure, no positional clustering.
   *Note:* the working copy of that historical runner had drifted from the hash
   its wrapper pins (`304c47bd…`); the authentic copy was recovered from
   `glm52-mdispatch/hf_publish_local_corrected_v1/reproducibility/…/eval/` and
   diffed — the drift only ADDS receipt writing, the KLD math is byte-identical.
2. **The SQG weight codec.** Verified against the ORIGINAL published BF16
   (`zai-org/GLM-5.2`), not against a decoder: K6 dense non-routed is
   **4.07%** rel-RMSE; the routed K4/K4/K3 expert FFN is **13.21%** with A16
   activations and **13.74%** with production A8, output RMS matching BF16 to
   0.4% (so no scale/layout/epilogue defect). The K6 endpoint also agrees
   bit-exactly with the independent generic W4A16 dense path.
   **Important:** the `glm52_fresh_sqg.reference` decoder is NOT a usable
   oracle — it misses the true BF16 weight by **161%** on a tensor the runtime
   reproduces to 4.07%.
3. **Dependencies.** torch 2.13.0 and NCCL 2.31.2 match the branch pins
   (min 2.27.3); host driver 610.43.02 equals the image's own
   `CUDA_DRIVER_VERSION` with a matching compat shim; multicast is unsupported
   on CC 12.0 so NVLS/symmetric-memory paths correctly self-exclude.

## Where the regression actually lives

The bulk of the distribution is at parity (median +1.7%; 1,399 of 2,047
positions below 0.01 nat; only 33 above 1 nat) while the tail degrades
monotonically. The top 10 positions carry 18.5% of total KLD mass.

Attribution, from campaign history plus local measurement:

* **SQG A16 previously measured 0.0624265 vs MCG 0.0624499 — parity.** The SQG
  codebook is not the problem.
* **`act = SiLU(gate)*up` quantized to MXFP8/E4M3 is the dominant cost.**
  Campaign Test 8b (same weights, layer 77, all 256 experts): `h`-only A8
  +3.92% NMSE, **`act`-only A8 +19.18%**, both +22.98%; the corrected
  cross-term `(H,B)` down encoder shipped in this checkpoint reduces the total
  to +10.22%/+10.86%, predicting ≈0.0692. Measured 0.0758 is ~9.5% above that.
* Residual candidates, not yet isolated: the 380 non-routed matrices converted
  to SQG K6, and `down_target_beta = 0.25` applied uniformly to all 76 routed
  layers where the sealed fit-only panel froze **0.0625** (the shipped 0.25
  sits +0.372% above the fit minimum, and is also the H13 alpha value — a
  possible conflation worth verifying). Both are encode-side.
* The shipped **H13 blend alpha 0.25** is already on record as FAILING the
  preregistered tail constraints (p99 +11.76%, CVaR-1% +3.57%) while improving
  mean only 0.62% — and the tail is exactly what regressed here.

## Coupled Hadamard: assessed, not applicable

b12x's "coupled" refers to Kimi's QSRT **coupled P24/P33 pair-container**
profile — a weight-storage/rate-pairing format, not an activation-precision
improvement. Our own kernel states the distinction: *"Unlike Kimi's coupled
P24/P33 runtime profile, GLM keeps an independent K3/K4 decision for every
expert tensor."* It is also absent from this image (coupled-Hadamard QSRT
landed on b12x master `2312e1b77`/`7356a15f5` on Aug 10–11, after our b12x
parent `7cecbb2c` of Aug 9, and the overlay replaces b12x wholesale). Finally,
the activation-side coupling GLM needs is **already present**: `rotation_layout
= shared_h_v1`, one shared `r7_shared.gate_up_suh` across gate/up, a shared
`down_svh`, and `run_glm_down_input_transform` already Hadamard-rotates `act`
before MXFP8 quantization. The rotation is not missing; A8 still costs ~19%
despite it.

## Runtime defects found in the fork (dev/infernal-invocation @ ce5f50f6)

1. `mla_attention.py` DCP LSE-merge assumed a `.decode` metadata attribute that
   `B12xMLASparseMetadata` does not have — **patched** (`getattr` probe).
2. `v1/attention/ops/common.py` `mask_dcp_empty_shards_` crashed on
   zero-sequence batches (pure-prefill profile pass under DCP>1) — **patched**.
3. **DCP4 produces finite-but-wrong output**: KLD **9.974** and incoherent
   completions, versus 0.0758 and coherent output at DCP1 with identical
   weights/model/KV. Not yet root-caused. Contributing context: no test in the
   branch covers `B12X_MLA_SPARSE` + DCP>1; the sibling DeepSeek sparse indexer
   *fails closed* on `interleave > 1` under DCP with the comment "not yet
   validated end-to-end (gsm8k parity fails)" while `B12X_MLA_SPARSE` has no
   such guard; and the fork hardcodes `_head_major_mla_output = True` with no
   kill switch. **Leading untested hypothesis:** the Aug-9 receipted DCP4
   regime (`results/20260809T194958Z-kld-fp8-dcp4`, mean 0.06245, 5 boots) set
   six DCP env vars whose branch defaults are wrong — notably
   `VLLM_USE_B12X_DCP_A2A` (defaults False, killing the b12x DCP channel) and
   `VLLM_DCP_A2A_MAX_TOKENS` (defaults 0, so the ag_rs large-batch fallback
   never engages). Two attempts to reproduce that regime died at engine init
   with "No available memory for the cache blocks" (its a2a + CKV-gather
   workspaces do not fit the current envelope), so **the hypothesis remains
   untested, not disproven**.
4. **BF16 KV cache returns NaN.** `B12X_MLA_SPARSE` advertises `bfloat16` in
   `supported_kv_cache_dtypes`, but a full run yields `mean_kld = NaN` over all
   2,047 positions (matching a historical 9.982 result on the r33 lineage). fp8
   is the only usable KV format on this path, so the fp8-KV contribution to KLD
   cannot be isolated by substitution.

## Contents

* `receipts/kld/` — per-boot KLD receipts (full 2,047 per-position values,
  token-sequence hash, reference hashes, image id, kernel hashes + schedule,
  topology, backend, GPU/CC, library versions)
* `receipts/kld_historical/` — the authentic historical runner and the
  `raw_logits` / bf16-KV cross-checks
* `receipts/model_codec_validation.json` — 76 routed layers x 384/384 K3/K4
  census, 380 K6 dense, zero MCG/mul1 markers, HF revision hash proof
* `receipts/gpu_probe_layer3.json` — 4-rank TP4 reshard + native kernel probe
* `receipts/ground_truth_oracle.json`, `receipts/a8_vs_a16_oracle.json` — decode
  verification against original BF16
* `receipts/runtime_path_verification.json` — native W4A8 load+execute closure
  per worker, no A16/MCG fallback
* `KLD_DECOMPOSITION.md` — full analysis
* `package/` — the reproducible serving package (Dockerfile, compose, serve.sh,
  scripts, KLD runner, overlay manifest + provenance)

## Status

Serving package works and native full-W4A8 execution is proven at TP4/DCP1.
**This checkpoint is not yet a quality win over MCG and should not be published
as one.** The evidence points at `act`-A8 plus two encode-side shortcuts
(uniform beta 0.25, H13 alpha 0.25), not at the SQG codebook.
