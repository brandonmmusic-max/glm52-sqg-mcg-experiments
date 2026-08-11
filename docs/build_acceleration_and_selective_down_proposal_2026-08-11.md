# Build acceleration and selective-down proposal, 2026-08-11

Status: owner-directed proposal. The owner has rejected the current build's
wall-clock ("encoding should not take a week") and directed that W4A8 quality
be addressed. This document proposes plan changes and a runtime quality
mechanism; it changes no sealed artifact and does not modify the active
binding. Adoption requires a new output root per the drift rule.

## Why the current design is slow: measured, not estimated

Raw encoding was never the problem: the sealed A16 four-layer encode ran in
15.5 minutes (70-72 experts/minute). The wall-clock lives in the search
superstructure added for the full build, measured from the active wave-1 and
layer-77 artifacts:

| Stage | Measured cost | Over 19 waves |
|---|---:|---:|
| 16-cell bootstrap profile search | ~2.5 h per 4 layers | ~12 GPU-days x fraction... dominant |
| Triplet scoring + exact-DP allocation | 51 min per layer (283 artifacts, L77) | ~65 GPU-hours |
| Per-layer beta panel | ~15-25 min per layer | ~25 GPU-hours |
| Selected encode | ~4-5 min per layer | ~6 GPU-hours |
| 4-layer fixed-point wave granularity | 19 serial waves | the serial chain itself |

## Proposed build-v3 changes (all seals and validators retained)

1. **Freeze the inherited per-tensor 384/384 K3/K4 map for build-v1; drop
   triplet scoring and DP allocation from the critical path.** The frozen map
   was the control variable for every accepted experiment in this repository;
   the topology-neutral contract is unchanged. Rate re-allocation becomes an
   optional post-build refinement wave, applied only where a later scored DP
   pass disagrees with the inherited map. Saves ~51 min/layer.
2. **Prune the bootstrap profile grid from 16 cells to the 4 sign draws x
   identity family, with a fallback trigger.** Identity has won 12 of 12
   completed layer selections across this program (layers 6, 28, 52, 74, 75,
   76, 77 draw-00/draw-03 era, and wave-1 layers 3, 4, 5, 6: draw-03, draw-02,
   draw-03, draw-03 — all identity). The 12 non-identity cells have never won
   a selection. Trigger: if the best identity cell's paired-document margin
   over draw-00 identity falls below the historical selection noise band, run
   the full 16-cell grid for that layer. Saves ~75% of the dominant stage.
3. **Widen fixed-point waves from 4 to 8 layers (19 waves -> 10).** The
   measured late-block propagation evidence (post-residual drift ratios
   1.04-1.29 over four adjacent converted layers, no amplification signature,
   tail-uncorrelated) supports 8-layer recapture granularity. The
   cannot-split-retroactively rule is preserved: an 8-layer wave is captured
   as 8 layers against the converted prefix.
4. **Beta fast path.** Keep the leakage-free per-layer beta choice, but when
   the fit/allocation panel selects the prior (beta = 0.0625) within the
   preregistered margin, reuse the bootstrap profile with the existing
   byte/SHA-256 proof and skip the second profile search (already in the
   canonical policy; stated here as the expected common path).
5. **No change** to: singleton encodes (batching stays rejected), capture
   contracts, H13 alpha 0.25 construction, candidate-conditioned (H,B) down,
   census/seal/receipt validators, or the DCP4 smoke.

Projected wall-clock: ~5-5.5 h per 8-layer wave x 10 waves ≈ **1.5-2.5 days**
of continuous GPU time, versus ~5-7 days for the current design. Nothing
completed is lost on adoption: wave-1 captures and all four completed
bootstrap profile selections predate any conversion and remain valid inputs
for a merged 3-10 first wave (layers 7-10 BF16 is already staged hot; its
capture, like 3-6's, observes the identical unconverted prefix).

6. **Add a per-wave progressive-checkpoint KLD (report-only).** ~30 min per
   wave. Inside repeat noise at wave 1; a statistically real trend line by
   waves 3-5 (12-40 converted layers). This is the cheapest protection against
   discovering a full-model quality failure at wave 10 instead of wave 3.

## W4A8 quality: per-layer selective down precision (runtime policy, zero re-encode)

The act-A8 damage is an execution-time choice, not an encode-time choice: the
same compact W4A8 trellis bytes serve both the A8 and the A16 down paths
(`GLM_HYBRID_DOWN_A8` demonstrated both arms over identical payloads).
Therefore per-layer down precision is a serving policy:

- Measure per-layer act-A8 functional damage with the Test 8b oracle
  (diagnostic schema, sealed late-block layers first; each layer costs
  ~10-15 min on one GPU).
- Rank layers by damage; serve down-A8 only on layers below an owner-chosen
  damage budget; the worst layers keep the A16 down path.
- Speed degrades gracefully: down is ~1/3 of MoE FLOPs, so keeping the worst
  20% of layers at A16-down retains roughly 93% of the full-W4A8 MoE speedup
  (1.75-1.82x measured) while capping the worst per-layer functional damage.
- This composes with, and does not replace, the coordinate-corrected (H,B)
  repair and the coupled down-suh line: every proxy point those recover
  shifts more layers under the budget.

Per-layer damage measurements on the sealed late block are running under the
diagnostic schema (`glm52-w4a8-activation-quality-perlayer-diagnostic-v1`);
results will be appended to `results/` when sealed. The policy needs no new
encoder work and no build delay: the build encodes full W4A8 everywhere, and
the per-layer A8/A16 decision is applied at serving with measured numbers.

## Honest limits

- The mantissa floor stands: full act-A8 at every layer cannot reach A16
  functional parity; the repair line plus selective policy bounds, not
  eliminates, the damage. Endpoint (KLD) adjudication of the residual remains
  required and is exactly what the per-wave KLD trend provides early.
- Cell-pruning and 8-layer waves trade a small preregistration-breadth loss
  for time; both carry explicit fallback triggers and the propagation
  evidence is from one block. The owner owns that trade.
