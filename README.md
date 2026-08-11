# GLM-5.2 SQG versus MCG experiments

This repository preserves the code, methodology, compact evidence, and audited
results from the GLM-5.2 3.5-bpw experiments comparing the existing MCG
trellis representation with KQuant-style SQG. It now includes the signed
top-8, tail-constrained `H13` blend ablation, the preregistered contiguous-block
follow-up, and a separate QSRT/Kimi K3 K1 feasibility audit.

The experiment kept the production topology and the per-tensor K3/K4 bit
assignment fixed. Bits were **not** reallocated per expert. The initial
diagnostic layers were 6, 28, 52, and 77; the first contiguous block is layers
74--77. Every selected layer has 256 experts and three expert projections.

## Bottom line

The evidence supports continuing SQG development, but still does **not**
justify a full 75-layer conversion or establish lower full-model KLD.

| Measurement | Result | What it establishes |
|---|---:|---|
| Raw encoded-weight NMSE, 3,072 matched tensors | SQG `15.105%` lower than MCG | SQG is geometrically closer to BF16 under ordinary weight NMSE |
| Fixed 75%-local `H13`, held-out complete-expert NMSE | `5.985%` lower than shared-`H13` SQG; `7.951%` lower than MCG | Expert-local calibration better matches the isolated routed expert function |
| Direct E4M3 label endpoint, 3,072 late-block tensors | RNE-rounding regularized MCG LUT labels before Hadamard/scales increased error `4.515%`; SQG label-conversion SSE was zero | The final reconstructed MCG tensor was not rounded; this is not a W4A8 model result |
| Late 74--77 five-boot A16 KLD | `0.0624264841` vs r33 `0.0624498626` | `-0.03744%`, far inside repeat noise; no obvious scalar catastrophe and no demonstrated KLD win |
| Late alpha-0.25 signed top-8 holdout | `2.7705%` worse than MCG; `39.0295%` of positions improved | The separated-layer blend does not transfer cleanly to the late block |
| Same-candidate-path 5v5 null | mean-delta p95 envelope `±0.0013867` | Ten reused boots provide a checkpoint/prompt-specific empirical reference, not independent experiments or a formal false-positive calibration |
| Late paired trace versus one r33-r33 pair | MoE drift ratio `1.49–1.72x`; post-residual `1.04–1.29x` | Treatment-consistent but unreplicated mechanistic evidence; not a causal estimate or proof that compounding is absent |

The direct E4M3 result strengthens SQG's W4A8 architecture thesis, but its full
`16.8964%` SQG-versus-MCG E4M3 NMSE gap is not solely an endpoint benefit:
SQG was already `13.1439%` better at A16. No GLM A8 quality or tensor-core
speed result exists yet. The late block is a scalar no-catastrophe screen,
while its adverse frozen-hidden-state proxy shows that the `25%` local / `75%`
shared `H13` blend is not a fleet-wide calibration answer.

## Current Test 10 status

The late contiguous block, layers 74-77, is the first of the preregistered
early/middle/late blocks.

| Stage | Status |
|---|---|
| 217-document plan | Sealed: 253,863 whole-document rows per layer; 150,368 fit, 51,232 selection, 52,263 holdout |
| Routed capture | Complete and sealed for layers 74-77 |
| Four-GPU layer preparation | Complete: 256 permutations/layer, fit-only, zero MCG inputs, zero fallback |
| Profile search | Complete: all 16 preregistered cells/layer; selected identity draw 0 for layers 74-76 and identity draw 3 for layer 77 |
| Alpha-0.25 encoding | Complete: alpha is 25% expert-local/75% layer-global; 1,024 experts and 3,072 SQG tensors |
| Rate/codebook census | Complete: 1,536 K3 + 1,536 K4; zero MCG tensors in treatment layers |
| Candidate materialization | Complete and separately sealed without mutating the protected source model |
| First KLD attempt | Excluded: failed before inference on a stale historical reserved-layer assertion; zero accepted records and no KLD result |
| Corrected five-boot A16 KLD | Complete: mean `0.0624264841`, `-0.03744%` versus r33, repeat-noise inconclusive |
| Same-candidate-path null | Complete: a second five-boot set and all balanced 5v5 partitions provide a prompt/runtime empirical reference; the identity record does not independently bind full checkpoint bytes |
| Routing/residual trace | Complete with one unchanged-r33 pair; treatment-consistent but unreplicated, with no causal compounding conclusion |
| Late signed top-8 holdout | Complete: alpha 0.25 was `2.7705%` worse than MCG; `97.7397%` of the regression was individual-expert SSE |
| Late-specific alpha panel | **Active and incomplete**: alpha `0`, `0.5`, `0.75`, and `1.0` arms are being encoded/resumed; no selection or holdout result exists yet |

The initial failure remains excluded. A later run-3 validator rejection exposed
a float32-roundoff floor that was smaller than one machine epsilon; the saved
position vector was revalidated under a pinned two-epsilon floor without
changing model, logits, or tensor bytes. Five accepted boots then completed.

Test 10 establishes scalar KLD parity/no obvious scalar catastrophe for one
late four-layer A16 block; its matched per-position tail gate remains unclosed,
and this is not a general SQG quality win. The active late-specific alpha panel
must complete before a calibration choice is made. Middle layers 38--41, early
layers 10--13, and all W4A8 quality/speed endpoints remain pending.

The late null does not numerically clear Test 9's separated-layer tail failure:
it is not a matched-checkpoint null for Test 9, and those harmful-tail metrics
exceed even the late p95 reference.

## Start here

- [Complete experiment ledger](docs/sqg_mcg_experiment_record.md) — methodology,
  holdouts, assumptions, discovered errors, validation, results, and the next
  hypothesis for every test.
- [Raw encoded NMSE report](results/raw_encoded_nmse_sqg_vs_mcg.md)
- [Hessian-weighted NMSE report](results/hessian_weighted_nmse_sqg_vs_mcg.md)
- [Expert-local calibration report](results/recalibrated_sqg_vs_mcg.md)
- [Paired KLD analysis](results/h13e_vs_current_sqg_kld.json)
- [Test 9 alpha-blend selection](results/h13_blend_selection_r1.md)
- [Test 9 encoder-unseen holdout](results/h13_blend_holdout_alpha025_r1.md)
- [Test 9 final-logit KLD](results/h13_blend_final_kld_r1.md)
- [Test 9 routing/residual trace](results/h13_blend_tail_trace_alpha025_r1.md)
- [Test 10 compact late-block evidence](published_evidence/contiguous_late/README.md)
- [Direct E4M3 endpoint result](results/e4m3_endpoint_distortion_late_r1.md)
- [Late signed top-8 holdout result](results/signed_top8_contig_late_holdout_r1.md)
- [Same-checkpoint KLD null](results/contiguous_late_same_checkpoint_tail_null_r1.md)
- [Late trace analysis](results/contiguous_late_tail_trace_a025_r1.md)
- [Same-checkpoint trace null](results/contiguous_late_same_checkpoint_tail_trace_null_r1.md)
- [QSRT/Kimi K3 K1 feasibility audit](docs/qsrt_kimi_k3_k1_feasibility.md)
- [Fast parallel encoding notes](docs/fast_parallel_sqg_encoding.md)
- [Publication scope and excluded payloads](PUBLISHING_SCOPE.md)
- [Publication validation](PUBLICATION_VALIDATION.md)
- [Original implementation README](docs/implementation_readme_historical.md)

Machine-readable results live in [`results/`](results/), experiment contracts
in [`contracts/`](contracts/), compact execution evidence in
[`published_evidence/`](published_evidence/), and runnable analysis/encoding
code in [`scripts/`](scripts/), [`src/`](src/), and
[`bmmlaw_r7_encoder/`](bmmlaw_r7_encoder/).

## KQuant provenance

[`kquant/`](kquant/) is a vendored working tree based on
[`local-inference-lab/kquant`](https://github.com/local-inference-lab/kquant) at
commit `104dd9233f850a3955f4991bea68b07dd34deeb8`. It includes the local SQG
integration changes used here. See [KQuant local
provenance](kquant/LOCAL_PROVENANCE.md) and
[`kquant/LOCAL_CHANGES.patch`](kquant/LOCAL_CHANGES.patch).

## Important limits

- Raw and Hessian-weighted NMSE are proxies, not model-level KLD.
- The final KLD comparison uses one fixed 2,047-position prompt and five boots
  per arm. It does not estimate document-level or task-level generalization.
- Test 9's holdout was encoder-unseen but had already been inspected during
  Test 7, so it is not a globally blind confirmation set.
- The earlier four selected layers were separated; the new 74--77 block is
  contiguous but covers only 5.33% of the 75 routed layers.
- The MCG baseline and SQG candidate did not share the same selected-layer
  dispatch path. The preserved `+0.64057%` four-layer SQG-versus-r33 result is
  consequently a weight-plus-dispatch result, not a codebook-only result.
- This Git repository intentionally excludes model-scale tensor payloads and
  regenerated caches. Their manifests, hashes, receipts, and compact logs are
  preserved in `published_evidence/`.
- Direct MCG-versus-SQG E4M3 weight-endpoint distortion has been measured, but
  real GLM W4A8 activation quality and speed have not.

No inference service or GPU process is started by this repository.
