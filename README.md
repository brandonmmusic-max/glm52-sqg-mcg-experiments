# GLM-5.2 SQG versus MCG experiments

This repository preserves the code, methodology, compact evidence, and audited
results from the GLM-5.2 3.5-bpw experiments comparing the existing MCG
trellis representation with KQuant-style SQG. It now includes the signed
top-8, tail-constrained `H13` blend ablation, the preregistered contiguous-block
follow-up, and a separate QSRT/Kimi K3 K1 feasibility audit.

The experiment kept the production topology and the per-tensor K3/K4 bit
assignment fixed. Bits were **not** reallocated per expert. The tested layers
were 6, 28, 52, and 77, with 256 experts and three expert projections per layer.

## Bottom line

SQG reconstructed the BF16 weights more closely than MCG in the raw encoded
comparison. A 25% expert-local/75% layer-global `H13` blend also improved the
signed routed top-8 proxy on both selection and encoder-unseen holdout data.
Those results still do **not** demonstrate that SQG lowers full-model KLD.

The alpha-0.25 candidate moved the five-boot KLD mean in the favorable
direction, but the repeat uncertainty crossed zero and the preregistered
final-logit positive-tail gate failed. It is therefore the best proxy blend
tested, not a validated full-model winner.

| Measurement | Result | What it establishes |
|---|---:|---|
| Raw encoded-weight NMSE, 3,072 matched tensors | SQG `15.105%` lower than MCG | SQG is geometrically closer to BF16 under ordinary weight NMSE |
| Fixed 75%-local `H13`, held-out complete-expert NMSE | `5.985%` lower than shared-`H13` SQG; `7.951%` lower than MCG | Expert-local calibration better matches the isolated routed expert function |
| Alpha-0.25 signed top-8 selection NMSE | `6.541%` below alpha 0; `89.701%` of positions improved | Sole coarse arm to pass all five proxy tail gates |
| Alpha-0.25 signed top-8 holdout NMSE | `6.389%` below alpha 0; `89.261%` improved | Encoder-unseen, analysis-seen confirmation |
| Alpha-0.25 final KLD | `0.0624616917` vs alpha-0 `0.0628498963` | Favorable `-0.6177%` mean; repeat interval crosses zero |
| Alpha-0.25 final position wins | `1,065 / 2,047` (`52.027%`) | Broad direction is only slightly positive |
| Alpha-0.25 final positive tail | p99 `1.20666` vs `1.07967`; worst-1% CVaR `1.90405` vs `1.83838` | Preregistered final-logit tail gate failed |

The strongest supported conclusion is therefore narrower than “SQG wins” or
“MCG wins”: expert-conditioned calibration fixes a real activation-geometry
mismatch, and alpha 0.25 is the best blend tested by the isolated routed
objective, but the final-logit tail remains unresolved. Three contiguous A16
blocks and separate W4A8 quality/speed tests are required before extrapolating
to a full SQG quant.

## Current Test 10 status

The late contiguous block, layers 74-77, is the first of the preregistered
early/middle/late blocks.

| Stage | Status |
|---|---|
| 217-document plan | Sealed: 253,863 whole-document rows per layer; 150,368 fit, 51,232 selection, 52,263 holdout |
| Routed capture | Complete and sealed for layers 74-77 |
| Four-GPU layer preparation | Complete: 256 permutations/layer, fit-only, zero MCG inputs, zero fallback |
| Profile search | Not yet complete |
| Alpha-0.25 encoding/materialization | Not yet complete |
| Final-logit KLD and propagation trace | Not yet run |

There is consequently no Test 10 quality conclusion yet. Middle layers 38-41,
early layers 10-13, and all W4A8 quality/speed endpoints remain pending.

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
- The four selected layers are separated rather than contiguous, so the test is
  weak for consecutive-layer error compounding and routing drift.
- The MCG baseline and SQG candidate did not share the same selected-layer
  dispatch path. The preserved `+0.64057%` four-layer SQG-versus-r33 result is
  consequently a weight-plus-dispatch result, not a codebook-only result.
- This Git repository intentionally excludes model-scale tensor payloads and
  regenerated caches. Their manifests, hashes, receipts, and compact logs are
  preserved in `published_evidence/`.
- Direct MCG-versus-SQG E4M3 endpoint distortion and real GLM W4A8 quality and
  speed have not yet been measured.

No inference service or GPU process is started by this repository.
