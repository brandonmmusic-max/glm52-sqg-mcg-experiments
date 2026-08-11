# Coordinate-corrected full-W4A8 cross-term down repair, layer 77

Status: completed functional proxy; not final-logit KLD and not a full-model
acceptance result.

## Method

- Official BF16 GLM-5.2 parent, routed layer 77, all 256 experts.
- Frozen uncoupled alpha-0.25 winner-native SQG gate/up bytes with independent
  per-tensor K3/K4 rates.
- Native direct-E4M3 SQG labels; K32 UE8M0/E4M3 QDQ at both `h` and `act`;
  exact GLM `SiLU(gate) * up`; FP16 inter-projection boundaries; FP32 GEMM
  accumulation; Hadamard on the activation side.
- Fit routes only, weighted by applied router gate squared. Selection and
  holdout rows were excluded from H, B, and target construction.
- Canonical H/B used the effective label-side operand. KQuant received
  `q_pre = q_label H128 D`, so its internal `R D H D R` finalization recovered
  the intended label-space covariance instead of applying the transform twice.
- Base private down `suh` and selected shared down `svh` were anchored exactly.
- Exact captured top-8 IDs and applied gates; signed expert outputs were summed
  before squaring.
- MCG inputs, payloads, transforms, and scales: zero.

The earlier beta-1 pilot is invalid as a control: it supplied an already
transformed Hessian to KQuant and allowed the encoded private down input scale
to drift from target construction.

## Result

| Split | Base full-W4A8 NMSE | Corrected `(H,B)` NMSE | Relative change | Positions improved | Paired-document 95% improvement interval |
|---|---:|---:|---:|---:|---:|
| Selection | 0.004468411498 | 0.004040509502 | **-9.5762%** | 28.8394% | [0.000445164, 0.000569533] |
| Holdout | 0.004574636368 | 0.004148997857 | **-9.3043%** | 29.9083% | [0.000443685, 0.000549287] |

Both paired-document intervals exclude zero favorably. Individual-route SSE
falls 9.49%/9.22%, so the gain is not primarily cancellation. Relative to
Test 8b's matched SQG A16 arm, corrected full W4A8 remains 10.22% worse on
selection and 10.86% worse on holdout, down from 22.98%/22.78%.

Absolute squared-error worst-1% CVaR falls from 571.83 to 402.24 on selection
and 585.44 to 406.69 on holdout; maxima fall from 6156.23 to 4166.90 and
5982.45 to 4619.44. Relative-error p99 rises slightly, and about 70% of
positions have larger relative error despite the lower total SSE and absolute
tail.

## Validation and decision

- Experts: 256/256.
- Selection positions/routes: 51,232 / 409,856.
- Holdout positions/routes: 52,263 / 418,104.
- Result ID:
  `594862cfad3fcd33b55f14f400f6b418a5ce54886c51743e1d1c51b6ab39d0c6`.
- Source score SHA256:
  `ff4ad88c5bbec4ce09f1ea0500dfab2bb47ba165a7ce5f40a2452a3035a13529`.
- Code-recovery ID:
  `d71bf9e697d74e3fcd431286847aa4fa81259cab3094b75c3d9d920c9bbf8576`.

The corrected `(H,B)` construction is promoted as the W4A8 down-calibration
basis. Beta 1 itself is not promoted; beta is selected on a fit-only subfold.
