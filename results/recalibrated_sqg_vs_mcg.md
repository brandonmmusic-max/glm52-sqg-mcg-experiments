# Expert-local H13 SQG recalibration

Lower NMSE is better. Complete-expert metrics include gate, up, SwiGLU, and down and are weighted by applied router-gate squared.

## Overall

| Metric | MCG | Current SQG | Corrected SQG | Corrected vs current | Corrected vs MCG |
|---|---:|---:|---:|---:|---:|
| Raw weights, all projections | 1.593874364e-02 | 1.353123630e-02 | 1.445579835e-02 | -6.833% | +9.304% |
| Fit routed gate/up | 3.058657929e-03 | 3.115421806e-03 | 1.676869676e-03 | +46.175% | +45.176% |
| Selection routed gate/up | 2.996738617e-03 | 3.327406100e-03 | 2.970626946e-03 | +10.722% | +0.871% |
| Holdout routed gate/up | 3.046226476e-03 | 3.330348199e-03 | 2.980227734e-03 | +10.513% | +2.167% |
| Fit complete expert function | 3.025458345e-03 | 2.707252775e-03 | 2.169658062e-03 | +19.858% | +28.287% |
| Selection complete expert function | 3.036441909e-03 | 2.967203829e-03 | 2.786487874e-03 | +6.090% | +8.232% |
| Holdout complete expert function | 3.043410360e-03 | 2.979759966e-03 | 2.801428997e-03 | +5.985% | +7.951% |

## Holdout complete expert function by layer

| Layer | MCG | Current SQG | Corrected SQG | Corrected vs current | Corrected vs MCG |
|---:|---:|---:|---:|---:|---:|
| 6 | 8.081666468e-04 | 9.678211483e-04 | 8.325492205e-04 | +13.977% | -3.017% |
| 28 | 7.827259835e-03 | 8.630394828e-03 | 7.552345493e-03 | +12.491% | +3.512% |
| 52 | 9.509290160e-03 | 1.002041474e-02 | 9.332718129e-03 | +6.863% | +1.857% |
| 77 | 2.909124303e-03 | 2.833240676e-03 | 2.665836380e-03 | +5.909% | +8.363% |

## Calibration contract

- Weighted-OAS local alpha: min `0.750000`, median `0.750000`, mean `0.750000`, max `0.750000`.
- Frozen BF16 source, K3/K4 assignment, profiles, permutations, transforms, SQG codebook, and C128 tail-biting.
- Down H2 rebuilt independently from each corrected decoded gate/up candidate.
- Selection and holdout rows were not used by the corrected encoder.

## Interpretation limit

This measures isolated routed expert functions. It does not include cross-expert error cancellation, later-layer routing drift, cross-layer compounding, or final-logit KLD.
