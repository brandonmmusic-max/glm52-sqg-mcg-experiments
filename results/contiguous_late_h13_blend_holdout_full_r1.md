# Signed top-8 H13 blend score — holdout

Errors are summed as signed router-weighted expert-output vectors before squaring.

| Blend | Signed top-8 NMSE | Sum/individual SSE | Positions improved | Squared-error CVaR 1% | Relative-error CVaR 1% | p99 relative error |
|---|---:|---:|---:|---:|---:|---:|
| sqg_a000 | 7.333270917e-03 | 1.003405 | 24.281% | 6.004204507e+02 | 2.722442588e-02 | 2.483902714e-02 |
| sqg_a025 | 7.023407971e-03 | 1.001914 | 39.030% | 5.874295248e+02 | 2.725572534e-02 | 2.480939929e-02 |
| sqg_a050 | 7.102442922e-03 | 1.001980 | 34.615% | 5.934961772e+02 | 2.790534702e-02 | 2.536839895e-02 |
| sqg_a075 | 7.361940297e-03 | 1.001879 | 21.470% | 6.025308389e+02 | 2.914846234e-02 | 2.655279042e-02 |
| sqg_a100 | 8.604870653e-03 | 1.001205 | 5.027% | 6.424576613e+02 | 3.412239860e-02 | 3.115832401e-02 |
| mcg | 6.834071680e-03 | 1.001339 | 0.000% | 6.463883525e+02 | 2.735032113e-02 | 2.429805166e-02 |

## Selection

- Status: `no_nonbaseline_candidate_passed_all_hard_constraints_baseline_retained`
- Eligible candidates: `[]`
- Selected candidate: **`mcg`**
- Diagnostic nonbaseline fallback: `sqg_a025`

The improved-position fraction is optimized only after aggregate and positive-tail constraints pass.
