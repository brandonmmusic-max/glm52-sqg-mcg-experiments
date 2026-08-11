# Signed top-8 H13 blend score — selection

Errors are summed as signed router-weighted expert-output vectors before squaring.

| Blend | Signed top-8 NMSE | Sum/individual SSE | Positions improved | Squared-error CVaR 1% | Relative-error CVaR 1% | p99 relative error |
|---|---:|---:|---:|---:|---:|---:|
| sqg_a000 | 7.193880859e-03 | 1.004073 | 16.070% | 6.180597635e+02 | 2.870186200e-02 | 2.588116619e-02 |
| sqg_a025 | 6.919615390e-03 | 1.002448 | 25.570% | 6.059095746e+02 | 2.878978729e-02 | 2.592324541e-02 |
| sqg_a050 | 6.995750936e-03 | 1.002071 | 22.865% | 6.066570979e+02 | 2.959485665e-02 | 2.660518596e-02 |
| sqg_a075 | 7.262164828e-03 | 1.002380 | 14.647% | 6.201945448e+02 | 3.097513194e-02 | 2.767203979e-02 |
| sqg_a100 | 8.486056401e-03 | 1.001755 | 3.437% | 6.600118059e+02 | 3.672363261e-02 | 3.283269798e-02 |
| mcg | 6.628382237e-03 | 1.001878 | 0.000% | 6.824178070e+02 | 3.275976393e-02 | 2.890151754e-02 |

## Selection

- Status: `no_nonbaseline_candidate_passed_all_hard_constraints_baseline_retained`
- Eligible candidates: `[]`
- Selected candidate: **`mcg`**
- Diagnostic nonbaseline fallback: `sqg_a025`

The improved-position fraction is optimized only after aggregate and positive-tail constraints pass.
