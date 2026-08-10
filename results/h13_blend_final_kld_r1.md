# H13 blend final-logit KLD tail comparison

Calibration-selected arm: **`alpha025`**; confirmation status: **`calibration_winner_failed_one_or_more_final_kld_constraints`**.

| Arm | Mean KLD | Mean delta | Positions improved | KLD CVaR 1% | KLD p99 | Eligible |
|---|---:|---:|---:|---:|---:|---:|
| alpha0 | 6.284989666e-02 | 0.000000000e+00 | 0.000% | 1.838378917e+00 | 1.079668074e+00 | true |
| alpha025 | 6.246169148e-02 | -3.882051885e-04 | 52.027% | 1.904047408e+00 | 1.206657501e+00 | false |

Eligibility requires mean KLD, worst-1% KLD CVaR, and p99 KLD all to be no worse than the alpha-0 baseline. Position win rate is descriptive and is never allowed to override a failed tail gate.

## Uncertainty

| Arm | Boot-noise SE | Welch t | Boot direction | Position-block 95% CI | Position direction |
|---|---:|---:|---|---:|---|
| alpha025 | 7.929287286e-04 | -0.489583 | inconclusive | [-3.572643524e-03, 2.770346077e-03] | inconclusive |

Welch statistics describe independent fresh-boot variation. The circular block bootstrap describes correlated positions within the one fixed prompt and is not a document-generalization interval.

The exploratory best arm is reported for diagnosis only. The calibration-selected arm is the only preregistered confirmation target; choosing another arm from this fixed prompt would make the prompt selection data.
