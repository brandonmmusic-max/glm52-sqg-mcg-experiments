# Exploratory layerwise H13 blend selection

- Status: `no_all_sqg_combination_passed_mcg_hard_gates`
- Search space: `625` combinations
- Eligible: `0`
- Winner: `{'74': 'sqg_a025', '75': 'sqg_a025', '76': 'sqg_a025', '77': 'sqg_a050'}`
- Signed top-8 NMSE: `6.924001292e-03`
- Positions improved: `25.201%`
- Absolute-error worst-1% CVaR: `6.022194511e+02`
- Relative-error worst-1% CVaR: `2.893611032e-02`
- Advances over uniform winner: `False`
- Recommended mapping: `{'74': 'sqg_a025', '75': 'sqg_a025', '76': 'sqg_a025', '77': 'sqg_a025'}`

The layerwise search and materiality correction are exploratory; the 0.1% floor was added after the first position-count ranking selected a numerically trivial 0.000032% NMSE change. The result requires holdout and final-logit confirmation.
