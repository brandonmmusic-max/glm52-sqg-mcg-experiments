# Exploratory layerwise H13 blend selection

- Status: `tail_and_mean_constraints_passed`
- Search space: `625` combinations
- Eligible: `125`
- Winner: `{'6': 'alpha100', '28': 'alpha050', '52': 'alpha025', '77': 'alpha025'}`
- Signed top-8 NMSE: `2.373317182e-03`
- Positions improved: `89.713%`
- Absolute-error worst-1% CVaR: `6.374142361e+02`
- Relative-error worst-1% CVaR: `2.191457006e-02`
- Advances over uniform winner: `False`
- Recommended mapping: `{'6': 'alpha025', '28': 'alpha025', '52': 'alpha025', '77': 'alpha025'}`

The layerwise search and materiality correction are exploratory; the 0.1% floor was added after the first position-count ranking selected a numerically trivial 0.000032% NMSE change. The result requires holdout and final-logit confirmation.
