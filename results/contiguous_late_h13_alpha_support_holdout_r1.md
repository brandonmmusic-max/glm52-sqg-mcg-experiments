# H13 local-alpha versus effective routed support

- Role: `holdout`
- Experts: `1024`
- Spearman effective support vs per-expert winning alpha: `0.255193`
- Per-expert winners: `{'sqg_a000': 142, 'sqg_a025': 790, 'sqg_a050': 90, 'sqg_a075': 2, 'sqg_a100': 0}`

## Effective-support quartiles

| Quartile | n_eff range | Winning alpha | Mean SSE ratios to MCG |
|---:|---:|---:|---|
| 1 | 128.9–1798.2 | 0.25 | `{'sqg_a000': 1.0069279839866874, 'sqg_a025': 0.9793428539509814, 'sqg_a050': 1.0035156338009705, 'sqg_a075': 1.057603823679485, 'sqg_a100': 1.2918931482467495}` |
| 2 | 1801.3–2608.5 | 0.25 | `{'sqg_a000': 1.0297737326301355, 'sqg_a025': 0.9962808972448436, 'sqg_a050': 1.0182924441566907, 'sqg_a075': 1.0749086137816408, 'sqg_a100': 1.3507924175750268}` |
| 3 | 2611.0–3497.8 | 0.25 | `{'sqg_a000': 1.0475792594769233, 'sqg_a025': 1.0099059679434839, 'sqg_a050': 1.026807589693184, 'sqg_a075': 1.078996896605533, 'sqg_a100': 1.3408809164136855}` |
| 4 | 3499.5–55173.9 | 0.25 | `{'sqg_a000': 1.065385825928738, 'sqg_a025': 1.0262243983963328, 'sqg_a050': 1.036151035591815, 'sqg_a075': 1.075064068289922, 'sqg_a100': 1.2698748131604556}` |

diagnostic association only; alpha bytes were encoded under frozen profiles/permutations selected by the layer-global arm
