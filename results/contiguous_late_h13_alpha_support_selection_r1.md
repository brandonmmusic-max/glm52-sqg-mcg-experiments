# H13 local-alpha versus effective routed support

- Role: `selection`
- Experts: `1024`
- Spearman effective support vs per-expert winning alpha: `0.240647`
- Per-expert winners: `{'sqg_a000': 152, 'sqg_a025': 765, 'sqg_a050': 102, 'sqg_a075': 5, 'sqg_a100': 0}`

## Effective-support quartiles

| Quartile | n_eff range | Winning alpha | Mean SSE ratios to MCG |
|---:|---:|---:|---|
| 1 | 128.9–1798.2 | 0.25 | `{'sqg_a000': 1.0766389542898314, 'sqg_a025': 1.0507661072093448, 'sqg_a050': 1.0749225922212613, 'sqg_a075': 1.134067376984797, 'sqg_a100': 1.3873710524823442}` |
| 2 | 1801.3–2608.5 | 0.25 | `{'sqg_a000': 1.1106268004957185, 'sqg_a025': 1.079262340785434, 'sqg_a050': 1.1036684701847266, 'sqg_a075': 1.1663225790736862, 'sqg_a100': 1.4663006481417704}` |
| 3 | 2611.0–3497.8 | 0.25 | `{'sqg_a000': 1.11182222003117, 'sqg_a025': 1.0745441111419263, 'sqg_a050': 1.0931738566903166, 'sqg_a075': 1.1491972749404396, 'sqg_a100': 1.429974776788816}` |
| 4 | 3499.5–55173.9 | 0.25 | `{'sqg_a000': 1.1219081633291246, 'sqg_a025': 1.0830323246190243, 'sqg_a050': 1.0942599891009817, 'sqg_a075': 1.1364177726509672, 'sqg_a100': 1.3458845186063582}` |

diagnostic association only; alpha bytes were encoded under frozen profiles/permutations selected by the layer-global arm
