# Hessian-weighted encoded distortion: SQG versus MCG

Lower is better. All reported matrix traces and routed-output reductions are exact FP32 GPU computations; this is not KLD.

## Overall

| Metric | Tensors | MCG HNMSE | SQG HNMSE | SQG reduction | SQG wins |
|---|---:|---:|---:|---:|---:|
| Gate/up: layer-global H13 | 512 | 5.627330198e-03 | 3.236130229e-03 | +42.493% | 512 (100.0%) |
| Gate/up: expert-local H13_e | 512 | 2.934506604e-03 | 2.813184095e-03 | +4.134% | 288 (56.2%) |
| Down: H2 from BF16 upstream | 256 | 7.000809367e-03 | 5.991741207e-03 | +14.414% | 256 (100.0%) |
| Down: H2 from MCG upstream | 256 | 7.000252199e-03 | 5.999989317e-03 | +14.289% | 256 (100.0%) |
| Down: H2 from SQG upstream | 256 | 7.003111988e-03 | 5.989065134e-03 | +14.480% | 256 (100.0%) |

## Contract

- Gate/up layer-global scores use each sealed fit-only H13 plus the encoder's 0.025 diagonal damping.
- Expert-local gate/up scores replay the exact fit routes and applied gate-square weights, with the same diagonal damping.
- Down scores rebuild the frozen OAS-to-scaled-identity H2 under BF16, decoded-MCG, and decoded-SQG upstream paths; each common H2 scores both down reconstructions.
- MCG and SQG are decoded from their packed bytes and mapped back to the same official BF16 coordinate order.
- The K3/K4 assignment and BF16 source payload are held fixed.

## Interpretation limit

Hessian-weighted local distortion is closer to functional sensitivity than raw NMSE, but it still does not include cross-layer compounding, changed routing, runtime dispatch, or final-logit KLD.
