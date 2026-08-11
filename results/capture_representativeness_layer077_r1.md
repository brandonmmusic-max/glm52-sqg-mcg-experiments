# Layer-77 reduced/full calibration representativeness diagnostic

This is a geometry and selection-stability diagnostic, not a causal corpus-size result.

## H13 geometry

- Full fit rows: **601,343**
- Reduced fit rows: **150,368**
- Diagonal-mean change: **`-0.4227%`**
- Diagonal correlation: **`0.90060976`**
- Relative Frobenius difference: **`0.503860`**
- Frobenius cosine: **`0.89194025`**

## Profile stability

- Full selected: **`draw-00__identity`**
- Reduced selected: **`draw-03__identity`**
- Selection reversed: **True**
- Common-cell rank correlation: **`0.488235`**
- Expert-panel overlap: **2 / 16**

### `draw-00__identity`

- Full error/rank: `1.757286971e-03` / 1
- Reduced error/rank: `2.037878650e-03` / 4

### `draw-03__identity`

- Full error/rank: `1.828253459e-03` / 3
- Reduced error/rank: `1.942465086e-03` / 1

## Decision

A reversed selection cannot be dismissed as harmless, but it also cannot be attributed only to corpus size because the candidate constructions differ. The required next gate is an external cross-score of the already encoded reduced draw-00 and draw-03 candidates on full-capture documents excluded from the reduced plan. Middle/early encoding should not reuse the reduced selector until that gate is resolved.
