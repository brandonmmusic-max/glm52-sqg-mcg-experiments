# Same-rate W4A8 encoder batching equivalence audit

Status: complete negative result. Production encoding remains strictly
singleton per tensor.

## Method

The proposed scheduling optimization grouped same-rate candidate-specific down
encodes. Real layer-77 BF16 tensors, routed fit evidence, W4A8-native H13,
exact upstream operands, anchored `suh/svh`, coordinate-correct `(H,B)`
targets, beta, and all source/runtime seals were held fixed.

Every candidate was compared with the production batch-size-one path.
Acceptance required exact equality—not tolerance—for packed trellis bytes,
payload hash, independent decoded reconstruction, proxy scalar, both scale
vectors, and the complete manifest. MCG inputs were forbidden.

Layer-77 expert 0 mapped the initial failure surface. The decisive follow-up
used selected production `beta=0.0625` on expert 243, route-mass rank 8. It was
the first expert in a preregistered eight-expert route-mass panel; the panel was
fail-fast because one valid mismatch disproves universal scheduling equality.

## Results

- K4 groups of four changed two of four expert-0 candidate encodes.
- K4 groups of two still changed both candidates in the first pair.
- K3 groups of four happened to match all four expert-0 candidates.
- K3x4 then failed on the first selected-beta panel expert. For expert 243,
  the `(gate K3, up K3, down K3)` cell changed trellis bytes, decoded
  reconstruction, proxy scalar, and manifest. The other three K3 cells
  happened to match.

The decisive failure was:

| Field | Serial | K3x4 batch |
|---|---|---|
| Candidate seconds | `5.1897265429` | `3.5459514819` |
| Proxy error | `0.0069042488674654675` | `0.006904248648560349` |
| Trellis payload SHA256 | `f75f5e8406b9c16a2c779e9aefcaa92aed9e93871a6a80e7a044ba5e2236a902` | `0bd502a6169a5581fe50a3775b3a634045c854432a61ca61cc1147d7f5043ede` |
| Decoded EXL SHA256 | `75062f74db357c389c59edea99fd3e1b880f81d59e1d5971326d36910b8b92f6` | `1eb223caeb64bbb80d4b076a7d1311553d3783e18d9bd4cf87b55fcccd13e676` |
| `suh` / `svh` | exact | exact |

The changed payload and decoded reconstruction rule out a manifest-only
discrepancy. Multi-source CUDA evaluation changes a marginal trellis decision
through floating-point evaluation order. The tiny proxy delta cannot make the
bytes interchangeable.

The decisive result ID is
`3562b082ffae4f1045b075b9a023cc836042de2b836ad19fcc4512fec635fb69`.
Its local file SHA256 is
`c45700a6d478f24f3eb71c669bb2f5e133025666555ecd13d9be88e67140a9e2`.
The final disposition SHA256 is
`a1374c75ccb2f4d6da4983e4a18aa49acf5380e42a89f69b1f5f193758f5715c`.

## Decision

Reject all same-rate batching for the production encoder. Every gate, up, and
down candidate remains batch size one; the triplet scorer retains fourteen
physical encodes per expert. This preserves deterministic accepted bytes at
the cost of the proposed construction-time speedup. It does not affect the
route-packed inference kernel or its W4A8 serving speed.
