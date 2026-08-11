# Publication validation

Validation was rerun for the 2026-08-10 interim status update with
`CUDA_VISIBLE_DEVICES` empty. No inference container or GPU workload was
started by the publication work.

## Test results

| Suite | Result |
|---|---:|
| Main project, current update | 267 passed, 1 skipped |
| Vendored KQuant, prior publication validation | 346 passed, 1 skipped |
| BMM Law R7 encoder, prior publication validation | 44 passed |
| Upstream QSRT audit recorded in the new Kimi K3/K1 report | 491 passed, 1 skipped |

The latter three rows were not rerun during this documentation-only publish
step. The QSRT count belongs to the pinned upstream audit described in
[`docs/qsrt_kimi_k3_k1_feasibility.md`](docs/qsrt_kimi_k3_k1_feasibility.md),
not to the vendored KQuant tree.

The BMM Law directory retains its historical local name
`bmmlaw_r7_encoder`, while its tests import `r7_encoder`. The validation run
provided a temporary `r7_encoder` path alias; no source was changed for that
compatibility detail.

## Publication portability correction

The first main-project run exposed eight failing capture-launcher safety tests.
The launcher had bound `project` to the original absolute workspace path, so a
copy at a different path was not recognized as a protected root. The published
copy now derives `project` from the launcher's own canonical directory. After
that correction, all 259 collected main-project tests passed or skipped as
shown above.

## Evidence and repository checks

- Expert-local encode: 2,118 source non-tensor files and 2,118 published files.
- Shared-H13 encode: 7,394 source non-tensor files and 7,394 published files.
- Expert-local KLD: 72 source non-cache files and 72 published source files.
- Shared-H13 KLD: 75 source non-cache files and 75 published files.
- Every local Markdown link in the audited experiment ledger resolves.
- The README, experiment ledger, and Test 10 evidence index have no broken
  repository-relative links.
- All 35 newly relevant result, contract, source-seal, and compact Test 10 JSON
  files parsed successfully.
- The Test 10 capture manifest reports `complete: true` and 253,863 rows for
  each of layers 74--77; the four published preparation logs each report 256
  permutations, fit-only construction, zero MCG inputs, and zero fallback.
- No nested Git repository is present.
- No file exceeds GitHub's 100 MB per-file limit.
- A high-confidence token/private-key scan found no secret.

The repository-wide `SHA256SUMS` file was generated after these checks.
