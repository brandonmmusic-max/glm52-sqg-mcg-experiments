# Publication validation

Validation was rerun for the 2026-08-10 interim status update with
`CUDA_VISIBLE_DEVICES` empty. No inference container or GPU workload was
started by the publication work.

## Test results

| Suite | Result |
|---|---:|
| Late-block runner/blend/construction contracts, current update | 26 passed |
| Main project, prior interim publication validation | 267 passed, 1 skipped |
| Vendored KQuant, prior publication validation | 346 passed, 1 skipped |
| BMM Law R7 encoder, prior publication validation | 44 passed |
| Upstream QSRT audit recorded in the new Kimi K3/K1 report | 491 passed, 1 skipped |

The latter four rows were not rerun during this publication step. The current
CPU-only run targeted the code and contracts changed by the late-block update.
The QSRT count belongs to the pinned upstream audit described in
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
- All 5,446 published JSON files parsed successfully; the compact contiguous-
  late evidence subtree contains 26 JSON files and 40 total files.
- The Test 10 capture manifest reports `complete: true` and 253,863 rows for
  each of layers 74--77; the four published preparation logs each report 256
  permutations, fit-only construction, zero MCG inputs, and zero fallback.
- The published late-block encode seal hashes to
  `567066231b882732d5bc82a093123002573a9cc8cf507ec763e2c2e7184606cf`
  and records 1,024 experts, 3,072 SQG tensors, 1,536 K3 tensors, 1,536 K4
  tensors, and zero MCG tensors.
- The materialization receipt hashes to
  `b9fd238a90370d4ea0fae4dfbb150390ae23f71e5ed11c0a3db1f54d9b35f3ca`;
  the candidate manifest hashes to
  `5720c1aba18af0917d142f1f755cd03a8311591a638c80cf53d4f410791f30b6`.
- The rejected first KLD attempt's log hashes to
  `07d398cf6cf3abe623db1876692aab0ee0dd8961776b80aa583ced0400485593`.
  It failed during engine initialization, before inference, and produced an
  empty `runs.jsonl`, no accepted record, no summary, and no KLD result.
- Shell syntax checks, Python byte compilation, and the sealed pure-SQG source
  manifest check passed. The corrected runner hashes to
  `b66bf7e38e01fc9f48113684cf6ef0ad3af4c4f4366c603223564d0d233ae9dc`.
- No nested Git repository is present.
- No file exceeds GitHub's 100 MB per-file limit.
- A high-confidence token/private-key scan found no secret.

The repository-wide `SHA256SUMS` file was generated after these checks.
