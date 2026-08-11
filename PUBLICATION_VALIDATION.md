# Publication validation

Validation was rerun for the 2026-08-11 late-block results update with
`CUDA_VISIBLE_DEVICES` empty. No inference container or GPU workload was
started by the publication work.

## Test results

| Suite | Result |
|---|---:|
| Main project, current update | 301 passed, 1 skipped |
| New retained-profile co-routing unit tests | 2 passed |
| Vendored KQuant, prior publication validation | 346 passed, 1 skipped |
| BMM Law R7 encoder, prior publication validation | 44 passed |
| Upstream QSRT audit recorded in the new Kimi K3/K1 report | 491 passed, 1 skipped |

The latter three rows were not rerun during this publication step. The current
main-project run covered all collected tests; its one skip is the production
CUDA FP16 boundary regression that requires SM120.
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
that correction, all 259 tests collected at that publication stage passed or
skipped.

The first full run for this update found two native-control contract drifts:
the per-position validator hash still named the pre-roundoff validator, and the
native arm lacked the candidate runner's three trace-environment keys. The
published native runner now pins the current validator and supplies trace-off
values for those keys. The complete suite then passed.

## Evidence and repository checks

- Expert-local encode: 2,118 source non-tensor files and 2,118 published files.
- Shared-H13 encode: 7,394 source non-tensor files and 7,394 published files.
- Expert-local KLD: 72 source non-cache files and 72 published source files.
- Shared-H13 KLD: 75 source non-cache files and 75 published files.
- Every local Markdown link in the audited experiment ledger resolves.
- The README, experiment ledger, and Test 10 evidence index have no broken
  repository-relative links.
- All 5,543 published JSON files parsed successfully; the compact contiguous-
  late evidence subtree contains 63 JSON files and 97 total files (about 1.2
  MB).
- After the completed alpha/Test 8b/Test 8c update, all 5,566 published JSON
  files parse successfully.  The seven current entry-point/evidence Markdown
  files have zero broken repository-relative links.  Three pre-existing links
  in explicitly historical/vendored documentation remain nonportable and are
  outside this update's entry-point audit.
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
- The accepted late alpha-0.25 summary hashes to
  `50648c13e4b52bfb873e45aee0bba87e8956d53e57bf725dc23b438fccfb279b`.
  Ten compact 2,047-position vectors across the original and second five-boot
  sets match their validation-record SHA-256 values.
- The new E4M3 endpoint, signed top-8 holdout, empirical-null, cross-arm trace,
  and r33-r33 trace-control JSON hashes match the values recorded in the
  experiment ledger.
- All five completed late H13 alpha run seals are archived with a local
  SHA-256 manifest.  The selected alpha-0.25 seal matches the already
  published contiguous-late seal byte for byte.  All-arm selection/secondary-
  holdout JSON, support-conditioned diagnostics, and the 625-combination
  layerwise result are published; bulky per-position NPZ arrays are bound by
  path, byte count, and SHA-256 instead of duplicated.
- Test 8b publishes its full 256-expert aggregate JSON/Markdown and exact
  scorer/tests; its omitted per-position arrays are hash-bound.  Test 8c
  publishes its compact 126-case report plus exact benchmark/test/kernel
  snapshots.  The omitted 3,516,542-byte timing JSON hashes to
  `6740f8be4480a9272866b03055ea0f2dee3e5ed8d4453a8e1d4f5d757f64bb58`.
- The winner-native profile result, realized exact h-A8 `(H,B)` down reject,
  and unary-bounded retained-profile result are published with their methods,
  holdout dispositions, and authoritative IDs. The omitted co-routing result
  JSON hashes to
  `11e483476866862b47f9fc74cd1c715fbcabdbe228bb5fd9a4b48a2b4361304b`.
  Its preregistration hashes to
  `c236fd36b0e3e2cbd422ec0ef46a76fe4e103f939abbbb9ba5b6a2b2fc80825e`.
- The co-routing runner passes Python byte compilation, Ruff, shell syntax,
  and both deterministic solver unit tests. The 112-second production run
  completed with zero new encodes, zero uniform-K3 substitution, and zero MCG
  inputs.
- Shell syntax checks, Python byte compilation, and the sealed pure-SQG source
  manifest check passed. The corrected runner hashes to
  `aa7f5d556d2b5700b7fe49a441778822853bd75eeb0db70090eb71bf93dda636`;
  the resumable active-panel launcher hashes to
  `b5d23574cf4497843018eb3c77e5a9b82e42e08524b6e9e7af31d6ed9464ebe9`.
- No nested Git repository is present.
- No file exceeds GitHub's 100 MB per-file limit.
- A high-confidence token/private-key scan found no secret.

The repository-wide `SHA256SUMS` file was generated after these checks.
