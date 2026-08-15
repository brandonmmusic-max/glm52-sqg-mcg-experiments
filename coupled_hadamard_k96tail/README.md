# Coupled-Hadamard K96-tail campaign source

This directory is the compact source and evidence snapshot for the GLM-5.2
coupled H512/H128 K96-tail SQG re-encode. It mirrors the code that produced the
measured receipts without copying model tensors, saved activation captures,
compiled extensions, container layers, caches, or credentials.

The snapshot has seven parts:

- `campaign/` contains the 102-file executable and focused-test closure used by
  the controller, local and remote launchers, workers, validators, assembly,
  upload/prefetch tooling, and final acceptance seal.
  Its test `conftest.py` adds the adjacent complete published QSRT snapshot to
  `sys.path`; this is the sole publication-only normalization and makes the 23
  focused tests runnable directly from this repository.
- `sources/` contains complete working source snapshots of the active QSRT and
  KQuant checkouts. Git metadata, generated output, caches, compiled objects,
  and tensor payloads are excluded.
- `patches/` contains the exact binary-safe QSRT and KQuant diffs plus a copy of
  every changed tracked file. They preserve audit deltas but are not required
  to reconstruct the complete working snapshots.
- `runtime/` contains the final acceptance compose file, server launcher,
  Docker build context, complete source overlay, KLD runner, codec validator,
  and smoke test. Generated caches and binaries are excluded. The active
  overlay checksum file named five Ruff cache entries, so the build-context
  copy is regenerated over the 248 published source files. The exact active
  checksum file remains in `manifests/active_runtime_OVERLAY_SHA256SUMS`.
- `reproduction/` contains the current human method, machine-readable campaign
  contract, reproduction index, and public staging model card.
- `orchestration/` contains the exact frozen Vast supervisor and local systemd
  configurations. Credential paths are retained as configuration; no token,
  provider secret, SSH endpoint, or private key value is included.
- `evidence/` retains the earlier 2026-08-14 checkpoint and the new frozen
  `snapshot_2026-08-15T002639-0400/`: compact manifests, quality and native
  runtime oracles through sealed layer 46; exact score/encoder parity and
  production allocation receipts for K96 layers 4--46; compact recipe/profile
  lineage; and SHA-256 bindings plus aggregate metrics for every complete
  atomic score JSON/row-SSE pair in layers 4--50.

`SOURCE_SHA256SUMS` binds every published file in this directory except itself.
Run the internal check from the repository root:

```bash
python3 coupled_hadamard_k96tail/verify_source_sync.py
```

On the measured host, compare every mirrored source byte to the active
workspaces and prove that both patches apply to their pinned base revisions:

```bash
python3 coupled_hadamard_k96tail/verify_source_sync.py --compare-active
```

The expected source revisions, snapshot counts, and patch hashes are in
`manifests/source_provenance.json`. Runtime image, package, and extension
identities are in `manifests/runtime_dependencies.json`.

## Scope boundary

This is an exact source mirror, not a standalone model or capture archive. A
second host still needs access to the frozen SQG checkpoint, the pinned Hessian
dataset revision, the named container images, and the compiled extensions. The
controller also preserves the measured host's absolute roots. Install the
snapshot at those roots or make an explicitly reviewed relocation patch before
running it elsewhere. Do not describe a cross-host execution as reproduced
until those external bindings and the resulting receipts pass.

The full-model artifact remains unsupported. This snapshot proves source
identity and the completed layer-level measurements only. Final end-to-end KLD,
the final Hub model commit, codec census, MTP3 smoke, and release acceptance are
all explicitly **PENDING FINAL**.
