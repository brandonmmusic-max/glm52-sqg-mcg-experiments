# Coupled-Hadamard K96-tail campaign source

This directory is the compact source and evidence snapshot for the GLM-5.2
coupled H512/H128 K96-tail SQG re-encode. It mirrors the code that produced the
measured receipts without copying model tensors, saved activation captures,
compiled extensions, container layers, caches, or credentials.

The snapshot has five parts:

- `campaign/` contains the 52-file executable closure used by the controller,
  its launchers, workers, validators, assembly, and final acceptance seal.
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
- `evidence/` contains the 2026-08-14 timestamped score receipt hashes, layers
  3 through 18 manifests and native oracles, layers 4 through 18 exact parity
  receipts, the layer-19 runtime binding, and the GPU recovery records.

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
identity and the completed layer-level measurements only. It does not prove a
candidate KLD result, codec census, MTP3 smoke, or release acceptance.
