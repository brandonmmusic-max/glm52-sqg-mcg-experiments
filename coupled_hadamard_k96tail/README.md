# Coupled-Hadamard K96-tail campaign source and evidence

This directory contains the source, reproduction procedure, and compact
evidence for the GLM-5.2 coupled H512/H128 K96-tail SQG re-encode. The local
checkpoint is mechanically complete and its TP4/DCP4/MTP3 runtime is
qualified. The model is **research-only** because its full-vocabulary KLD is
84.849 percent worse than the frozen SQG source checkpoint. The public Hugging
Face model is **unsupported** as a complete model until its remaining routed
layers and final card are uploaded and verified.

## Status labels

- **implemented** means the named code path and fail-closed checks exist.
- **qualified** means the named artifact passed the stated measurement.
- **research-only** means the artifact or result does not satisfy the
  model-quality release gate.
- **unsupported** means the named use or claim must not be made.

| Object | Status | Basis |
|---|---|---|
| Coupled H512/H128/H128 encoding implementation | implemented | Exact source snapshot, patches, campaign scripts, and focused tests are present. |
| Routed layers 3 through 77 | qualified | Every target layer passed byte parity, materialization, and its TP1 native B12X oracle. |
| Local assembled checkpoint and 76-layer codec census | qualified | Assembly is complete at `/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail`; the codec census passes through MTP layer 78. |
| Exact-r11 TP4/DCP4/MTP3 runtime and task bundle | qualified | Four rank receipts, no fatal audit matches, MTP3 metrics, Estonia 5/5, and LAVD 4 exact plus 1 near are sealed. |
| Full-model KLD | research-only | Candidate mean `0.1401771516114036` exceeds source mean `0.07583317451217256`. |
| Hidden replay | research-only | The original preregistered maximum-position limit fails; only a post-observation operational envelope passes. |
| Public Hugging Face model | unsupported | Routed layers 3 through 50 are still being uploaded; no complete public revision has been sealed. |

Task results and serving closure do not supersede the failed KLD quality gate.
The checkpoint must not be described as a quality-passing replacement for the
frozen source.

## Directory map

- `campaign/` contains the executable and focused-test closure used by the
  controller, workers, validators, assembly, and upload tooling.
- `sources/` contains complete working-source snapshots of QSRT and KQuant.
- `patches/` contains binary-safe QSRT and KQuant diffs plus changed tracked
  files.
- `runtime/` contains the exact Infernal Invocation r11 runtime, source
  overlays, MTP3 prefix patch, KLD tools, qualification scripts, and upload
  gates.
- `reproduction/` contains the human procedure, machine-readable contract, and
  model-card source. `reproduction/hub/HESSIAN_DATASET_README.md` is the
  upload-ready dataset-card source for the calibration archive `main` branch;
  it does not modify the accepted immutable dataset tag.
- `orchestration/` contains the distributed supervisor and local systemd
  configuration without secret values.
- `evidence/` contains compact layer evidence, full KLD, hidden replay,
  preregistered replay history, and the sealed TP4/DCP4/MTP3 qualification
  bundle.

`SOURCE_SHA256SUMS` binds every published file except itself. Verify the
snapshot from the repository root:

```bash
python3 coupled_hadamard_k96tail/verify_source_sync.py
```

On the measured host, also compare the source mirror with the active
workspaces:

```bash
python3 coupled_hadamard_k96tail/verify_source_sync.py --compare-active
```

## Checkpoint and rate contract

The source checkpoint is
`brandonmusic/GLM-5.2-SQG-W4A8@593dd0d2de6f79ce4e65303930c22c75e1359d44`.
Calibration inputs are pinned to
`brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians@a05b3b92d749f6a641af5cfd52de2b4720380dfd`.
The encode reconstructs routed weights from the frozen SQG checkpoint. It does
not download or read the official BF16 routed weight shards.

| Layer range | Treatment | K3 | K4 | Routed bpw |
|---|---|---:|---:|---:|
| 3 | Sealed coupled K48 layer | 720 | 48 | 3.0625 |
| 4 through 77 | Coupled K96 layers | 672 | 96 | 3.125 |
| 78 | Preserved source MTP layer | 384 | 384 | 3.5 |

Target layers 3 through 77 average `3.1241666666666665` bpw. All 76 routed
layers, including MTP layer 78, average `3.1291118421052633` bpw. This is not a
uniform 3.0625-bpw model and is not a uniform K96 model.

The coupled transformation uses residual H512, H128 before the activation,
H128 after the activation, exact GLM `silu(gate) * up`, H13 local alpha
`0.25`, and candidate-conditioned downstream H2. Layers 4 through 77 use the
full per-layer profile and beta recipe. Fleet beta reuse, the B300 owner-speed
rescue, identity-only fallback, and byte-changing same-rate batching are
excluded.

## Quality result

The full-vocabulary measurement is `KL(BF16 reference || candidate)` on the
same fixed 2,048-token sequence, all 2,047 causal positions, all 154,880
vocabulary entries, no trimming, and TP4/PP1/DCP1.

| Metric | Frozen source | Candidate |
|---|---:|---:|
| Mean | `0.07583317451217256` | `0.1401771516114036` |
| Median | | `0.0015440876595675945` |
| p95 | | `0.6778242588043213` |
| p99 | `1.397577404976` | `2.480538845062256` |
| Worst-1% CVaR | `2.207112874304` | `4.160942645300002` |
| Maximum | `5.978030681610` | `8.928885459899902` |

All 2,047 candidate positions are finite. The candidate mean is
`0.06434397709923104` higher, `1.84849x` the source mean, and 84.849 percent
worse. The lower-mean, lower-p99, and lower-CVaR release gates fail.

The candidate receipt is
[`evidence/final-exact-ii-r11/full-kld/kld_exact_ii_r11_tp4dcp1.json`](evidence/final-exact-ii-r11/full-kld/kld_exact_ii_r11_tp4dcp1.json),
SHA-256
`7979c9c8b0c81714cd38e225646e42a88b2cb8eb03232be271373255c506a408`.

## Hidden replay boundary

The operational hidden replay reports mean KLD `0.1401762649458023` and top-1
agreement `0.9174401563263312` for 2,047 positions. Its receipt SHA-256 is
`34f14cada0424ddb1387fec78a96a16ebe2109ddf2c063862d52108d0450e6b2`.

The original preregistered receipt has SHA-256
`7433ad312740dabaa1f3dc0c6e2a8317e741a441bccf06eb4a6f2ebe0569b3ad`
and `qualification_pass=false`. Its mean absolute KLD delta is
`8.866656012740393e-7`, below the `5e-5` limit, but its maximum position delta
is approximately `0.0035558`, above the `5e-4` limit. The original comparison
script SHA-256 is
`655301634275b617cb7e933a698811ad0605941c8d31264b4c2410086c05a038`.
The original service journal SHA-256 is
`380f3ed2e6c7ee8c46953a1fb3ab48678cdde813ceda489a348c7e8f6845c939`.

A post-observation operational envelope of mean `<=5e-5`, p99 `<=1e-4`, and
maximum `<=5e-3` passes. Because those limits were set after observing the
discrepancy, that pass is research-only.

## Runtime qualification

The accepted bundle is
[`evidence/final-exact-ii-r11/qualification-5x/exact-r11-tp4dcp4mtp3-20260815T084427Z`](evidence/final-exact-ii-r11/qualification-5x/exact-r11-tp4dcp4mtp3-20260815T084427Z/quality-summary.json).

- Image ID:
  `sha256:ab6bd60716b0a8e453b6345cb10e43e79726d92729b1f29058e31d7cc1c67def`
- Topology: TP4/DCP4/MTP3
- KV cache: `nvfp4_ds_mla`
- Quality summary SHA-256:
  `4f19ba5e4a8676c80bc49e89d346b0985faa209f14bdd6d9713e9ee6c4397f57`
- Runtime MTP summary SHA-256:
  `d8a7f22f6da05423a00d972e186deb17ba9f36133aae692c2477d53cd4f0f4ff`
- Four complete rank receipts, loaded and executed layers 3 through 78, and no
  fatal audit matches
- Estonia: 5/5 correct, zero truncations, average 3,371.6 completion tokens,
  average `43.5170233617816` generation tokens per second per run
- LAVD: 5/5 correct under published tolerance, 4 exact and 1 near, zero
  truncations, average 16,682.2 completion tokens, average
  `22.087665860284208` generation tokens per second per run

The LAVD near answer is `71, 45.75` against expected `72, 46`. It is incorrect
to report LAVD as 5 exact.

MTP3 generated 106,204 tokens, drafted 88,704 tokens, and accepted 76,631
draft tokens. Aggregate acceptance is `0.863895652958153`; per-position rates
are `0.9242762445887446`, `0.8623511904761905`, and
`0.8050595238095238`.

The MTP EXL3 prefix patch maps
`model.layers.78.mtp_block.mlp.shared_experts` to the canonical checkpoint
prefix. A real-image static prefix check passed. Full pytest did not run because
the test environment lacked a dependency, so no full-pytest claim is made.

## Public repository boundary

Hugging Face destination:
<https://huggingface.co/brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail>.

Server-side copy commit
`0c38e683eda27ca84982e3d513c89dd780dcdb22` copied and verified 390 files,
25,685,857,224 bytes, from the frozen source revision. Layers 51 through 77
were already present, and MTP layer 78 was copied. Two outer clients briefly
measured 21.75 Mbit/s, then each ramped to 16 internal streams. The resulting
32 CAS streams timed out and sustained transfer averaged about 9.6 Mbit/s. At
`2026-08-15T05:26:59-04:00`, the incomplete upload restarted with one outer
client. `UPLOAD_WORKERS=1` is the bound default. No completion ETA is a
canonical property.

The public model remains unsupported until the upload finishes, the canonical
full-folder verification passes, the research-only KLD result is disclosed on
the card, and anonymous public hash verification seals an immutable revision.

The follow-on wrapper does not wait for the intentionally stopped
qualification server. It uploads with two workers while excluding `README.md`
and `HUB_FILE_VERIFICATION.json`, then verifies every other local file against
Hugging Face metadata. LFS files are checked by SHA-256 and size; Git files are
checked by Git blob SHA-1 and size. The wrapper and verifier SHA-256 values are
`98fed5174cc349bd54a307e87d7ed9d7dcd4d9492548afbf0d3ae17d62adf36a`
and `aebc8b97fe332eb5b086e8f62b698e631fa10814f5fdf9504d201abb366efa11`.
The receipt is uploaded only after it records `complete=true`.

See the
[`reproduction procedure`](reproduction/docs/K96_COUPLED_DISTRIBUTED_REPRODUCTION_20260814.md),
[`machine-readable contract`](reproduction/machine/k96tail-distributed-campaign.json),
and [`runtime closure`](runtime/exact-ii-r11/README.md) for exact commands and
artifact bindings.
