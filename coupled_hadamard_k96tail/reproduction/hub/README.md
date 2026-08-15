---
base_model: zai-org/GLM-5.2
library_name: vllm
pipeline_tag: text-generation
tags:
  - glm
  - sqg
  - w4a8
  - mixture-of-experts
  - blackwell
  - quantization
  - coupled-hadamard
  - research
license: other
---

# GLM-5.2 SQG Coupled H512/H128 K96Tail

> **Model status: research-only.** The checkpoint is mechanically complete and
> its TP4/DCP4/MTP3 runtime is qualified, but full-vocabulary KLD is 84.849
> percent worse than the frozen SQG source. The public file set is unsupported
> as a complete model until routed layers 3 through 50 finish uploading and an
> hash-bound revision passes anonymous verification.

This checkpoint re-encodes the routed experts from
`brandonmusic/GLM-5.2-SQG-W4A8@593dd0d2de6f79ce4e65303930c22c75e1359d44`
using saved calibration captures and Hessians from
`brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians@a05b3b92d749f6a641af5cfd52de2b4720380dfd`.
The encode did not download or read the official BF16 routed weight shards.
The KLD methods did use BF16 reference logits and the unchanged BF16 LM head;
that evaluation use is not an official BF16 routed-weight read.

## Status

| Object | Status | Result |
|---|---|---|
| Coupled encoding implementation | implemented | Residual H512, H128 before and after the activation, exact `silu(gate) * up`, H13 alpha 0.25, and candidate-conditioned H2 are bound in the receipts. |
| Routed layers 3 through 77 | qualified | All 75 target layers have sealed manifests, quality receipts, and passing TP1 native runtime oracles. Exact K96 scorer/encoder parity covers layers 4 through 77. Layer 3 is the sealed K48 exception and has no K96 parity receipt. |
| Local assembly and codec census | qualified | All 76 routed layers pass, including preserved source MTP layer 78. |
| Exact-r11 TP4/DCP4/MTP3 runtime | qualified | Four rank receipts, loaded and executed layers 3 through 78, no fatal audit matches, and sealed MTP3 metrics. |
| Estonia and LAVD task runs | qualified | Estonia is 5/5 correct. LAVD is 4 exact plus 1 near under the published tolerance. |
| Full-model KLD quality | research-only | Candidate mean KLD `0.1401771516114036` is worse than source mean `0.07583317451217256`. |
| Hidden replay | research-only | The original preregistered maximum-position limit fails. A post-observation operational envelope passes. |
| Public file set | unsupported | Routed layers 3 through 50 and final anonymous verification are incomplete. |

`implemented`, `qualified`, `research-only`, and `unsupported` are the only
status labels used by this card. Task behavior and serving closure do not
override the failed full-model KLD gate.

## Format and rate

The checkpoint uses the updated-QSRT coupled transform. Each routed expert is
treated as one gate/up/down function:

- residual coordinate: signed block Hadamard H512;
- preactivation coordinate: signed block Hadamard H128;
- activation: exact GLM `silu(gate) * up`;
- postactivation coordinate: signed block Hadamard H128;
- H13 local alpha: `0.25`; and
- down objective: candidate-conditioned H2.

Layers 4 through 77 run the full per-layer profile and beta search. Fleet beta
reuse, the B300 owner-speed rescue, identity-only fallback, and byte-changing
same-rate batching are excluded.

| Layer range | Treatment | K3 | K4 | Routed bpw |
|---|---|---:|---:|---:|
| 3 | Sealed coupled K48 layer | 720 | 48 | 3.0625 |
| 4 through 77 | Coupled K96 layers | 672 | 96 | 3.125 |
| 78 | Preserved source MTP layer | 384 | 384 | 3.5 |

Target layers 3 through 77 average `3.1241666666666665` bpw. All 76 routed
layers, including MTP layer 78, average `3.1291118421052633` bpw. This model is
not uniform 3.0625 bpw and is not uniform K96.

Non-routed and BF16 components remain byte-identical to the frozen source.
MTP layer 78 is also retained from the source.

## Full-vocabulary KLD

The primary quality measurement is `KL(BF16 reference || candidate)` over all
154,880 vocabulary entries at each of 2,047 causal positions. It uses the same
fixed 2,048-token input as the source measurement, no position trimming, and
TP4/PP1/DCP1.

| Metric | Frozen source | Candidate |
|---|---:|---:|
| Mean | `0.07583317451217256` | `0.1401771516114036` |
| Median | | `0.0015440876595675945` |
| p95 | | `0.6778242588043213` |
| p99 | `1.397577404976` | `2.480538845062256` |
| Worst-1% CVaR | `2.207112874304` | `4.160942645300002` |
| Maximum | `5.978030681610` | `8.928885459899902` |
| Finite positions | `2047/2047` | `2047/2047` |

The candidate mean is `0.06434397709923104` higher, `1.84849x` the source
mean, and 84.849 percent worse. The required lower mean, lower p99, and lower
worst-1% CVaR gates fail. This is why the model is research-only.

Candidate receipt SHA-256:
`7979c9c8b0c81714cd38e225646e42a88b2cb8eb03232be271373255c506a408`.

Source receipt SHA-256:
`5d8aedb462658c693f1ce790f48ce5ed3cd6876897b1367a5cd36e42c0e2d434`.

## Hidden replay

The operational hidden replay uses a raw BF16 `[2048,6144]` pre-LM-head
capture and the unchanged BF16 `[154880,6144]` LM head. It reports mean KLD
`0.1401762649458023` and top-1 agreement `0.9174401563263312` across 2,047
positions. Operational receipt SHA-256:
`34f14cada0424ddb1387fec78a96a16ebe2109ddf2c063862d52108d0450e6b2`.

The original preregistered receipt has SHA-256
`7433ad312740dabaa1f3dc0c6e2a8317e741a441bccf06eb4a6f2ebe0569b3ad`
and `qualification_pass=false`. Mean absolute KLD delta is
`8.866656012740393e-7`, below the `5e-5` limit, but maximum position delta is
about `0.0035558`, above the preregistered `5e-4` limit.

A later operational envelope uses mean `<=5e-5`, p99 `<=1e-4`, and maximum
`<=5e-3`; it passes. Those limits were chosen after the discrepancy was
observed, so that pass is research-only and is not preregistered qualification.

Original comparison script SHA-256:
`655301634275b617cb7e933a698811ad0605941c8d31264b4c2410086c05a038`.
Original service journal SHA-256:
`380f3ed2e6c7ee8c46953a1fb3ab48678cdde813ceda489a348c7e8f6845c939`.

## TP4/DCP4/MTP3 qualification

The qualified runtime is derived from Infernal Invocation r11. Infernal
Invocation r13 supplies native-SQG donor code only. v20 is not part of the
runtime lineage.

- Image:
  `verdictai/glm52-k96-ii-r11:20260815-tpfix-mtpfix`
- Image ID:
  `sha256:ab6bd60716b0a8e453b6345cb10e43e79726d92729b1f29058e31d7cc1c67def`
- Topology: TP4/DCP4/MTP3
- KV cache: `nvfp4_ds_mla`
- Quality summary SHA-256:
  `4f19ba5e4a8676c80bc49e89d346b0985faa209f14bdd6d9713e9ee6c4397f57`
- Runtime MTP summary SHA-256:
  `d8a7f22f6da05423a00d972e186deb17ba9f36133aae692c2477d53cd4f0f4ff`
- Four complete native-SQG rank receipts
- Loaded and executed layers: 3 through 78
- Fatal audit matches: 0

The qualification bundle's `SHA256SUMS` validates all sealed files.

### Estonia

Five of five runs are correct, with zero errors and zero truncations. Average
completion length is 3,371.6 tokens. Average generation speed per run is
`43.5170233617816` tokens per second.

### LAVD

Five of five runs are correct under the published tolerance: 4 exact and 1
near, with zero errors and zero truncations. The near answer is `71, 45.75`
against expected `72, 46`. Average completion length is 16,682.2 tokens.
Average generation speed per run is `22.087665860284208` tokens per second.
This result must not be described as 5 exact.

### MTP3

The runtime generated 106,204 tokens, drafted 88,704 tokens, and accepted
76,631 draft tokens. Aggregate acceptance is `0.863895652958153`. Acceptance
by speculative position is `0.9242762445887446`, `0.8623511904761905`, and
`0.8050595238095238`.

The MTP EXL3 patch maps the speculative `mtp_block` module prefix to the
canonical layer-78 checkpoint prefix. A real-image static check passed for both
prefix forms. Full pytest did not run because a dependency was unavailable, so
no full-pytest result is claimed.

## Reproduction

The browsable reproduction closure contains:

- the no-shortcut campaign controller and per-wave scripts;
- exact QSRT and KQuant source snapshots and patches;
- layer allocation, parity, materialization, and TP1 oracle validators;
- the final mechanical receipt binding 75 manifests/oracles, 74 K96 parity
  receipts, the layer-3 K48 exception, assembly manifest, and codec receipt;
- checkpoint assembly and 76-layer codec validation;
- full-vocabulary KLD and hidden-replay tools;
- the exact-r11 image build, MTP3 prefix patch, Compose configuration, and
  qualification scripts;
- the exact measured v0.4.29 benchmark bytes in a deterministic gzip archive,
  with decompressed SHA-256 verification and a clearly labeled non-identical
  ASCII derivative;
- Estonia, LAVD, rank, server-audit, and MTP3 receipts; and
- SHA-256 manifests.

The human procedure is
`reproduction/docs/K96_COUPLED_DISTRIBUTED_REPRODUCTION_20260814.md`. The
machine contract is
`reproduction/machine/k96tail-distributed-campaign.json`.

## Public file boundary

Server-side copy commit
`0c38e683eda27ca84982e3d513c89dd780dcdb22` copied and verified 390
byte-identical files totaling 25,685,857,224 bytes from the frozen source
revision. Routed layers 51 through 77 were already present, and MTP layer 78
was copied. The active targeted upload restarted at
`2026-08-15T05:39:16-04:00` with one outer client and default adaptive Xet.
Adaptive concurrency began at 2. Early sustained evidence showed 22.09 Mbit/s,
success ratio 1.0, and zero errors. No completion ETA is part of this card.

The repository is unsupported as a complete downloadable model until all
missing layers finish, the canonical full-folder verification passes, this
research-only quality disclosure is published, and a hash-bound public
revision passes anonymous representative-file hash checks.

The follow-on upload does not wait for the intentionally stopped qualification
server. It excludes this `README.md` and `HUB_FILE_VERIFICATION.json`, uploads
every remaining file with two workers, and verifies the resolved Hub revision
against local bytes. LFS files must match SHA-256 and size; Git files must
match Git blob SHA-1 and size. The verification receipt is uploaded only after
every file matches. No passing public verification receipt exists at this
documentation snapshot.
