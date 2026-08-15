---
license: other
license_name: glm-5.2-derived-calibration-data
license_link: https://huggingface.co/zai-org/GLM-5.2/blob/b4734de4facf877f85769a911abafc5283eab3d9/LICENSE
pretty_name: GLM-5.2 BMM-Law SQG Hessian and Calibration Archive
tags:
- glm
- quantization
- hessian
- sqg
- w4a8
- bmm-law
---

# GLM-5.2 BMM-Law SQG Hessian and calibration archive

This is the complete hash-bound reproduction archive for the GLM-5.2 SQG
W4A8 build. It is **3.096 TB logical**, but only about **1.128 TB is unique
local payload**. The larger displayed size comes from retaining the raw
capture names and two canonical zero-copy views as separate repository paths.

For ordinary reuse, prefer the deduplicated **1.154 TB** canonical repository:

[`brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians-Canonical`](https://huggingface.co/datasets/brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians-Canonical/tree/canonical-v1)

The accepted full archive remains sealed at tag
`accepted-glm52-sqg-w4a8-b300-r1` and is the authority for exact frozen-path
campaign replay.

## Coupled-Hadamard K96-tail reuse and result

The GLM-5.2 coupled-Hadamard K96-tail campaign consumed the dataset at pinned
revision `a05b3b92d749f6a641af5cfd52de2b4720380dfd`. Publishing this card advances
the dataset `main` branch, so `main` must not be described as remaining at that
consumed revision. The accepted tag
`accepted-glm52-sqg-w4a8-b300-r1` remains unchanged.

The re-encode is implemented with residual H512, H128 before the nonlinear
boundary, H128 after the nonlinear boundary, exact GLM `silu(gate) * up`, H13
local alpha `0.25`, and candidate-conditioned downstream H2. It consumes the
saved H13 preparation and activation/routing captures described below:

```text
derived/direct_h13/layer_NNN/preparation/w4a8_global_h13.safetensors
capture_view/layer_NNN/
```

Layer-global H13 is blended with expert-local evidence streamed from the saved
capture. Candidate-conditioned down H2/B is also constructed from the saved
capture. The weight source is the frozen SQG checkpoint
`brandonmusic/GLM-5.2-SQG-W4A8@593dd0d2de6f79ce4e65303930c22c75e1359d44`.
The re-encode did not download or read official BF16 routed-weight shards.
Files such as `capture_view/layer_NNN/hidden.bf16.bin` are saved activation
captures, not BF16 model weights.

The mechanically complete checkpoint uses this routed rate:

| Layer range | Treatment | Routed bpw |
|---|---|---:|
| 3 | Coupled K48, 720 K3 plus 48 K4 tensors | 3.0625 |
| 4 through 77 | Coupled K96, 672 K3 plus 96 K4 tensors | 3.125 |
| 78 | Preserved source MTP, 384 K3 plus 384 K4 tensors | 3.5 |

Target layers 3 through 77 average `3.1241666666666665` bpw. All 76 routed
layers, including MTP layer 78, average `3.1291118421052633` bpw.

Status: **research-only**. The checkpoint and TP4/DCP4/MTP3 runtime are
mechanically qualified, but full-model quality does not pass the KLD release
gate. On the same fixed 2,048-token input, all 2,047 causal positions, full
154,880-token vocabulary, no trimming, and TP4/PP1/DCP1:

| Metric | Frozen SQG source | Coupled K96-tail candidate |
|---|---:|---:|
| Mean KLD | `0.07583317451217256` | `0.1401771516114036` |
| p99 KLD | `1.397577404976` | `2.480538845062256` |
| Worst-1% CVaR | `2.207112874304` | `4.160942645300002` |

Candidate minus source mean is `0.06434397709923104`. The candidate is
`1.84849x` the source mean, or 84.849 percent worse. The lower mean, lower p99,
and lower worst-1% CVaR gates fail. This dataset remains a valid, hash-bound
calibration archive; that validity does not imply that every model built from
it improves end-to-end KLD.

The original preregistered hidden-replay maximum-position gate also fails:
approximately `0.0035558` observed against a `5e-4` limit. A later
post-observation `5e-3` operational limit passes and is research-only.

- Model repository:
  [brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail](https://huggingface.co/brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail)
- GitHub source, reproduction, and evidence:
  [coupled_hadamard_k96tail](https://github.com/brandonmmusic-max/glm52-sqg-mcg-experiments/tree/agent/coupled-hadamard-k96tail-publication/coupled_hadamard_k96tail)
- Full KLD receipt SHA-256:
  `7979c9c8b0c81714cd38e225646e42a88b2cb8eb03232be271373255c506a408`
- TP4/DCP4/MTP3 quality summary SHA-256:
  `4f19ba5e4a8676c80bc49e89d346b0985faa209f14bdd6d9713e9ee6c4397f57`

The public model file set remains unsupported as a complete model until routed
layers 3 through 50 finish uploading and a hash-bound public revision passes
anonymous verification.

## START HERE: the actual numerical Hessian files

If you want the **ready-to-use, precomputed Hessian tensors**, use these paths.
You do **not** need to download the 3.096 TB archive.

### Routed MoE H13 (layers 003–078, about 11.6 GB total)

Exact path pattern:

```text
derived/direct_h13/layer_NNN/preparation/w4a8_global_h13.safetensors
```

**[Open the actual routed H13 folder](https://huggingface.co/datasets/brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians/tree/accepted-glm52-sqg-w4a8-b300-r1/derived/direct_h13)**

For example, layer 077 is
[`derived/direct_h13/layer_077/preparation/w4a8_global_h13.safetensors`](https://huggingface.co/datasets/brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians/blob/accepted-glm52-sqg-w4a8-b300-r1/derived/direct_h13/layer_077/preparation/w4a8_global_h13.safetensors).
The JSON beside every SafeTensors file records construction, source rows, and
hashes.

Download only these routed H13 tensors:

```bash
hf download brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians \
  --type dataset \
  --revision accepted-glm52-sqg-w4a8-b300-r1 \
  --include 'derived/direct_h13/**' \
  --local-dir GLM-5.2-Hessians
```

### Dense K6 Hessian inputs

Exact paths:

```text
raw_capture/dense_h_pp_rank_*.safetensors
raw_dense_h/layer_078/dense_hessian.safetensors
```

**[Open the ordinary dense-H files](https://huggingface.co/datasets/brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians/tree/accepted-glm52-sqg-w4a8-b300-r1/raw_capture)**
or **[open the MTP78 dense Hessian](https://huggingface.co/datasets/brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians/tree/accepted-glm52-sqg-w4a8-b300-r1/raw_dense_h/layer_078)**.
Prepared/encoded dense-K6 artifacts are under
[`derived/k6`](https://huggingface.co/datasets/brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians/tree/accepted-glm52-sqg-w4a8-b300-r1/derived/k6).

For the smaller, nonduplicated dataset, use the same paths in the
[`Canonical` repository](https://huggingface.co/datasets/brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians-Canonical/tree/canonical-v1).

## Inputs for expert-local H13 and candidate H2/B

Expert-local H13 and candidate-conditioned down H2/B were constructed by
streaming routed activations; they are not one giant precomputed matrix per
expert. The canonical activation/routing evidence is:

```text
capture_view/layer_NNN/          # ordinary layers 003 through 077
capture_view_mtp78/layer_078/    # MTP layer 078
```

Use [`capture_view`](https://huggingface.co/datasets/brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians/tree/main/capture_view) and
[`capture_view_mtp78`](https://huggingface.co/datasets/brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians/tree/main/capture_view_mtp78). Each ordinary layer provides
`hidden.bf16.bin`, `topk_ids.u8.bin`, `topk_weights.f32le.bin`, and canonical
row-order metadata.

## Why three large views exist

- `raw_capture/layer_003..077`: original capture filenames.
- `raw_capture/sqg-view/`: zero-copy SQG adapter view inside the capture tree.
- `capture_view/`: the same canonical view at its stable campaign path.

These were hardlinked on the build machine, but Hugging Face counts every path
toward displayed logical size. Do not download all three unless you need an
exact audit of the capture-to-view transition.

## Reproduction source and frozen decisions

- `reproduction/`: exact calibration, encoding, quantization, scheduling,
  materialization, and publication source.
- `derived/final_profiles/`: frozen native profile choices.
- `derived/preparation/`: preparation contracts.
- `derived/lineage/reproducibility/`: campaign and source-hash bindings.
- `HESSIAN_DATASET_MANIFEST.json`: complete file census and identities.

The build used BF16 source revision
`b4734de4facf877f85769a911abafc5283eab3d9`, TP1/PP8 capture, canonical prompt
row order, `0.75 * H_layer + 0.25 * H_local,e`, frozen independent per-tensor
K3/K4 assignments, exact GLM `SiLU(gate) * up`, candidate-conditioned W4A8
down calibration, and dense SQG K6 for 380 selected non-routed matrices.

This repository does not relicense GLM-5.2 or third-party source code.
