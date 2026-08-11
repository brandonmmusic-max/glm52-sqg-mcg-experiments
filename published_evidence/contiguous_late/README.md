# Test 10 late-block evidence boundary

This directory publishes the compact evidence available at the current Test 10
boundary. It does not contain a completed profile search, SQG encoding, model
candidate, KLD run, or speed result.

## Completed stages

- The capture manifest is sealed with `complete: true` for layers 74--77 and
  253,863 rows per layer.
- The frozen whole-document plan contains 131 fit documents/150,368 rows, 41
  selection documents/51,232 rows, and 45 holdout documents/52,263 rows.
- Each of the four preparation logs reports 256 expert permutations, fit-only
  construction, no selection or holdout use, zero MCG inputs, and zero
  fallback.
- The real layer-74/expert-0 scale and candidate-H2 smoke reports
  `complete: true`; it is explicitly marked ineligible for profile selection
  and final materialization, as a smoke artifact should be.

## Pending stages

- completion and selection of the layer profile search;
- full alpha-0.25 SQG encoding of all 3,072 block tensors;
- zero-MCG payload validation and candidate materialization;
- repeated final-logit KLD and signed per-position tail analysis; and
- layer-by-layer routing, signed top-8 output, hidden-state, and residual trace.

No conclusion about contiguous error propagation follows from the completed
capture and preparation stages alone.

## Published hashes

| File | SHA256 |
|---|---|
| `capture_manifest.json` | `d8eff2f48c205414a4c653efbe53df1de5ae6822259cf89184ea509d9c3297d6` |
| `layers/layer_074_manifest.json` | `7b65528a8b3dc63da611424ba6da63bb5c2412301bfdd83d5a3cd1c5181e49e4` |
| `layers/layer_075_manifest.json` | `40bc6c5ef46884e460cb10b45cece9d77fb13d57c1c79808e498169817a3d982` |
| `layers/layer_076_manifest.json` | `5ebaafef4c9b2ae50a703fff0b35542b4f1530284a0913d7f2d0804fe824b0f6` |
| `layers/layer_077_manifest.json` | `f4726e3f3c7e7e7e40ee82da86a82093169db01aeb84cb469d6f2aba8ff7b113` |
| `preparation/prepare-74.log` | `f110e6abaa8b0a1442e275c098f582a91d9d16f9c19f63c18c6283b92d5e82dd` |
| `preparation/prepare-75.log` | `a4b97e01bf9a29e4d31f0ffde2c8e22f13d1b5f4221e7a6e2cc475930ed0c21c` |
| `preparation/prepare-76.log` | `310e9633f103c89c66482754021efaa2983039be93614a9faa444e81dc042666` |
| `preparation/prepare-77.log` | `989b6e1611ede656d1194e3ab2f52e2a5f1bc5dc4477c4d4bee50cf49171bd66` |
| `preparation/absolute_gate_scale_smoke.json` | `1db17bea6a75d68f57b1f1259a065e358a2e2f6f364908c6ac88494007bb0279` |
| `preparation/absolute-scale-smoke.log` | `cc7be09307a4d6a9bc257bf3c9615760fefdad37cfeafd7986d6f2159043bb0d` |
| `preparation/successor_preflight_receipt.json` | `b4f81bf57d457fb160259fad35eaa95b2ac3c375050136cc927b63011eec37ea` |

The repository-wide [`SHA256SUMS`](../../SHA256SUMS) binds these files again
after publication assembly.

## Excluded payloads

The capture's BF16 hidden-state binaries, routing arrays, BF16 source shards,
prepared Hessian tensors, per-expert permutation records, smoke tensor payload,
and all model payloads remain local. The in-progress profile-worker logs are
also intentionally excluded so that an incomplete worker launch is not
mistaken for a completed search result.
