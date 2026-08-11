# Test 10 late-block evidence boundary

This directory publishes compact evidence through the current Test 10
boundary. Capture, profile selection, alpha-0.25 SQG encoding, sealing, and
candidate materialization are complete. It does not contain an accepted KLD
boot, propagation result, or speed result.

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
- All 16 preregistered profile cells per layer were encoded and scored. The
  frozen selections are draw-00 identity for layers 74--76 and draw-03 identity
  for layer 77.
- Alpha 0.25 means 25% expert-local H13 and 75% layer-global H13. The run seal
  reports 1,024 experts, 3,072 SQG tensors, 1,536 K3, 1,536 K4, and zero MCG
  tensors in treatment layers 74--77.
- The candidate is materialized separately with no protected-source mutation.

## Pending stages

- repeated final-logit KLD and signed per-position tail analysis; and
- layer-by-layer routing, signed top-8 output, hidden-state, and residual trace.

The first KLD attempt failed during engine initialization, before inference,
because sealed historical arguments expected reserved layers `6,28,52` while
the dynamic late treatment correctly produced no reserved layer. `runs.jsonl`
is empty and no accepted record, summary, per-position tensor, or KLD value was
produced. That attempt is excluded. The runner now derives the expectation from
the treatment layers and allows a fail-closed same-directory retry, but the
retry was not launched by this publication update.

No conclusion about contiguous error propagation follows from the completed
construction stages or the rejected pre-inference attempt.

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
| `profile_search/layer_074/selection.json` | `50b74b115cd657a1884a6b17841709826058489a175ac0a654046dfd9ea9ba08` |
| `profile_search/layer_075/selection.json` | `85825d7592cd78116d45c4f0b632b0e245a8921c6d64c8a59b4b9e58b82ac0f2` |
| `profile_search/layer_076/selection.json` | `d57a874c0bd858e824ec450da38bafc2a3523e78f663d9eb46a7741f9579ab06` |
| `profile_search/layer_077/selection.json` | `1f2104d2dc6803eabc0fd6ef1d5d28e33bdce2bb15dcd5eda9d76b465c426547` |
| `encoding/run_seal.json` | `567066231b882732d5bc82a093123002573a9cc8cf507ec763e2c2e7184606cf` |
| `encoding/materialize-receipt.json` | `b9fd238a90370d4ea0fae4dfbb150390ae23f71e5ed11c0a3db1f54d9b35f3ca` |
| `candidate/MANIFEST.json` | `5720c1aba18af0917d142f1f755cd03a8311591a638c80cf53d4f410791f30b6` |
| `kld_failed_run1/run1.log` | `07d398cf6cf3abe623db1876692aab0ee0dd8961776b80aa583ced0400485593` |
| `kld_failed_run1/runs.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The repository-wide [`SHA256SUMS`](../../SHA256SUMS) binds these files again
after publication assembly.

## Excluded payloads

The capture's BF16 hidden-state binaries, routing arrays, BF16 source shards,
prepared Hessian tensors, per-expert permutation records, 1,024 expert tensor
payloads/manifests, assembled model tensors, unchanged model hard links, runtime
caches, and all other model-scale payloads remain local. The compact profile
decisions, layer assembly results, seals, candidate manifest, failure log, and
hash lists preserve the review boundary.
