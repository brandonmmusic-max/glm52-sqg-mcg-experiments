# Exact Infernal Invocation r11 qualification runtime

This directory is the hash-bound runtime and evaluation closure for the
coupled-Hadamard K96Tail checkpoint. The final serving base is **Infernal
Invocation r11 with MTP3**. Infernal Invocation r13 contributes native-SQG
donor changes only; neither r13 nor any v20 image is a permitted serving base.

## Hash-bound identity

- base digest:
  `sha256:01b973d1ae132882bcc1bf62ea232f6aabe649dd4a89b961d81f3c41cc53f971`;
- final image: `verdictai/glm52-k96-ii-r11:20260815-tpfix-mtpfix`;
- final image ID:
  `sha256:ab6bd60716b0a8e453b6345cb10e43e79726d92729b1f29058e31d7cc1c67def`;
- topology: TP4/DCP4/MTP3, NVFP4 DS-MLA KV,
  `MAX_MODEL_LEN=262144`;
- vLLM base: `ce5f50f6d01b02336c4207f11277fd7bedacb4d6`;
- native-SQG donors: vLLM PR 315 at `ca966847`, B12X PR 197 at
  `b234532`;
- supplemental MTP3 loader patch SHA-256:
  `658cbdd678774b0a1167c6244ff78d627b77613709f610e26e8d0e1933cfa03e`.

The complete vLLM patch and supplemental MTP3 patch passed ordered
`git apply --check`/apply replay on the pinned base. `py_compile`, Ruff, and
diff-check passed. A real-image static test resolves both the canonical
layer-78 prefix and the speculative `mtp_block` prefix. Full pytest did not run
because a test dependency was unavailable, so no passing full-pytest result is
claimed.
The TP4 fix reassembles global gate/up coordinates before the coupled H128
closure and exact `silu(gate) * up` activation.

`verify_runtime.py` is an image gate. The host vLLM ABI lacks the required
exact-r11 symbols, so an unqualified host invocation is not valid. Re-run the
gate inside the candidate image:

```bash
docker run --rm \
  --entrypoint /opt/venv/bin/python \
  verdictai/glm52-k96-ii-r11:20260815-tpfix-mtpfix \
  /opt/ii-r11-k96/verify_runtime.py
```

The archival patches are stored as deterministic `.patch.gz` files. Extract
each with `gzip -dc`, verify its documented decompressed SHA-256, and apply the
extracted bytes to the pinned base. The readable repository contains no
silently normalized patch presented as exact.

## Evaluation contract and sealed results

`deploy/run-quality-5x.sh` runs Estonia five times sequentially and LAVD five
times at concurrency five with `local-inference-lab/llm-inference-bench`
v0.4.29, commit `0b4185b5b435e948b199c9077a00b084864aa963`, script SHA-256
`59dd767c933e06f9724a84a8883d2aac156252dbbc279ce155658005d27424d7`.
The exact measured bytes are in deterministic archive
`benchmarks/llm_decode_bench.py.gz`, archive SHA-256
`ea6b898216417bff1b35f0522536c7c46b9ca2abb5ac340704675cd82830d520`.
The runner extracts and verifies the decompressed hash before execution. The
adjacent ASCII derivative is readable but explicitly not byte-identical.
The bound reuse-capable runner SHA-256 is
`807afc1518207faece5a378c391561e6a23fd5fbfada8c3ff1361a18574415eb`;
the idempotent post-task sealer SHA-256 is
`68b59ddfe3fb667b60ac774135789bbc7d75ce462812659557607b50541c0b49`.
The historical hidden-replay-gated characterization launcher is retained for
audit at `deploy/run-quality-after-hidden-measurement-exact-r11.sh`, SHA-256
`d9195e5a8b7142e2bd70891b73fae6e377b47cb52f7d5ac027f957b876a962cd`;
it is not the final sealed-run entrypoint.

The sealed run is
`exact-r11-tp4dcp4mtp3-20260815T084427Z`. Its quality-summary SHA-256 is
`4f19ba5e4a8676c80bc49e89d346b0985faa209f14bdd6d9713e9ee6c4397f57`,
its MTP summary SHA-256 is
`d8a7f22f6da05423a00d972e186deb17ba9f36133aae692c2477d53cd4f0f4ff`,
and its original measured run-manifest SHA-256 is
`bf57161cec69b9e8e8ebb5e705c1c5c6ed88c80046364c2a07a8882e60d0e3a9`.
That exact manifest is stored as `SHA256SUMS.measured.gz` and the listed value
is its decompressed hash. `qualification.complete` is present and the
publication `SHA256SUMS` verifies every stored file. The four
rank receipts (PIDs 767, 825, 926, and 1030) are complete TP4,
`full-w4a8`, routed-expert, no-A16-fallback receipts for loaded/executed layers
3--78. The fatal-audit file has zero matches.

Estonia completed 5/5 valid and correct with zero failures/truncations; raw
JSON SHA-256 is
`62551513dabdfdb0c389edc15220e03d3d95d12e5f85e26125ca0ba8962b6e92`.
Aggregate generation was `42.67293371871177 tok/s`. Its reported
`148,287 tok/s` prefill scout reused cache warmed by rejected overlapping
diagnostics and is not a cold or uncached prefill result.

LAVD completed 5/5 valid and correct with four exact results and one near
result (`71,45.75` versus `72,46`). Raw JSON SHA-256 is
`2a9e2772049be73e59d14a31f1f47b17dbcd871f536af6a7f09febc8ab29b4e1`.
Aggregate generation was `22.13984480814618 tok/s`; there were no failures or
truncations.

The original runner's post-task assertion stopped at layer 77, although final
MTP3 correctly loaded and executed preserved source-SQG MTP layer 78. The
bound runner records the corrected inclusive 3--78 invariant, supports
`QUALITY_REUSE_RESULTS_DIR`, and captures final MTP metrics without rerunning
the already-valid task requests.

## KLD scopes

`kld/` performs full-vocabulary teacher-forced KLD on all 2,047 next-token
positions. `hidden-replay/` captures `[2048,6144]` BF16 states immediately
before the unchanged `[154880,6144]` BF16 LM head and replays the same
positions. These sealed receipts remain bound to the earlier exact-r11 `tpfix`
measurement image ID
`sha256:79dfbf6e697a1081016a3256ff5c96b1c2d6dcddd6f04662c9c693881310fa87`;
they are not relabeled with the final MTP3 image.

Candidate mean KLD is `0.1401771516114036`, versus frozen source mean
`0.07583317451217256`. The candidate is `0.06434397709923104` higher,
`1.84849x`, or 84.849 percent worse. Overall model quality is
**research-only** even though the runtime is **qualified**.

## Publication orchestration

The destination is
`https://huggingface.co/brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail`.
Server-side copy commit `0c38e683eda27ca84982e3d513c89dd780dcdb22`
copied and verified 390 byte-identical files totaling 25,685,857,224 bytes from
`brandonmusic/GLM-5.2-SQG-W4A8` revision
`593dd0d2de6f79ce4e65303930c22c75e1359d44`. Layers 51 through 77 were
already present on the destination, and preserved MTP layer 78 was copied.
Public completeness remains **unsupported** until a complete file-by-file Hub
verification receipt exists.

`deploy/accelerate-hf-routed-upload.sh` is only a targeted routed first stage.
Its SHA-256 is
`a4469e0f1fdcdba70b95e41fe442edd149bbabfc192b78c095115966c1a733bc`.
The active targeted upload restarted at `2026-08-15T05:39:16-04:00` with one
outer client. It unsets `HF_XET_HIGH_PERFORMANCE` and
`HF_HUB_DISABLE_XET`, so Xet uses default adaptive mode. Adaptive concurrency
began at 2. Early sustained evidence showed 22.09 Mbit/s, success ratio 1.0,
and zero errors. This is operational evidence, not a completion claim or a
reproducible ETA. The targeted stage did not prove full checkpoint
completeness. The
authoritative full-folder stage is `deploy/complete-final-model-upload.sh`,
SHA-256
`f6b3352db4a288793254bd0c2656e31c805e3718b23fe2239a6649fcd7646d46`,
run by persistent unit `glm52-k96tail-final-model-upload.service`, unit SHA-256
`f73b3a35718ff43955c37f4a9f9425b9f117576b260fba684130ce6b8ed7e1e8`.

`deploy/wait-and-publish-hf-release.sh` SHA-256
`6c57cad0226dbd77e7f9eb660e42c344e58a4ed4868e07b4d0a7e01c754c35ae`
is run by persistent unit `glm52-k96tail-final-hf-publication.service`, unit
SHA-256
`ff033d7d3573bedf282b1535065aaf9f1a9d719cd53f0cd1fc89fb74be4eda62`.
It waits for canonical upload plus an explicit readiness marker, publishes the
browsable closure and card, anonymously verifies file presence and exact card
bytes, then promotes and publishes final provenance. The readiness marker is
intentionally absent until local card/provenance/manifests are finalized.

The follow-on wrapper `deploy/wait-and-upload-final-model.sh`, SHA-256
`ce467eb1218cc8a5d9334916ea39d2d3d816b46df73cd5496b1e97b9576ac63a`,
does not wait for the intentionally stopped qualification server. It uploads
with two workers while excluding `README.md` and
`HUB_FILE_VERIFICATION.json`, then invokes
`deploy/verify-final-hf-model.py`, SHA-256
`aebc8b97fe332eb5b086e8f62b698e631fa10814f5fdf9504d201abb366efa11`.
The verifier matches every other local file to the resolved Hub revision. LFS
files use SHA-256 and size; Git files use Git blob SHA-1 and size. It writes the
verification receipt atomically, and the wrapper uploads that receipt only
after `complete=true`. Shell syntax, Python compilation, and diff-check pass.
No passing public verification receipt exists at this documentation snapshot.

## Contents

- `Dockerfile`, `PROVENANCE.json`, and `verify_runtime.py`: exact-r11 build and
  fail-closed static checks;
- `patches/`: complete vLLM/B12X changes and supplemental MTP3 loader patch;
- `deploy/`: KLD, serving, quality, sealing, upload, and publication scripts;
- `orchestration/systemd-user/`: persistent canonical upload/publication units;
- `benchmarks/`, `kld/`, and `hidden-replay/`: frozen evaluation inputs and
  implementations;
- `SHA256SUMS`: byte binding for this closure.
