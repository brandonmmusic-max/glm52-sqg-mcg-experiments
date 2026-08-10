# Capture teacher identity seal

The activation teacher is read-only, but its legacy `MANIFEST.json` is not an
identity authority: the checkpoint itself declares
`payload_hash_verification=false`.  Fresh SQG capture therefore uses the
independent receipt at `evidence/teacher_model_identity.json`.

The receipt closes exactly 239 loader-consumable files:

- all 156 unique safetensors payloads referenced by
  `model.safetensors.index.json`;
- all 75 `r7-experts-layer-003..077.json` bit-map/loader sidecars declared by
  `quantization_config.json`;
- `model.safetensors.index.json`, `config.json`,
  `quantization_config.json`, `tier_bitmap.json`, `tokenizer.json`,
  `tokenizer_config.json`, `chat_template.jinja`, and
  `generation_config.json`.

Every entry records the exact byte length and an independently computed
SHA256.  The canonical `seal_sha256` covers the sorted compact-JSON seal body;
the separate receipt-file SHA256 covers its serialized bytes.  Validation also
requires the embedded and external quantization configs to agree, binds each
R7 sidecar to its indexed shard and claimed shard hash, rejects duplicate JSON
keys, unsafe/non-top-level index paths, symlinks, unreferenced safetensors,
unexpected R7 sidecars, tokenizer alternatives, top-level remote code, and
alternate weight formats.

The receipt is create-once; the build command refuses to overwrite it:

```bash
python3 seal_teacher_model.py build \
  --model-root /home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED \
  --receipt evidence/teacher_model_identity.json \
  --workers 4
```

The mandatory capture gate is a full validation from the direct Python capture
process against the read-only mounted model, immediately before tokenizer/model
load.  It must use the hard-coded expected seal and embed the returned full
record in smoke/full runtime provenance:

```python
teacher_identity = validate_teacher_identity_receipt(
    args.model,
    Path("/work/evidence/teacher_model_identity.json"),
    expected_seal_sha256=(
        "e256a8c5c6e3734e47da61ecb7649bee"
        "596091178981ded0bbc00b69c50b766a"
    ),
    verify_mode="full",
    workers=4,
)
```

The sealed receipt has SHA256
`eb88fcd2bf66b0cfef195a232d9b81efcb107c400381b22c913a2c738413479e`.
On the build host, create-once hashing took 51.39 seconds and an independent
full validation took 42.58 seconds.  Those measurements cover 343,070,719,678
allowlisted file bytes, of which 342,683,459,652 are indexed payload-file
bytes.  The HF index declares 342,661,251,548 tensor bytes across 187,580
weight-map entries; the difference from payload-file bytes is safetensors
container/header overhead, so the two totals are intentionally not equated.

`verify_mode="metadata"` exists only for auxiliary post-preflight inspection;
it hashes all non-payload identity/sidecar files and checks payload paths and
sizes, but it is not an acceptable capture gate.  Smoke and full capture must
both use `verify_mode="full"`.
