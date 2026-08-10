# Preregistered runtime controls

The primary candidate uses `r33_exact_runtime.args`. Selected layers 6, 28,
52, and 77 contain only fresh SQG calibration and encoding results. Their
legacy MCG payloads, transforms, scales, permutations, seeds, decoded weights,
and sidecar lineage are forbidden. The frozen per-tensor K3/K4 assignment is
the sole inherited selected-layer quantization control. Layers 6, 28, and 52
reserve the same three positions they occupied in the preserved 48-layer MCG-
fused set; layer 77 was already native/nonfused. The loader must close at 45
actually fused MCG layers plus 3 reserved slots.

`run_fresh_kld.sh` accepts only a candidate bound to the external four-layer
run seal, the sanitized bit contract, and the full teacher-identity receipt.
It does not accept the earlier offline hybrid audit as provenance. Before a
run enters `runs.jsonl`, its exact 2,047-position `KL(ref||model)` tensor is
independently checked for SHA256, schema, direction, dtype, shape, finite
nonnegative values, and agreement with the emitted scalar mean.

Two controls use the same image, overlay, DCP4/A2A/64 interleave, FP8 KV,
prompt, saved BF16 logits, and KLD code:

1. `r33_overlay_historical.args` loads the protected MCG checkpoint read-only
   with all 48 historical layers fused and zero reserved slots. One run checks
   that the common overlay itself reproduces the preserved baseline path.
2. `r33_native_mcg_control.args` loads that same protected checkpoint
   read-only but forces exactly layers 6, 28, 52, and 77 onto native MCG
   dispatch. Layers 6, 28, and 52 reserve their former fused slots. This
   isolates the dispatch change that is unavoidable for SQG.

The native-MCG arm is mandatory for this experiment, regardless of the fresh
candidate delta. It must use the protected MCG checkpoint read-only, a distinct
container name, a distinct result root, and a new per-run runtime cache; it may
not mount the SQG candidate or any candidate result/cache path. Interpretation
uses the paired per-position output against the same BF16 reference; scalar
KLD subtraction alone is not treated as a causal decomposition. The control
environment fails if any selected layer contains SQG, lacks an exclusive MCG
marker, or fused-slot accounting does not close exactly. A candidate summary
without this separate control remains a candidate-arm result, not the completed
causal comparison.

## Native-MCG runner

`run_native_mcg_control.sh` is a separate fail-closed runner; it accepts no
model path, so it cannot be redirected to the SQG candidate or another MCG
checkpoint. It mounts only the hard-coded protected checkpoint at `/model:ro`.
It requires the exact sealed native-control args (SHA256
`fa572621970a0b0a3ab4fdba4af3f783eef7f9c2d7975744c8a1e09e9f67f598`),
the exact r33 image and runtime overlay, and exactly five fresh boots. The
complete teacher-receipt runtime file set and the selected treatment files are
rehashed before every launch. Production must remain stopped, and each boot
uses a new runtime cache and seeded offline WikiText cache.

Use a result root that is disjoint from both the candidate results and the
project. `PREFLIGHT_ONLY=1` performs all host validation and hashing but does
not launch a container:

```bash
NATIVE_MCG_RESULTS_ROOT=/a/new/native-mcg-control-results \
STAMP=fresh-sqg-native-mcg-r1 \
PREFLIGHT_ONLY=1 \
  evaluation/run_native_mcg_control.sh
```

Run the five-boot arm with the same command without `PREFLIGHT_ONLY`. The
runner accepts a result only if every boot emits all four native-MCG dispatch
proof lines, the three preserved-slot reservation lines, exact
`actual=45/reserved=3/total=48` accounting, zero SQG dispatch lines, and a
separately validated 2,047-position `KL(ref||model)` tensor.

## Paired interpretation

After both five-run summaries exist, run the analyzer into a fresh output
directory:

```bash
python3 evaluation/analyze_native_mcg_pair.py \
  /a/candidate-results/summary.json \
  /a/native-mcg-control-results/summary.json \
  --json-output /a/paired-results/sqg-vs-native-mcg.json \
  --tensor-output /a/paired-results/sqg-vs-native-mcg.safetensors
```

The analyzer rehashes and independently reads all ten per-position tensors,
requires identical reference, prompt, image, overlay, and regime bindings, and
publishes both five run-paired delta vectors and the across-run mean delta at
each position. Its sign is `candidate SQG - native MCG`, so a negative mean is
better for SQG. Position quantiles, the fraction improved, and a deterministic
circular-block bootstrap are reported in addition to scalar closure. The
bootstrap remains a within-one-prompt description; it is not evidence of
generalization across text or tasks.
