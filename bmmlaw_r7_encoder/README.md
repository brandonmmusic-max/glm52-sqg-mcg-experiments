# Round 7 routed-expert encoder draft

Marker: `CODEX_ROUND7`

This package is the draft, layer-sequential TR3-v4 encoder. Importing the
package is inert. The checked-in tests use synthetic tensors and temporary
files only; no real checkpoint encode, serving container, or production
service was run.

## Locked invariants

- Replacement names are limited to 256 routed experts x three projections in
  layers 3..77; layer 78/MTP is outside the replacement domain
  (`r7_encoder/constants.py:12-20`, `r7_encoder/assemble.py:42-53`).
- Every layer has 768 integer choices in `{3,4,5}` and an exact sum of 2,688,
  which is exactly 3.5 bpw because the three projection matrices have equal
  element counts (`r7_encoder/constants.py:29-46`,
  `r7_encoder/allocation.py:142-237`).
- Allocation consumes exact accumulated float32 routed-weight mass and measured
  3/4/5 round-trip sensitivity; duplicate experts within a token are rejected
  (`r7_encoder/routing.py:121-245`, `r7_encoder/allocation.py:52-140`).
- Down inputs are computed through the selected reconstructed gate/up pair, and
  down receives one full 2048 x 2048 covariance and one joint tensor encode
  (`r7_encoder/hessian.py:106-139`, `r7_encoder/layer.py:516-668`).
- The rolling state is prompt-preserving, corpus-plan-bound, and forwarded only
  through already installed predecessors. Layer 77 is sealed without a forward
  into layer 78 (`r7_encoder/state.py:291-518`,
  `r7_encoder/walk.py:474-546`).
- Intermediate permutations are baked into gate/up rows and down columns and
  checked by inverse and functional oracles
  (`r7_encoder/permutation.py:56-176`).
- Schema v2 stores full topology-neutral tensors, one pair of layer-shared
  residual vectors, and per-tensor bit metadata; TP slicing is deferred to the
  loader/converter (`r7_encoder/schema.py:104-283`,
  `r7_encoder/convert_v2_to_v1.py:168-376`).

## Deterministic runtime closure

Before Torch CUDA or Transformers imports, the run pins
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, disables hub kernels, and rejects conflicting
ambient values (`r7_encoder/determinism.py:21-43`,
`r7_encoder/walk.py:110-129`). The reference adapter then:

- verifies the complete inventoried Transformers, tokenizers, and Round 7
  source closure both before and after import;
- pins eager attention and eager expert dispatch;
- enables deterministic algorithms with `warn_only=False`, disables TF32, and
  rejects a CUDA context initialized before the sealed environment; and
- records the resolved wrapper/original expert callables and tokenizer class
  in a dispatch audit (`r7_encoder/transformers_runtime.py:197-350`).

The numerical inventory binds the CUDA-only LDL policy and exact TRELLIS
extension path/hash. CUDA factorization OOM aborts instead of selecting CPU,
and an ambient same-named extension at another path is rejected
(`r7_encoder/inventory.py:275-350`, `r7_encoder/trellis.py:79-164`). Each codec
encode also repeats extension reconstruction byte-for-byte
(`r7_encoder/trellis.py:342-400`).

After all 256 packed-decoded experts are installed, a layer oracle builds the
official eager `GlmMoeDsaNaiveMoe`, uses fixed routes with repeated expert hits,
runs it twice, and compares it against a separately accumulated packed-decoded
reference (`r7_encoder/transformers_runtime.py:1018-1253`,
`r7_encoder/glm52_backend.py:886-985`). Successor state additionally requires a
repeat oracle for BF16 hidden bytes and int32 DSA-index bytes
(`r7_encoder/transformers_runtime.py:1280-1369`,
`r7_encoder/state.py:463-488`). Real CUDA results remain **UNVERIFIED**.

## Preemption boundaries

- `CORPUS_PLAN.json` seals the ordered tokenizer output, token IDs, global row
  starts, and complete expected prompt domain before prompt zero is forwarded
  (`r7_encoder/transformers_runtime.py:379-489`,
  `r7_encoder/walk.py:174-276`).
- Carried layers 0..2 have a hash-checked per-layer/per-prompt prefix journal;
  a restart resumes at the first missing unit. Normal successor forwarding
  yields each prompt immediately so `PARTIAL.json` is updated before the next
  prompt (`r7_encoder/transformers_runtime.py:651-875`,
  `r7_encoder/transformers_runtime.py:1280-1369`).
- State partials bind the plan hash and full domain before work and refuse
  incomplete-domain commit (`r7_encoder/state.py:291-518`).
- Shared promoted full-round-trip candidates seal every sampled expert score
  before deriving the mass-weighted aggregate
  (`r7_encoder/search.py:329-382`).
- Row and final-expert shard/manifest pairs treat only incomplete or
  content-mismatched derivative pairs as recomputable. Identity or binding
  drift still fails closed (`r7_encoder/row_cache.py:31-123`,
  `r7_encoder/expert_cache.py:50-180`).

## Package map

- `glm52_backend.py`: exact routing sidecars, mass, row plans, BF16 source
  streaming, and validation around the runtime adapter.
- `transformers_runtime.py`: official single-CUDA-device GLM layer execution,
  exact corpus plan, prompt journals, packed installation, and runtime oracles.
- `walk.py`, `state.py`: integrated sequential state machine and durable stage
  journal.
- `layer.py`, `hessian.py`, `sensitivity.py`, `allocation.py`: probes,
  fixed-point allocation, full covariances, and final layer encoding.
- `search.py`, `permutation.py`, `rotations.py`, `search_artifact.py`: seeded
  multi-draw search, baked channel order, shared/per-expert vectors, and block
  scales.
- `trellis.py`: forced-vector, full-matrix adapter over the inventoried v31 MCG
  tile/LDLQ primitives.
- `schema.py`, `assemble.py`, `convert_v2_to_v1.py`, `oracles.py`: v2 emission,
  complete checkpoint assembly, transactional v1 conversion, and audits.

## Safe preparation and offline tests

Inventory outputs must be under `WORK`, never inside either model directory.
The runtime inventory must include the entire resolved `transformers` and
`tokenizers` package roots, including extension binaries
(`r7_encoder/inventory.py:101-106`, `r7_encoder/inventory.py:353-422`,
`r7_encoder/transformers_runtime.py:227-250`).

```bash
python3 -m r7_encoder.cli preflight \
  --carrier /path/to/GLM-5.2-EXL3-DENSE6-MTP78 \
  --src /path/to/GLM-5.2-BF16

python3 -m r7_encoder.cli inventory-checkpoint \
  --checkpoint /path/to/GLM-5.2-EXL3-DENSE6-MTP78 \
  --role carrier --out WORK/carrier.inventory.json

python3 -m r7_encoder.cli inventory-checkpoint \
  --checkpoint /path/to/GLM-5.2-BF16 \
  --role bf16-source --out WORK/source.inventory.json

python3 -m r7_encoder.cli inventory-numeric \
  --numeric-core /path/to/encode_tr3_v31.py \
  --extension /exact/path/to/exllamav3_ext.so \
  --out WORK/numeric.inventory.json

python3 -m r7_encoder.cli inventory-runtime \
  --file ./r7_encoder \
  --file /resolved/site-packages/transformers \
  --file /resolved/site-packages/tokenizers \
  --out WORK/runtime.inventory.json

python3 -m pytest -q r7_encoder/tests
python3 -m ruff check r7_encoder
python3 -m compileall -q r7_encoder
```

Audit or convert only already-produced draft artifacts:

```bash
python3 -m r7_encoder.cli audit-v2 \
  --manifest WORK/v2/r7-experts-layer-003.json

python3 -m r7_encoder.cli convert-checkpoint-v1 \
  --checkpoint /path/to/assembled-v2 --out /path/to/v1 --tp 4
```

The exact owner-run command, rental/storage plan, sample-layer stop gate, and
residual risks are in `R7_SYNTHESIS.md`. No command in this README was run
against real weights in Round 7.
