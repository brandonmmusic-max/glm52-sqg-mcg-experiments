# Late H13 alpha run seals

These are the byte-identical run seals from all five completed late-block
alpha arms. The four nonwinning roots were archived before their model-scale
payloads were scheduled for cleanup; the selected diagnostic alpha-0.25 final
seal is included explicitly as the comparison anchor. The seals bind layers
74--77, the exact per-tensor K3/K4 census, zero MCG treatment tensors,
selected profile identities, layer artifact hashes, and packed layer-shard
hashes.

| Expert-local alpha | File | Run-seal ID |
|---:|---|---|
| 0.00 | `alpha000_run_seal.json` | `7fb34ddb484425d5af2d46e78134de3e01af0d670de048a92790e16289c36f01` |
| 0.25 | `alpha025_final_run_seal.json` | `45bd96310da349fe55bc70205000db528c620dc62f5fcde3567e5b7c2c29b56d` |
| 0.50 | `alpha050_run_seal.json` | `a29774b295ef8b3e2a8599b31fa95b1fbece76b2ba7259ebdf8d3dfd2db5fb31` |
| 0.75 | `alpha075_run_seal.json` | `362378b93f3307cbe6ba4ef5b15ece1ececc4813bd3089879c70fa5520b2d7d5` |
| 1.00 | `alpha100_run_seal.json` | `20b7e8be13156ad5cd6b05e68c9ed0ecc7dfeab2af9927a5a7edfa17b3a7fd7f` |

The selected diagnostic alpha-0.25 seal is also separately preserved under
[`published_evidence/contiguous_late/encoding/run_seal.json`](../contiguous_late/encoding/run_seal.json).
Both published copies hash to
`567066231b882732d5bc82a093123002573a9cc8cf507ec763e2c2e7184606cf`.
There is no distinct preparation-stage alpha-0.25 run seal; the sealed record
was produced by `fresh-sqg-contig-late-final-a025-r1`. The seals are
provenance records, not substitutes for the omitted packed tensor payloads.
