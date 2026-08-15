# K96 coupled-Hadamard reproduction index

The authoritative finalized publication contract is
`k96tail-distributed-campaign.json`. Local encoding, merge, assembly, codec,
mechanical evidence, KLD, hidden replay, runtime qualification, and task
receipts are sealed. The contract remains `complete=false` only because the
public tensor and publication revision fields are not fully sealed.

Method, orchestration, validation gates, incident history, current measured
ledger, and final results are in
`../docs/K96_COUPLED_DISTRIBUTED_REPRODUCTION_20260814.md`.

Run the non-mutating audit from the project root:

```bash
python3 scripts/audit_k96tail_campaign.py
```

Require the completed local closure while allowing the explicit Hub fields to
remain open:

```bash
python3 scripts/audit_k96tail_campaign.py --strict-local
```

After the remaining Hub fields are populated and `complete` is set to true:

```bash
python3 scripts/audit_k96tail_campaign.py --strict-complete
```

The executable flow is:

```text
prepare_vast_k96_node.sh
  -> run_vast_wave_and_upload.sh
  -> archive_vast_k96_evidence.sh
  -> check_k96tail_hub_ready.py
  -> wait_merge_finalize_k96tail.sh
  -> run_full_coupled_3p0625_campaign.sh
  -> run_finalize_k96tail_model.sh
  -> seal_coupled_k96tail_release.py
```

Each partial rental wave runs the ordinary no-shortcut controller with
`PARTIAL_ONLY=1`. Only the local finalizer assembles the full model and runs
end-to-end quality gates.
