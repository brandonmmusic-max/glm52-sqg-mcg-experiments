# K96 coupled-Hadamard reproduction index

The authoritative live campaign contract is
`k96tail-distributed-campaign.json`. It remains incomplete until the final
assembled-model KLD, MTP3 smoke, and public model commit are sealed.

Method, orchestration, validation gates, incident history, current measured
ledger, and final-result placeholders are in
`../docs/K96_COUPLED_DISTRIBUTED_REPRODUCTION_20260814.md`.

Run the non-mutating audit from the project root:

```bash
python3 scripts/audit_k96tail_campaign.py
```

After the final receipt fields are populated and `complete` is set to true:

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
