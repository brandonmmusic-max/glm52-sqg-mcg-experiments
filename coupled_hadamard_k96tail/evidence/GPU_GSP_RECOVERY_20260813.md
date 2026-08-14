# GPU GSP recovery checkpoint — 2026-08-13

- Authoritative goal: `019ffa7c-7914-7633-a7b8-a2f4b2e23fc0`.
- Wave `019-022` exact K3/K4 scoring started at `23:11:04 EDT` with 32
  workers (eight contexts per physical GPU).
- Last verified atomic receipt census before the driver incident:
  layer 19 = 152, layer 20 = 145, layer 21 = 155, layer 22 = 126;
  total = 578 / 1024.
- At `23:51:22 EDT` the NVIDIA 610.43.02 open kernel driver reported a GSP
  task-watchdog timeout. CUDA, NVML, NVIDIA modeset, and the COSMIC compositor
  subsequently blocked in the shared NVIDIA RM path. This was a driver/GSP
  failure, not a scorer exception.
- The campaign and its 32 containers were terminated. Completed expert JSON
  and row-SSE receipts are atomic and remain valid; the launchers validate and
  skip them on resume. No candidate selection or allocation had begun for this
  wave.
- Per-device PCI function resets did not restore the shared RM state. A clean
  host reboot is required.
- The persistent user service
  `glm52-full-coupled-k96tail-no-shortcut-goal019ffa7c.service` is installed in
  `~/.config/systemd/user/`, verified, enabled, and the user has linger enabled.
  It runs the authoritative campaign with `START_WAVE=3`, `STOP_WAVE=75`, and
  `CLEANUP_VALIDATED_WAVES=1`, so after boot it verifies/seals prior waves and
  resumes wave 019-022 from the existing receipts.
- Pinned saved inputs for wave 031-034 finished staging and have a 15-shard
  manifest plus four 12,897,349,632-byte capture tensors. Wave 035-038 has a
  resumable partial local directory (about 6.2 GB); do not count it complete
  until its normal manifest and required-file validation pass.
- Do not run any further background HF prefetch on the active score/encode
  path. Wave 031-034 was successfully staged, but the timing overlap obscured
  the onset of the GSP failure and does not justify further timed-path I/O.
