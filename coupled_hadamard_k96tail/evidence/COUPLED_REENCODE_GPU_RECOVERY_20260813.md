# Coupled re-encode GPU recovery state — 2026-08-13

## Preserved campaign state

- Active wave: `wave-019-022`, exact K3/K4 triplet scoring.
- Last advancing receipt count: `579/1024` (`L19=152`, `L20=146`, `L21=155`, `L22=126`).
- All score artifacts are per-expert atomic outputs. Restarting the campaign skips validated existing receipts and recomputes only missing experts.
- Layers 3 through 18 remain fully materialized and runtime-oracle sealed.

## Driver failure evidence

- At 2026-08-13 23:52:53 EDT the kernel reported an NVIDIA GSP task watchdog timeout.
- Subsequent crash reports identify GPU 3 and repeat the same GSP program counter (`0x1bc70d8`).
- `nvidia-smi`, all 32 scorer CUDA processes, and `cosmic-comp` became blocked in NVIDIA kernel-driver wait paths.
- Receipt counts stopped advancing after 579, proving this is a driver-wide stall rather than a slow expert.
- GPU 3 was 78–81 C before the crash and reported no active hardware thermal slowdown.

## In-place recovery attempted

- PCI function reset for GPU 3 (`0000:c1:00.0`) returned success once, but did not unwind the NVIDIA global semaphore.
- Scorer processes and the compositor were sent termination signals; uninterruptible driver tasks could not exit.
- NVIDIA driver unbind and a second PCI reset both blocked and were force-terminated.
- A host reboot is therefore required to reload the NVIDIA kernel/GSP state.

## Automatic resume preparation

- Persistent unit source: `/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625/systemd/glm52-full-coupled-k96tail-resume.service`
- Installed unit: `/etc/systemd/system/glm52-full-coupled-k96tail-resume.service`
- Unit state: enabled for `multi-user.target`.
- The unit waits for the NVMe automount, Docker, and all four NVIDIA GPUs before launching the canonical campaign with `START_WAVE=3` and `STOP_WAVE=75`.
- The campaign's seals and atomic receipts make this a resume, not a restart from scratch.
